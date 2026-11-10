"""Shared helpers for the training entries: run directory, CLI overrides, accelerator,
optimizer / LR groups, checkpoint loading and dataloader epoch handling."""

import copy
import os
import re
import threading
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import send_to_device
from PIL import Image, ImageOps
from transformers import get_scheduler

logger = get_logger(__name__)

# Default parent of every run directory (<run_root_dir>/<run_id>). scripts/train.sh and
# scripts/resolve_runtime_config.py use the same default.
DEFAULT_RUN_ROOT_DIR = "playground/Checkpoints"
# Subdirectory of the run directory that scripts/train.sh writes its logs into before the
# trainer starts; the trainer ignores it when checking that a fresh run directory is empty.
LAUNCHER_LOG_DIR = "logs"


def resolve_output_dir(cfg) -> str:
    """Set and return ``cfg.output_dir = <run_root_dir>/<run_id>``.

    ``run_id`` is required; ``run_root_dir`` defaults to ``playground/Checkpoints``. Both are
    top-level config keys and can be overridden with ``--run_id`` / ``--run_root_dir``.
    """
    run_id = cfg.get("run_id", None)
    if run_id in (None, ""):
        raise ValueError("The config must set a top-level `run_id` (or pass --run_id <name>).")
    if cfg.get("run_root_dir", None) in (None, ""):
        cfg.run_root_dir = DEFAULT_RUN_ROOT_DIR
    cfg.output_dir = os.path.join(str(cfg.run_root_dir), str(run_id))
    return cfg.output_dir


class DoubleBufferedWorkerIterator:
    """Overlap a bounded worker generation with preparation of the next one.

    This is an opt-in compatibility path for decoders that still require
    ``persistent_workers=False`` and periodic process recycling. The active
    iterator is drained completely; a shallow loader clone starts epoch N+1 near
    the end of epoch N, then a background thread waits for and holds its first CPU
    batch. Samples therefore retain the exact ``(epoch, index, seed)`` order while
    the initial worker prefetch overlaps model compute.
    """

    def __init__(self, dataloader, epoch_counter: int, prewarm_batches: int, reset_fn):
        self._loader = dataloader
        self._epoch = int(epoch_counter)
        self._prewarm_batches = int(prewarm_batches)
        self._reset_fn = reset_fn
        self._generation_batches = len(dataloader)
        if self._generation_batches <= 0:
            raise ValueError("double-buffered worker generation requires a non-empty dataloader.")
        if self._prewarm_batches <= 0 or self._prewarm_batches >= self._generation_batches:
            raise ValueError(
                "worker_generation_prewarm_batches must be in [1, generation_batches); "
                f"got {self._prewarm_batches} for {self._generation_batches} batches."
            )
        base = getattr(dataloader, "base_dataloader", dataloader)
        if getattr(base, "persistent_workers", False):
            raise ValueError("double-buffered worker generations require persistent_workers=false.")

        if callable(getattr(self._loader, "set_epoch", None)):
            self._loader.set_epoch(self._epoch)
        self._iterator = iter(self._loader)
        self._yielded = 0
        self._next_thread = None
        self._next_loader = None
        self._next_iterator = None
        self._next_epoch = None
        self._next_first_batch = None
        self._next_device = None
        self._next_non_blocking = False
        self._next_error = None

    @property
    def epoch(self) -> int:
        return self._epoch

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self._iterator)
        except StopIteration:
            batch = self._swap_generation()

        self._yielded += 1
        remaining = self._generation_batches - self._yielded
        if remaining <= self._prewarm_batches and self._next_thread is None and self._next_error is None:
            self._start_prewarm()
        return batch

    def _start_prewarm(self) -> None:
        # DataLoader starts multiprocessing workers from ``iter(loader)``.  Do
        # that on the training thread: forking a process from this background
        # thread can deadlock native decoder/tokenizer libraries which already
        # own threads in the parent.  Only the blocking first-batch read runs in
        # the background, after the worker processes are fully started.
        next_loader = copy.copy(self._loader)
        target_device = getattr(next_loader, "device", None)
        non_blocking = bool(getattr(next_loader, "_non_blocking", False))
        if target_device is not None:
            # Keep CUDA work on the training thread.  The held CPU batch is sent
            # to the original target device when the generation is swapped in.
            next_loader.device = None
        try:
            next_iterator, next_epoch = self._reset_fn(next_loader, self._epoch)
        except BaseException as exc:
            self._next_error = exc
            return

        self._next_loader = next_loader
        self._next_iterator = next_iterator
        self._next_epoch = next_epoch
        self._next_device = target_device
        self._next_non_blocking = non_blocking
        self._next_thread = threading.Thread(
            target=self._read_next_generation_first_batch,
            name=f"worker-prewarm-epoch-{self._epoch + 1}",
            daemon=True,
        )
        self._next_thread.start()

    def _read_next_generation_first_batch(self) -> None:
        try:
            self._next_first_batch = next(self._next_iterator)
            if self._next_device is not None:
                self._next_loader.device = self._next_device
        except BaseException as exc:  # re-raised on the training thread
            self._next_error = exc
            self._shutdown_iterator(self._next_iterator)

    def _swap_generation(self):
        if self._next_thread is None and self._next_error is None:
            self._start_prewarm()
        if self._next_thread is not None:
            self._next_thread.join()
        if self._next_error is not None:
            raise RuntimeError(f"background worker generation {self._epoch + 1} failed") from self._next_error
        if self._next_iterator is None or self._next_first_batch is None:
            raise RuntimeError(f"background worker generation {self._epoch + 1} produced no batch")

        self._loader = self._next_loader
        self._iterator = self._next_iterator
        self._epoch = int(self._next_epoch)
        batch = self._next_first_batch
        if self._next_device is not None:
            batch = send_to_device(batch, self._next_device, non_blocking=self._next_non_blocking)

        self._yielded = 0
        self._next_thread = None
        self._next_loader = None
        self._next_iterator = None
        self._next_epoch = None
        self._next_first_batch = None
        self._next_device = None
        self._next_non_blocking = False
        self._next_error = None
        return batch

    def close(self) -> None:
        """Release active and prefetched iterator generations."""

        if self._next_thread is not None:
            self._next_thread.join()
        for iterator in (self._iterator, self._next_iterator):
            self._shutdown_iterator(iterator)
        self._iterator = None
        self._next_iterator = None
        self._next_first_batch = None
        self._next_loader = None
        self._next_thread = None

    @staticmethod
    def _shutdown_iterator(iterator) -> None:
        if iterator is None:
            return
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
            return
        shutdown_workers = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown_workers):
            shutdown_workers()


def mean_action_mse(predicted_actions, target_actions) -> float:
    """Return the element-wise action MSE after validating tensor compatibility."""
    predicted = np.asarray(predicted_actions)
    target = np.asarray(target_actions)
    if predicted.shape != target.shape:
        raise ValueError(
            "predicted and target action shapes must match; " f"got predicted={predicted.shape}, target={target.shape}"
        )
    value = float(np.mean(np.square(predicted - target)))
    if not np.isfinite(value):
        raise FloatingPointError(f"action MSE is non-finite: {value!r}")
    return value


def normalize_dotlist_args(args):
    """
    Convert ['--x.y', 'val'] and ['--flag'] → ['x.y=val', 'flag=true']
    """
    normalized = []
    skip = False
    for i in range(len(args)):
        if skip:
            skip = False
            continue

        arg = args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            if "=" in key:
                normalized.append(key)
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                normalized.append(f"{key}={args[i + 1]}")
                skip = True
            else:
                normalized.append(f"{key}=true")
        else:
            pass  # skip orphaned values
    return normalized


def resolve_gradient_accumulation(cfg):
    """Derive DeepSpeed gradient-accumulation steps from a target GLOBAL batch size.

    By default, gradient accumulation is rounded from
    ``global_batch_size / (per_device_batch_size * world_size)``. With
    ``trainer.strict_global_batch=true``, the requested global batch must instead be an
    exact positive multiple of the per-step distributed microbatch. ``world_size`` comes
    from WORLD_SIZE (set by ``accelerate launch`` and equal to ``accelerator.num_processes``).

    Configs without `trainer.global_batch_size` use the explicit
    `trainer.gradient_accumulation_steps` (default 1).

    Returns (grad_accum, effective_global_bs_or_None, world_size).
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_bs = cfg.trainer.get("global_batch_size", None)
    if global_bs is None:
        return int(cfg.trainer.get("gradient_accumulation_steps", 1)), None, world_size
    # Per-device samples per micro-step, summed over all co-training streams.
    micro = cfg.datasets.get("total_per_device_batch_size", None)
    if micro is None:
        micro = cfg.datasets.vla_data.per_device_batch_size
    micro = int(micro)
    global_bs = int(global_bs)
    denominator = micro * world_size
    strict_global_batch = bool(cfg.trainer.get("strict_global_batch", False))
    if strict_global_batch and (global_bs <= 0 or global_bs % denominator != 0):
        raise ValueError(
            "trainer.strict_global_batch=true requires global_batch_size to be "
            "an exact positive multiple of per_device_batch_size * world_size; "
            f"got global_batch_size={global_bs}, per_device_batch_size={micro}, "
            f"world_size={world_size}, minimum/exact unit={denominator}."
        )
    grad_accum = max(1, round(global_bs / denominator))
    return grad_accum, grad_accum * micro * world_size, world_size


def build_accelerator(cfg):
    """Construct the Accelerator with grad-accum derived from cfg.trainer.global_batch_size.

    Requires ds_config's `gradient_accumulation_steps: "auto"`: a concrete value in the
    ds_config file would override the plugin and silently ignore the global-batch target.
    """
    grad_accum, effective, world_size = resolve_gradient_accumulation(cfg)
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        deepspeed_plugin=DeepSpeedPlugin(gradient_accumulation_steps=grad_accum),
    )
    assert (
        accelerator.num_processes == world_size
    ), f"WORLD_SIZE({world_size}) != accelerator.num_processes({accelerator.num_processes})"
    if accelerator.is_main_process:
        requested = cfg.trainer.get("global_batch_size", None)
        logger.info(
            f"[grad-accum] world_size={world_size}, global_batch_size={requested}, "
            f"gradient_accumulation_steps={grad_accum}, effective_global={effective}"
        )
        if effective is not None and requested is not None and int(effective) != int(requested):
            logger.warning(
                f"[grad-accum] requested global_batch_size={requested} not exactly achievable "
                f"with per_device*world_size; rounded to grad_accum={grad_accum} -> "
                f"effective global={effective}."
            )
    return accelerator


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Build fused AdamW over the per-module LR groups plus the configured LR scheduler.

    The schedule spans ``trainer.resolved_max_train_steps`` with
    ``trainer.resolved_num_warmup_steps`` warmup steps; the training entries resolve both
    (from absolute values or ratios) before calling this.
    """
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=True,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    sched_kwargs = {k: v for k, v in cfg.trainer.scheduler_specific_kwargs.items()}
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.resolved_num_warmup_steps,
        num_training_steps=cfg.trainer.resolved_max_train_steps,
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


def build_param_lr_groups(model, cfg):
    """
    build multiple param groups based on cfg.trainer.learning_rate.
    support specifying different learning rates for different modules, the rest use base.

    Args:
        vla: nn.Module model object
        cfg: config object, requires cfg.trainer.learning_rate dictionary

    Returns:
        List[Dict]: param_groups that can be used to build optimizer with torch.optim
    """

    lr_cfg = cfg.trainer.learning_rate
    base_lr = lr_cfg.get("base", 1e-4)  # default base learning rate

    freeze_modules = cfg.trainer.get("freeze_modules", "")
    if not isinstance(freeze_modules, str):
        freeze_modules = ""
    freeze_patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()]

    used_params = set()
    frozen_params = set()
    param_groups = []

    for freeze_path in freeze_patterns:
        module = model
        try:
            for attr in freeze_path.split("."):
                module = getattr(module, attr)
            frozen_params.update(id(p) for p in module.parameters())
        except AttributeError:
            print(f"⚠️ freeze module path does not exist: {freeze_path}")
            continue

    for module_name, lr in lr_cfg.items():
        if module_name == "base":
            continue
        # try to find the module under vla by module_name (support nested paths)
        module = model
        try:
            for attr in module_name.split("."):
                module = getattr(module, attr)
            # filter out frozen parameters (explicit freeze list or requires_grad=False)
            params = [p for p in module.parameters() if id(p) not in frozen_params and p.requires_grad]
            if params:  # only add param group if there are trainable parameters
                param_groups.append({"params": params, "lr": lr, "name": module_name})
                used_params.update(id(p) for p in params)
        except AttributeError:
            # A learning_rate key that matches no module would silently put its
            # params into the `base` group; warn so a misspelled name is caught.
            print(
                f"⚠️ trainer.learning_rate: module path `{module_name}` not found on the model; "
                f"its params (if any) fall into the `base` LR group. Check the attribute name."
            )

    # assign base learning rate to the remaining unused parameters (exclude
    # frozen ones and any param with requires_grad=False, e.g. a backbone frozen
    # inside the framework __init__).
    other_params = [
        p for p in model.parameters() if id(p) not in used_params and id(p) not in frozen_params and p.requires_grad
    ]
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "name": "base"})

    return param_groups


def resize_images(images, target_size=(224, 224)):
    """Recursively resize every image in a (possibly nested) list to ``target_size``.

    Uses aspect-preserving resize-to-cover + center-crop (``ImageOps.fit``), NOT a plain
    stretch — the same recipe as ``view_tiling.resize_cover_crop`` and the ``auto``
    gen-target encoder, so the observation preprocessing is consistent across the
    dataloader (training) and every framework's inference path.

    :param images: nested list of PIL images, or a single PIL image.
    :param target_size: ``(width, height)`` the images are cover+cropped to.
    :return: resized images, keeping the original nested structure.
    """
    if isinstance(images, Image.Image):  # a single PIL image
        w, h = int(target_size[0]), int(target_size[1])
        return ImageOps.fit(images.convert("RGB"), (w, h), method=Image.Resampling.BICUBIC, centering=(0.5, 0.5))
    elif isinstance(images, list):  # recurse over the nested structure
        return [resize_images(img, target_size) for img in images]
    else:
        raise ValueError("Unsupported image type or structure.")


class TrainerUtils:
    @staticmethod
    def freeze_backbones(model, freeze_modules=""):
        """Freeze the submodules listed in ``trainer.freeze_modules``.

        ``freeze_modules`` is a comma-separated list of attribute paths relative to the model,
        e.g. ``"qwen_vl_interface, action_model.net"`` freezes ``model.qwen_vl_interface`` and
        ``model.action_model.net``. Unknown paths are skipped with a warning.

        Returns:
            model: the same model, with ``requires_grad=False`` on the frozen parameters.
        """
        frozen = []
        if freeze_modules and isinstance(freeze_modules, str):
            # split and remove whitespace
            patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()] if freeze_modules else []

            for path in patterns:
                # split the "relative path" by dots, for example "action_model.net" → ["action_model", "net"]
                attrs = path.split(".")
                module = model
                try:
                    for attr in attrs:
                        module = getattr(module, attr)
                    # if the module is successfully get, freeze it and its all submodule parameters
                    for param in module.parameters():
                        param.requires_grad = False
                    frozen.append(path)
                except AttributeError:
                    # if the attribute does not exist, skip and print warning
                    print(f"⚠️ module path does not exist, cannot freeze: {path}")
                    continue

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"🔒 Frozen modules: {frozen}")
        return model

    @staticmethod
    def print_trainable_parameters(model):
        """
        print the total number of parameters and trainable parameters of the model
        :param model: PyTorch model instance
        """
        if dist.get_rank() != 0:
            return
        print("📊 model parameter statistics:")
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
        )
        return num_params, num_trainable_params

    @staticmethod
    def load_pretrained_backbones(model, checkpoint_path=None, reload_modules=None):
        """
        load checkpoint:
        - if reload_modules is set, load by path part
        - otherwise → load the entire model parameters (overwrite model)

        return:
            model: the model with the checkpoint parameters loaded
        """
        if not checkpoint_path:
            return model
        if dist.get_rank() == 0:
            print(f"📦 loading checkpoint: {checkpoint_path}")
        try:
            if _is_safetensors_path(checkpoint_path):
                from safetensors.torch import load_file

                checkpoint = load_file(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"❌ loading checkpoint failed: {e}") from e

        loaded_modules = []

        if reload_modules:  # partial load
            module_paths = [p.strip() for p in reload_modules.split(",") if p.strip()]
            for path in module_paths:
                reload_modules = path.split(".")
                module = model
                try:
                    for module_name in reload_modules:  # find the module to modify level by level
                        module = getattr(module, module_name)
                    prefix = path + "."
                    sub_state_dict = {k[len(prefix) :]: v for k, v in checkpoint.items() if k.startswith(prefix)}
                    if sub_state_dict:
                        module.load_state_dict(sub_state_dict, strict=True)
                        if dist.get_rank() == 0:
                            print(f"✅ parameters loaded to module '{path}'")
                        loaded_modules.append(path)
                    else:
                        print(f"⚠️ parameters not found in checkpoint '{path}'")
                except AttributeError:
                    print(f"❌ cannot find module path: {path}")
        else:  # full load
            try:
                model.load_state_dict(checkpoint, strict=False)
                if dist.get_rank() == 0:
                    print("✅ loaded <full_model> model parameters")
                loaded_modules = ["<full_model>"]
            except Exception as e:
                raise RuntimeError(f"❌ loading full model failed: {e}") from e
        return model

    @staticmethod
    def setup_distributed_training(accelerator, *components):
        """
        use Accelerator to prepare distributed training components
        :param accelerator: Accelerate instance
        :param components: any number of components (such as model, optimizer, dataloader, etc.)
        :return: prepared distributed components (in the same order as input)
        """

        # use accelerator.prepare method to wrap components
        prepared_components = accelerator.prepare(*components)
        return prepared_components

    @staticmethod
    def _reset_dataloader(dataloader, epoch_counter):
        """Safely reset a dataloader iterator at an epoch boundary, reshuffling.

        Advances the epoch so the NEXT epoch draws a fresh sample order. This matters
        for StarVLA's map-style datasets whose shuffling is internal:
        LeRobotMixtureDataset.sample_step and VideoClipDataset reseed their sampling
        hash from ``dataset.epoch``. Without a bump, every epoch replays the SAME
        (episode, start-frame) sequence — which, on an episode_fraction-restricted
        subset, would mean re-feeding one fixed sequence instead of true multi-epoch.

        Two accelerate/torch subtleties:
          * accelerate's ``DataLoaderShard`` overwrites the dataset epoch on every
            ``__iter__`` via ``self.set_epoch(self.iteration)``. So the epoch must be
            set through the shard's OWN ``set_epoch`` (which also updates ``iteration``);
            poking ``dataset.set_epoch`` directly is clobbered on the next ``iter()``.
          * With ``persistent_workers=True`` the supported StarVLA datasets publish
            scalar epoch changes through ``SharedEpoch``. Reusing the pool is therefore
            both correct and materially faster than destroying it at every boundary.
            Transition-permutation datasets additionally replace a large order array;
            they opt into a boundary respawn so workers inherit that exact array.
        """
        # 1. update epoch counter
        epoch_counter += 1

        # 2. push the new epoch as far down as the object exposes.
        #    DataLoaderShard.set_epoch fans out to batch_sampler / sampler / dataset;
        #    a plain DataLoader has no set_epoch, so fall back to sampler + dataset.
        if callable(getattr(dataloader, "set_epoch", None)):
            dataloader.set_epoch(epoch_counter)  # accelerate shard: also sets .iteration
        else:
            if hasattr(dataloader, "sampler") and callable(getattr(dataloader.sampler, "set_epoch", None)):
                dataloader.sampler.set_epoch(epoch_counter)
            dataset = getattr(dataloader, "dataset", None)
            if dataset is not None and callable(getattr(dataset, "set_epoch", None)):
                dataset.set_epoch(epoch_counter)

        # 3. A transition-permutation epoch carries a newly built immutable order
        #    array in the trainer process, not just the shared scalar. Respawn only
        #    that dataset's persistent pool so worker copies inherit it. All other
        #    persistent loaders reset and reuse their existing workers.
        base = getattr(dataloader, "base_dataloader", dataloader)
        dataset = getattr(base, "dataset", getattr(dataloader, "dataset", None))
        if (
            getattr(base, "persistent_workers", False)
            and getattr(dataset, "persistent_epoch_requires_respawn", False)
            and getattr(base, "_iterator", None) is not None
        ):
            base._iterator = None

        # 4. create the next iterator. A non-persistent loader (and the explicit
        #    permutation case above) creates a fresh worker generation.
        return iter(dataloader), epoch_counter

    def _get_latest_checkpoint(self, checkpoint_dir):
        """Find the latest checkpoint in the directory based on step number."""
        if not os.path.exists(checkpoint_dir):
            self.accelerator.print(f"No checkpoint directory found at {checkpoint_dir}")
            return None, 0

        # Find all checkpoints matching the naming convention, supports .pt and .safetensors
        checkpoints = [
            f
            for f in os.listdir(checkpoint_dir)
            if re.match(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$", f)
            and os.path.isfile(os.path.join(checkpoint_dir, f))  # ensure it is a file
        ]

        if not checkpoints:
            self.accelerator.print(f"No checkpoints found in {checkpoint_dir}")
            return None, 0

        # Extract step numbers and sort
        try:
            checkpoints_with_steps = [
                (ckpt, int(re.search(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$", ckpt).group(1)))
                for ckpt in checkpoints
            ]
        except AttributeError as e:
            self.accelerator.print(f"Error parsing checkpoint filenames: {e}")
            return None, 0

        # Sort by step number and get the latest checkpoint
        checkpoints_with_steps.sort(key=lambda x: x[1])
        latest_checkpoint, completed_steps = checkpoints_with_steps[-1]

        latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        self.accelerator.print(f"Latest checkpoint found: {latest_checkpoint_path}")
        return latest_checkpoint_path, completed_steps


def _is_safetensors_path(path):
    """Check if a path refers to a safetensors file."""
    return str(path).endswith(".safetensors")

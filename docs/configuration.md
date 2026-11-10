# Configuration reference

A training run is described by one OmegaConf tree. This page lists the keys used
by the released recipes in [`configs/`](../configs/README.md), what they mean,
and which ones are derived automatically. Defaults come from
`QwenPIDefaultConfig` (`starVLA/model/framework/VLM4A/QwenPI.py`) and
`_GEN_TARGET_DEFAULTS` (`starVLA/model/modules/gen_model/__init__.py`); a YAML
file only needs the keys it changes.

## Loading pipeline

Both training entry points process the config in the same order:

```
load_config_with_bases(config_yaml)   # expand `_base_` (bases merged in order, the file wins; lists are replaced)
  -> merge CLI overrides               # `--dotted.key value`, the last occurrence wins
  -> apply_config_compat(cfg)          # idempotent normalisation (e.g. action_horizon <-> future_action_window_size)
  -> align_dataset_configs(cfg)        # propagate datasets.shared, derive per-stream alignment keys (see below)
  -> per-stream batch sizes            # split datasets.total_per_device_batch_size by sampling_ratio
  -> build_framework(cfg)              # framework defaults are filled in here (DiT shapes, input_dim, ...)
```

The resolved config is written to the run directory and is what
`baseframework.from_pretrained` uses to rebuild the model at evaluation time.

## Top level

| Key | Meaning |
|---|---|
| `run_id` | Run name; outputs go to `<run_root_dir>/<run_id>/`. |
| `train_entry` | Training script: `starVLA/training/train_starvla.py` (one robot stream) or `starVLA/training/train_starvla_gen_cotrain.py` (several streams, e.g. robot + egocentric video). |
| `seed` | Global seed. |
| `version_id` | Config schema version (`"0.22"`). |
| `wandb_project`, `wandb_entity` | Weights & Biases logging. |

## `framework`

| Key | Default | Meaning |
|---|---|---|
| `name` | `QwenPI` | Framework registry name. |
| `freeze_vlm` | `false` | Freeze the whole VLM. With the `vlm` generation backbone the learnable queries stay trainable. |
| `freeze_visual_encoder` | `true` | Keep the vision tower frozen regardless of `freeze_vlm` (this also keeps the `auto` generation target fixed). |
| `qwenvl.base_vlm` | `playground/Pretrained_models/Qwen3.5-2B` | VLM checkpoint; the path must contain `Qwen3.5`. |
| `qwenvl.attn_implementation` | `sdpa` | The Qwen3.5 wrapper always uses `sdpa`. |
| `qwenvl.require_gated_delta_fast_path` | — | Fail if the gated-delta kernels (`flash-linear-attention`, `causal-conv1d`) are missing instead of falling back to the slow path. |
| `qwenvl.max_pixels` | `null` | Qwen dynamic-resize pixel budget per image; required with `obs_resize_mode: native`. |

### Action expert (`framework.action_model`)

Layer-wise flow matching: one DiT block per LLM layer, each cross-attending to
that layer's hidden states.

| Key | Default | Meaning |
|---|---|---|
| `enable` | `true` | Build the action expert. |
| `action_dim`, `state_dim` | `7`, `7` | Action / proprioception size (`include_state: false` in all recipes). |
| `action_horizon` | `16` (recipes: `8`) | Action chunk length; the single source of truth, propagated to every data stream. |
| `repeated_diffusion_steps` | `2` | (time, noise) draws per sample, VLM features reused. |
| `num_inference_timesteps` | `4` | Euler steps at inference. |
| `num_target_vision_tokens` | `32` | Learnable tokens prepended to the action tokens. |
| `noise_beta_alpha`, `noise_beta_beta`, `noise_s` | `1.5`, `1.0`, `0.999` | Beta time-step distribution of flow matching. |
| `diffusion_model_cfg.*` | see `QwenPIDefaultConfig` | Non-shape DiT settings; the shape (layers, width, heads) is derived from the VLM. |

### Generation expert (`framework.gen_model`)

| Key | Default | Meaning |
|---|---|---|
| `enable` | `false` | Build the generation head. |
| `mode` | `regression` | `regression`: learnable queries regressed onto target features with a pointwise loss. `diffusion`: the generation expert is a flow-matching denoiser of the target features (forces `backbone: expert` and `loss: flow_matching`). |
| `backbone` | `vlm` | `vlm`: learnable queries appended to the VLM sequence. `expert`: a separate expert reading the LLM's layer-wise hidden states (used by every recipe). |
| `isolation` | `auto` | Whether the action expert may attend to the generation states. `auto` = isolated for the `expert` backbone, coupled for `vlm`. With isolation the generation branch is dropped at inference. |
| `frame_mode` | `parallel` | All `T` target frames are predicted at once. |
| `gen_horizon_seconds` | `null` | τ: how far ahead the target lies, in seconds. `null` aligns the target with the end of the action chunk (a<sub>t+H</sub>, H robot frames ahead), τ = `action_horizon` / `datasets.vla_data.dataset_fps` (see *Derived keys*). |
| `num_future_frames` | `1` | Number of future frames T predicted within τ (frame k at k·τ/T). |
| `normalize_target` | `false` | Regression with `mse` / `smooth_l1`: standardise the target as `(Z - mean) / std` with scalar per-encoder statistics (`target.feat_mean` / `feat_std`; calibrated defaults per encoder, overridable). `true` in every regression recipe; diffusion targets are not standardized. |
| `targets` | one `auto` target | List of targets, each `{target, decoder, loss, num_gen_queries, weight}`; missing fields are filled from `_GEN_TARGET_DEFAULTS`. With several targets the generation loss is their weighted mean. |
| `expert_freeze`, `expert_add_pos_embed`, `max_seq_len`, `expert_dit_cfg.*` | see `QwenPIDefaultConfig` | Generation-expert DiT settings; the shape is derived from the VLM. |

Per-target fields (`framework.gen_model.targets[i]`):

| Key | Default | Meaning |
|---|---|---|
| `target.type` | `auto` | Frozen encoder producing the target: `auto` (the VLM's own vision tower, "Qwen-ViT"), `siglip`, `dino`, `vjepa`, `lingbot`, `wan_vae`, `pixel`. |
| `target.path` | `null` | Encoder weights; required for every type except `auto` and `pixel`. |
| `target.vae_subfolder` | `null` | Sub-folder of a diffusers repository holding the VAE (`wan_vae`). |
| `target.resolution` | `224` | Resize of the future frame before encoding. |
| `target.pool` | `none` | `target_to_query` average-pools target tokens to `num_gen_queries` (needed when the encoder emits more tokens than queries, e.g. V-JEPA2). |
| `target.variant` | — | LingBot-Vision variant (`giant`). |
| `decoder` | `{type: mlp, num_layers: 3}` | Regression decoder (only `mlp`); unused in `diffusion` mode. |
| `loss.type` | `mse` | Regression: `mse`, `cosine`, `cosine_magnitude`, `smooth_l1`. Diffusion: `flow_matching`. |
| `num_gen_queries` | `auto` | Number of query tokens; `auto` uses the target's token count. |
| `weight` | `1.0` | Weight of this target when several are trained. |

## `datasets`

`datasets.total_per_device_batch_size` is the per-GPU micro-batch summed over all
streams; it is split between the streams in proportion to their
`sampling_ratio`.

### `datasets.shared`

Copied into every stream and always wins over per-stream values; these keys
must agree across streams because one model consumes all of them.

| Key | Recipes | Meaning |
|---|---|---|
| `view_prompt_mode` | `role` | Every image is preceded by a description of its camera role. |
| `gen_video_key` | `central_global` | Camera role used as the generation target. |
| `obs_resize_mode`, `obs_image_size` | `native`, `null` | `native` keeps the native resolution within `qwenvl.max_pixels`; `fixed` cover-crops to `obs_image_size`. |
| `view_tiling` | `"off"` | `tile` composes the views into one 2x2 canvas. |
| `gen_augment.*`, `gen_target_clean` | disabled | Optional photometric jitter of the observation (and, unless `gen_target_clean`, the target). |

### Per stream (`datasets.vla_data`, `datasets.video_data`)

Any block under `datasets` other than `shared` is a data stream. The released
recipes use `vla_data` (robot) and, for co-training, `video_data` (egocentric
video). `vla_data` is the anchor stream from which the generation head takes its
target shape.

| Key | Meaning |
|---|---|
| `dataset_py` | Loader: `lerobot_datasets` (actions only), `gen_lerobot_datasets` (actions + future frames), `video_datasets` (egocentric video, future frames only). |
| `data_root_dir` | Dataset root. |
| `data_mix` | LeRobot mixture name, registered in `examples/*/train_files/data_registry/data_config.py` (`droid_success`, `libero_all`, ...). |
| `lerobot_version`, `modality_json_path` | LeRobot layout (`v3.0` for DROID) and modality description. |
| `episode_filter.{require_instruction, require_motion}` | LeRobot v3 robot data: drop episodes without a non-empty instruction, with missing / non-finite action statistics, or without motion, using the episode metadata (DROID curation). |
| `index_jsonl`, `text_key`, `view_description` | Egocentric streams: the clip index (format in [`data.md`](data.md)), the caption field and the description of the single camera view. |
| `episode_fraction` | Robot streams: deterministic fraction of the episodes; egocentric streams: seeded fraction of the clips (data-scaling study). |
| `dataset_fps` | Recording frame rate of this stream (required for generation). |
| `losses` | Losses trained on this stream: `[action_loss, gen_loss]` for robot data, `[gen_loss]` for video. |
| `sampling_ratio` | Share of the per-GPU micro-batch. |
| `val_episode_fraction`, `split`, `seed` | Deterministic episode hold-out used for validation. |
| `sampling_strategy`, `balance_dataset_weights`, `balance_trajectory_weights` | Sampling order and weighting (`transition_permutation`: a without-replacement pass over all transitions). |
| `num_workers`, `prefetch_factor`, `persistent_workers`, `worker_preprocess_qwen`, `video_backend` | Loader performance settings. |

### Derived keys (do not set)

`align_dataset_configs` writes these into every stream; values set by hand are
overwritten with a warning.

- `action_horizon` from `framework.action_model.action_horizon`.
- `gen_future_offset`, `gen_stride_frames`, `num_target_frames` from
  the horizon τ, `num_future_frames` (T) and the stream's own `dataset_fps` (f):
  the target lies ⌊τ·f + ½⌋ frames ahead and frame k of T at ⌊k·τ·f/T + ½⌋.
  τ defaults to the end of the action chunk on the robot clock,
  `action_horizon / datasets.vla_data.dataset_fps`: 8 / 15 Hz = 0.533 s with DROID
  (8 DROID frames, 16 frames of a 30 fps egocentric video) and 8 / 20 Hz = 0.4 s with
  LIBERO. Every stream therefore predicts the same physical instants and the same
  number of frames.
- Per-stream batch sizes from `total_per_device_batch_size` and `sampling_ratio`.

## `trainer`

| Key | Meaning |
|---|---|
| `max_train_steps`, `num_warmup_steps` | Schedule length and linear warm-up. |
| `lr_scheduler_type`, `scheduler_specific_kwargs.min_lr` | `cosine_with_min_lr` and its floor. |
| `learning_rate.{base, qwen_vl_interface, action_model, gen_model}` | Per-module learning rates (`base` covers parameters no other group claims; frozen parameters are excluded). |
| `loss_scale.{action_loss, gen_loss}` | Loss weights; the total loss is `sum(scale * loss)`. |
| `optimizer.{name, betas, eps, weight_decay}` | AdamW settings. |
| `global_batch_size`, `strict_global_batch` | Global batch per optimizer step. Gradient accumulation is derived from it; with `strict_global_batch` it must be divisible by `micro_batch x num_gpus`. |
| `gradient_clipping`, `gradient_checkpointing` | Gradient-norm clipping (1.0; with DeepSpeed it is set in `starVLA/config/deepseeds/ds_config.yaml`) and activation checkpointing. |
| `freeze_modules` | Comma-separated module paths to freeze (prefer `framework.freeze_vlm`). |
| `save_interval`, `save_total_limit`, `save_full_state` | Checkpoint cadence, retention (`-1` keeps all) and whether optimizer / scheduler / data state is saved for resuming. |
| `eval_interval`, `val_num_batches`, `logging_frequency` | Validation and logging cadence. |
| `cotrain_mode` | `joint`: every step combines one mini-batch per stream; each mini-batch computes only its bound losses (averaged within the mini-batch), the losses are summed and their gradients accumulate into one optimizer step. |
| `fuse_joint_vlm_forward` | Optional (default `false`): encode all streams' mini-batches in one padded VLM forward instead of one forward per stream. A speed optimisation; the recipes forward the streams separately. |
| `pretrained_checkpoint`, `transfer_policy` | Fine-tuning: pre-training checkpoint and what to transfer: `vlm_only` (VLM; action expert re-initialized, the default), `full` (VLM + action expert), `action_core` (`full` except the action expert's noisy-action input and velocity output layers). |
| `transfer_retain_gen_model` | With `vlm_only`: also transfer the pre-trained generation expert (default in `configs/finetune/`). |

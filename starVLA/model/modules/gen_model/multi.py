# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Multi-target composite generation head.

A single generation head trains ONE generation loss against ONE target.  This
composite lets a run train against SEVERAL targets at once under the SAME
generation ``mode``: it builds one child head per target, each with its own group
of learnable queries (``vlm`` backbone) or its own generation expert (``expert``
backbone), sized to that target, and aggregates the N per-target losses with
per-target weights::

    framework.gen_model:
      mode: regression                 # shared by every target
      backbone: expert                 # shared by every target
      targets:
        - target: {type: auto, resolution: 224}
          weight: 1.0
        - target: {type: siglip, path: playground/Pretrained_models/siglip2-so400m-patch16-256}
          weight: 0.5

Each list entry is a PARTIAL ``gen_model`` config: every key except ``weight``
is deep-merged over the shared ``gen_model`` block to form that child's config,
so a target sets its own ``target`` / ``decoder`` / ``loss`` / ``num_gen_queries``
while inheriting ``mode`` / ``backbone`` / ``isolation`` / ``input_dim`` / expert
shape from the shared block.

The composite subclasses :class:`BaseGenHead` and presents the exact same surface
the framework talks to (``build`` / ``forward`` / ``isolation_context`` /
``requires_vlm_queries`` / ``requires_vl_embs`` / ``backbone``), so the framework
stays multi-target-agnostic. A single-target config is a 1-element composite whose
weighted-mean loss equals the plain single loss; EVERY generation config builds
through this class, so the child weights live under ``gen_model.heads.<i>.*`` in
the state dict.
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig, OmegaConf

from starVLA.model.modules.gen_model.base import BaseGenHead
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


def _raw(cfg):
    """Underlying OmegaConf node, unwrapping an AccessTrackedConfig if present."""
    return cfg._cfg if hasattr(cfg, "_cfg") else cfg


def _snapshot_touch(node):
    """Access every leaf of a (possibly access-tracked) config subtree.

    The training snapshot (``config.yaml``) records only keys that were read, so
    ``from_pretrained`` can rebuild the model.  Children are built from an
    UNTRACKED deep copy of the config, so without this the ``targets`` subtree
    would never be marked accessed and would be dropped from the snapshot.  This
    forces the whole ``targets`` subtree into the snapshot.  No-op on a plain
    (untracked) OmegaConf config.
    """
    cfg = getattr(node, "_cfg", None)
    if cfg is None:
        return  # plain OmegaConf (e.g. the __main__ smoke test): nothing to track
    if isinstance(cfg, ListConfig):
        for i in range(len(cfg)):
            _snapshot_touch(node[i])
    elif isinstance(cfg, DictConfig):
        for k in list(cfg.keys()):
            _snapshot_touch(node.get(k))


class MultiGenHead(BaseGenHead):
    """Composite head: one child head per target, weighted MEAN of their losses.

    The aggregate ``gen_loss`` is ``sum(w_i * loss_i) / sum(w_i)`` (normalized by the
    total weight), so its scale does not grow with the number of targets — the outer
    ``trainer.loss_scale`` keeps setting the gen:action ratio and the per-target
    weights only split the gen budget internally.
    """

    def __init__(self, global_config, vlm_interface=None):
        super().__init__(global_config, vlm_interface=vlm_interface)
        from starVLA.model.modules.gen_model import _build_single_gen_head

        gen_cfg = global_config.framework.gen_model
        targets_node = gen_cfg.get("targets", None)
        _snapshot_touch(targets_node)  # keep the whole targets subtree in config.yaml

        # normalize_gen_config (run once on the parent in get_gen_model) guarantees a
        # materialised, non-empty `targets` list — single-target is a 1-element list.
        raw_full = _raw(global_config)
        raw_gen = raw_full.framework.gen_model
        # Shared child base = the gen_model block WITHOUT the targets list (only the shared
        # axes: mode/backbone/isolation/frame_mode/input_dim/expert_*). The per-target
        # target/decoder/loss/num_gen_queries come from each (already default-filled) entry.
        base_gen = OmegaConf.create(OmegaConf.to_container(raw_gen, resolve=False))
        base_gen.pop("targets", None)

        entries = OmegaConf.to_container(_raw(targets_node), resolve=False)
        if isinstance(entries, dict):
            # Defensive: an accessed-only config snapshot can round-trip the list as an
            # int-keyed map ({0: {...}, 1: {...}}); coerce back to index order.
            entries = [entries[k] for k in sorted(entries, key=lambda x: int(x))]
        if not isinstance(entries, list) or len(entries) == 0:
            raise ValueError("framework.gen_model.targets must be a non-empty list of target specs.")

        heads, weights = [], []
        for entry in entries:
            entry = dict(entry)
            weights.append(float(entry.pop("weight", 1.0)))
            # child_gen = shared axes + this entry's target/decoder/loss at the child's top
            # level, so the mode head reads gen_cfg.target/.decoder/.loss unchanged.
            child_gen = OmegaConf.merge(base_gen, OmegaConf.create(entry))
            # Full untracked copy of the config with this child's single-target gen_model
            # swapped in — the child head reads framework.qwenvl / datasets.vla_data from here.
            child_global = OmegaConf.create(OmegaConf.to_container(raw_full, resolve=False))
            child_global.framework.gen_model = child_gen
            heads.append(_build_single_gen_head(child_global, vlm_interface))

        self.heads = nn.ModuleList(heads)
        self._weights = weights
        # Normalizer for the loss aggregation: gen_loss is the WEIGHTED MEAN (not sum)
        # of the per-target losses, so its magnitude is invariant to the number/weight
        # of targets. That keeps trainer.loss_scale.gen_loss in control of the
        # gen:action ratio, while the per-target weights only control the split WITHIN
        # gen (see forward()).
        self._weight_sum = float(sum(weights))
        if self._weight_sum <= 0:
            raise ValueError(f"multi-target: per-target weights must sum to a positive value, got {weights}.")
        # Every child shares ONE backbone (a shared axis: mode/backbone/isolation are
        # not per-target). Enforce it — a stray per-target `backbone:` override would
        # otherwise silently mis-drive the vlm-vs-expert build/coupling dispatch below.
        backbones = {str(h.backbone).lower() for h in self.heads}
        if len(backbones) != 1:
            raise ValueError(
                f"multi-target requires a single shared backbone across all targets, got {backbones}. "
                "Set framework.gen_model.backbone (not a per-target override)."
            )
        self.backbone = backbones.pop()
        self._vlm_slices = None  # [(offset, size)] per child, set for the vlm backbone
        logger.info(
            f"[MultiGenHead] {len(heads)} targets weights={weights} "
            f"backbone={self.backbone} mode={gen_cfg.get('mode')}"
        )

    # ── build ──
    def build(self, vlm_interface):
        if self.backbone == "vlm":
            # ONE registration of all groups: [<boi>, group_0…, group_1…, <eoi>].
            sizes = [int(h._resolve_num_queries()) for h in self.heads]
            vlm_interface.setup_gen_queries(sizes)
            self._vlm_slices = []
            offset = 0
            for h, n in zip(self.heads, sizes, strict=False):
                h._write_num_queries(n)
                h.requires_vlm_queries = True
                h._apply_freezes()
                self._vlm_slices.append((offset, n))
                offset += n
            self.requires_vlm_queries = True
        else:
            # Expert backbone (incl. diffusion): each child builds its own expert.
            for h in self.heads:
                h.build(vlm_interface)
            self.requires_vl_embs = any(bool(h.requires_vl_embs) for h in self.heads)

    # ── forward: weighted MEAN of the per-target losses ──
    def forward(self, *, query_states=None, vl_embs_list=None, target_frames, encoder_attention_mask=None):
        # gen_loss = sum(w_i * loss_i) / sum(w_i) — a weighted MEAN, not a plain sum.
        # Normalizing by the total weight makes the aggregate gen_loss magnitude
        # invariant to how many targets there are, so the OUTER trainer.loss_scale
        # (gen_loss vs action_loss) alone sets the gen:action ratio, and the per-target
        # weights only split the gen budget INTERNALLY. E.g. with loss_scale gen:action
        # = 1:1 and target weights auto:siglip = 1:1, the total loss decomposes as
        # 1/2 action + 1/4 auto + 1/4 siglip.
        total = None
        for i, h in enumerate(self.heads):
            qs = query_states
            if h.requires_vlm_queries and query_states is not None and self._vlm_slices is not None:
                off, n = self._vlm_slices[i]
                qs = query_states[:, off : off + n, :]
            loss_i = h(
                query_states=qs,
                vl_embs_list=vl_embs_list,
                target_frames=target_frames,
                encoder_attention_mask=encoder_attention_mask,
            )
            wi = self._weights[i]
            total = wi * loss_i if total is None else total + wi * loss_i
        return total / self._weight_sum

    # ── isolation=off (coupled) support: concat every child's per-layer states ──
    def isolation_context(
        self,
        *,
        query_states=None,
        vl_embs_list=None,
        encoder_attention_mask=None,
        t_override=None,
        target_frames=None,
    ):
        # Each child stashes its own condition/loss here; the paired gen-loss
        # `forward` below drains every child's cache exactly once (QwenPI always
        # runs the gen-loss forward after isolation_context), so no cross-step reuse.
        if self.backbone != "expert":
            return None
        per_child = []
        for h in self.heads:
            g = h.isolation_context(
                query_states=query_states,
                vl_embs_list=vl_embs_list,
                encoder_attention_mask=encoder_attention_mask,
                t_override=t_override,
                target_frames=target_frames,
            )
            if g is None:
                return None  # a child with no separate gen stream -> coupling unsupported
            per_child.append(g)  # each: list of L tensors (B, n_i, H)
        n_layers = len(vl_embs_list)
        return [torch.cat([g[layer] for g in per_child], dim=1) for layer in range(n_layers)]

    def _consume_cached_condition(self):
        """Drop every child's cached expert output (inference coupling path)."""
        for h in self.heads:
            h._consume_cached_condition()
        return None

    # ── diffusion isolation=off joint inference: per-step, per-layer concat ──
    def denoise_gen_trajectory(self, vl_embs_list, encoder_attention_mask, num_steps, device):
        child_trajs = [
            h.denoise_gen_trajectory(vl_embs_list, encoder_attention_mask, num_steps, device) for h in self.heads
        ]
        n_layers = len(vl_embs_list)
        traj = []
        for step in range(num_steps):
            traj.append([torch.cat([ct[step][layer] for ct in child_trajs], dim=1) for layer in range(n_layers)])
        return traj

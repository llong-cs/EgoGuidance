# Configurations

Every file in this directory is a complete training configuration. Files may
inherit from others with `_base_: [...]` (paths are relative to the file; bases
are merged in order, the file's own keys win and lists are replaced). Any key can
be overridden on the command line with `--dotted.key value`:

```bash
bash scripts/train.sh configs/pretrain/cotrain_openaoe.yaml --seed 1
```

`train_entry` selects the training script: `train_starvla_gen_cotrain.py` for
pre-training (one or more streams), `train_starvla.py` for LIBERO fine-tuning.
The hyper-parameters follow the report (App. A); every key is documented in
[`docs/configuration.md`](../docs/configuration.md) and the expected data layout
in [`docs/data.md`](../docs/data.md).

## Layout

| Path | Report | Description |
|---|---|---|
| `pretrain/cotrain_openaoe.yaml` | SPAR (default) | DROID + Open-AoE, 1:1 per step; one SigLIP2 frame at the end of the action chunk, flow matching. |
| `pretrain/robot_only.yaml` | Robot-only baseline | DROID only (action + generation losses). Holds the full recipe that every other pre-training config inherits. |
| `pretrain/visual_target/` | Representational target / regression supervision | Six targets (Qwen-ViT, SigLIP2, DINOv2, LingBot, V-JEPA2, Wan-VAE) × {`diffusion`, `regression`}. |
| `pretrain/future_frames/` | State prediction can be single framed | SigLIP2 + flow matching: one vs. four future frames up to the end of the action chunk. |
| `pretrain/ego4d_curation/` | Data processing (Ego4D) | DROID + Ego4D clips: raw, Gate 1, Gate 1 + Gate 2 (baseline: `robot_only.yaml`). |
| `pretrain/data_scaling/` | Data scaling | DROID + 150 / 450 / 1,500 h of Open-AoE (0 h: `robot_only.yaml`). |
| `pretrain/data_diversity/` | Data diversity | 200K Open-AoE clip–description pairs balanced across scene / object / interaction clusters. |
| `finetune/libero.yaml` | Downstream fine-tuning | LIBERO with L_robot; VLM and generation expert transferred, action expert re-initialized. Use it for every pre-training checkpoint with the default target. |
| `finetune/visual_target/`, `finetune/future_frames/` | Downstream fine-tuning | `libero.yaml` with the generation settings of the matching pre-training config. |

Files whose name starts with `_` are shared bases and are not launched directly.
The report averages every number over three runs with different seeds
(`--seed 0`, `--seed 1`, `--seed 2` with distinct `--run_id`s).

## Batch size and hardware

Pre-training uses a global batch of 1,024 (512 robot + 512 egocentric samples)
and fine-tuning 2,048; the recipes were run on 32 GPUs with 80 GB or more.
`datasets.total_per_device_batch_size` is the per-GPU micro-batch summed over all
streams (split by `sampling_ratio`), and `trainer.global_batch_size` is kept
exactly: with fewer GPUs the trainer derives the gradient-accumulation steps,
which requires `global_batch_size` to be divisible by `micro_batch × num_gpus`.
Lower `total_per_device_batch_size` if you run out of memory; the global batch is
unaffected.

## Timing of the generation target

The future frame o<sub>t+Δ</sub> is aligned with the end of the action chunk,
a<sub>t+H</sub>: the horizon is τ = H / f of the robot stream (`datasets.vla_data`),
i.e. 8 / 15 Hz = 0.533 s when pre-training on DROID and 8 / 20 Hz = 0.4 s when
fine-tuning on LIBERO. Every other stream converts τ with its own `dataset_fps` to
⌊τ·f + ½⌋ frames (16 frames of a 30 fps egocentric video), so all streams predict the
same instant. `framework.gen_model.gen_horizon_seconds` sets τ explicitly instead,
and `framework.gen_model.num_future_frames` predicts T frames evenly spaced within τ.

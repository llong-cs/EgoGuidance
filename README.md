<div align="center">

# Ego Guidance: Enhancing Robot Policy Learning with Action-Free Human Egocentric Videos

[**Project Page**](https://llong-cs.github.io/EgoGuidance/) &nbsp;|&nbsp; [**Paper**](#citation) &nbsp;|&nbsp; [**Code**](https://github.com/llong-cs/EgoGuidance)

</div>

Human egocentric videos are abundant, but almost none of them come with action
labels. This repository studies how such *action-free* videos can still improve
robot policy learning. Building on a pretrained VLM (Qwen3.5-2B), **SPAR**
(*Shared Prediction on Action and Representation*) is a Mixture-of-Transformers
with two experts that cross-attend to the LLM's layer-wise hidden states:

- an **action expert** that predicts action chunks with flow matching, and
- a **generation expert** that predicts the representation of a *future frame*
  (aligned with the end of the action chunk) extracted by a frozen visual encoder (SigLIP2 by default; also
  Qwen-ViT, DINOv2, LingBot-Vision, V-JEPA2 or the Wan2.2 VAE), trained with flow
  matching or regression.

Robot demonstrations supervise both experts; egocentric videos supervise only
the generation expert, so they need no action annotation and influence the
policy only through the shared VLM. The two experts never attend to each other,
and the generation expert is removed at inference, so the deployed policy has
the same architecture and cost as a standard VLA.

This repository contains the model and training code (built on
[StarVLA](https://github.com/starVLA/starVLA)), the configuration of every
experiment in the report, and the LIBERO / LIBERO-Plus evaluation code.

## Repository structure

```
├── configs/              # experiment configurations (see configs/README.md)
│   ├── pretrain/         #   robot (+ egocentric video) pre-training
│   └── finetune/         #   LIBERO fine-tuning
├── starVLA/              # library code (package name kept from StarVLA)
│   ├── model/            #   QwenPI framework, action expert, generation expert (gen_model/)
│   ├── dataloader/       #   LeRobot robot data, egocentric video data, temporal alignment
│   └── training/         #   pre-training (multi-stream) and fine-tuning entry points
├── scripts/train.sh      # training launcher (single node or multi-node)
├── deployment/           # WebSocket policy server used for evaluation
├── examples/
│   ├── LIBERO/           #   LIBERO data registry and evaluation client
│   ├── LIBERO_Plus/      #   LIBERO-Plus robustness evaluation
│   └── DROID/            #   DROID data registry
├── tools/analysis/       # training-log comparison
├── tests/
└── docs/                 # configuration reference, data layout
```

## Installation

```bash
git clone https://github.com/llong-cs/EgoGuidance.git
cd EgoGuidance

conda create -n egoguidance python=3.10 -y
conda activate egoguidance

# PyTorch with CUDA (tested with torch 2.6 / CUDA 12.4); pick the wheel for your system
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# Qwen3.5 uses gated-delta linear attention; the configs require its fast kernels:
pip install flash-linear-attention==0.3.2 causal-conv1d==1.5.0.post8
pip install -e .
```

The LingBot-Vision target additionally needs the
[`lingbot-vision`](https://github.com/robbyant/lingbot-vision) package
(`pip install --no-deps git+https://github.com/robbyant/lingbot-vision.git`).

## Data and models

All configs reference weights and data relative to `playground/` (git-ignored);
download or symlink them there:

| What | Source | Local path | Used by |
|---|---|---|---|
| Qwen3.5-2B | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | `playground/Pretrained_models/Qwen3.5-2B` | all |
| SigLIP2 | [google/siglip2-so400m-patch16-256](https://huggingface.co/google/siglip2-so400m-patch16-256) | `playground/Pretrained_models/siglip2-so400m-patch16-256` | all (default target) |
| DINOv2 | [facebook/dinov2-giant](https://huggingface.co/facebook/dinov2-giant) | `playground/Pretrained_models/dinov2-giant` | visual-target study |
| V-JEPA2 | [facebook/vjepa2-vitg-fpc64-256](https://huggingface.co/facebook/vjepa2-vitg-fpc64-256) | `playground/Pretrained_models/vjepa2-vitg-fpc64-256` | visual-target study |
| LingBot-Vision | [robbyant/lingbot-vision-vit-giant](https://huggingface.co/robbyant/lingbot-vision-vit-giant) | `playground/Pretrained_models/lingbot-vision-vit-giant` | visual-target study |
| Wan2.2 VAE | [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) | `playground/Pretrained_models/Wan2.2-TI2V-5B-Diffusers` | visual-target study |
| DROID (LeRobot v3) | [nvidia/Cosmos3-DROID](https://huggingface.co/datasets/nvidia/Cosmos3-DROID) | `playground/Datasets/DROID` (the folder containing `success/`) | pre-training |
| Open-AoE | released clips and step descriptions | `playground/Datasets/Open-AoE` | pre-training |
| Ego4D | [ego4d-data.org](https://ego4d-data.org/) (license required), 5 s clips + captions | `playground/Datasets/Ego4D` | Ego4D curation study |
| LIBERO (LeRobot) | see [`examples/LIBERO/README.md`](examples/LIBERO/README.md) | `playground/Datasets/LEROBOT_LIBERO_DATA` | fine-tuning |

DROID episodes are curated when the dataset is indexed (successful episodes with
a valid instruction, valid actions and motion). Egocentric clips are read from an
`index.jsonl` file per stream. [`docs/data.md`](docs/data.md) describes the
expected layout, the `index.jsonl` format and the Ego4D curation levels.

## Training

Training has two stages. `scripts/train.sh` launches any config on all visible
GPUs of a node (multi-node: set `NNODES`, `NODE_RANK`, `MASTER_ADDR`,
`MASTER_PORT`); any config key can be overridden with `--dotted.key value`.

```bash
# Pre-training: co-train on DROID + Open-AoE (robot data: action + generation; video: generation only)
bash scripts/train.sh configs/pretrain/cotrain_openaoe.yaml

# Fine-tuning on LIBERO from the pre-training checkpoint
bash scripts/train.sh configs/finetune/libero.yaml \
    --trainer.pretrained_checkpoint playground/Checkpoints/pretrain_cotrain_openaoe/checkpoints/steps_20000_pytorch_model.pt
```

Outputs go to `playground/Checkpoints/<run_id>/` (`--run_root_dir` and
`--run_id` change this): `checkpoints/steps_<N>_pytorch_model.pt`,
`config.yaml`, `dataset_statistics.json` and `logs/train_node<rank>_<time>.log`.
Metrics are logged to Weights & Biases when it is installed and logged in (or
`WANDB_MODE=offline`); otherwise they are written to `metrics.jsonl` in the run
directory (`--trainer.disable_tracking true` turns W&B off). All configuration
keys are described in [`docs/configuration.md`](docs/configuration.md).

Pre-training uses a global batch of 1,024 (512 robot + 512 egocentric samples;
32 GPUs × 32) and fine-tuning 2,048; on fewer GPUs the gradient-accumulation
steps are derived automatically. To try the pipeline on a single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh configs/pretrain/cotrain_openaoe.yaml \
    --run_id smoke --trainer.max_train_steps 2 --trainer.global_batch_size 4 \
    --datasets.total_per_device_batch_size 4 --datasets.vla_data.episode_fraction 0.002 \
    --trainer.save_interval 100 --trainer.eval_interval 100 --trainer.logging_frequency 1
```

## Evaluation

Evaluation runs a policy server (loads the checkpoint, returns actions) and a
simulator client in a separate LIBERO environment that connects to it over WebSocket:

```bash
CKPT=playground/Checkpoints/finetune_libero/checkpoints/steps_20000_pytorch_model.pt

# LIBERO: terminal 1 (this environment) serves the policy, terminal 2 (LIBERO environment) rolls out
CKPT=$CKPT bash examples/LIBERO/eval_files/run_policy_server.sh
CKPT=$CKPT bash examples/LIBERO/eval_files/eval_libero.sh

# LIBERO-Plus on several GPUs: one command from this environment
LIBERO_PYTHON=$(conda run -n libero_plus which python) GPUS="0 1 2 3 4 5 6 7" CKPT=$CKPT \
    bash examples/LIBERO_Plus/eval_files/run_eval.sh
```

Each run writes per-episode results, videos and a `summary.md` (success rate
per suite, and per perturbation dimension for LIBERO-Plus) under
`<run_dir>/eval/`; re-running the same command resumes. Installation of the
simulators and all options:

- **LIBERO** (Spatial / Object / Goal / Long): [`examples/LIBERO/README.md`](examples/LIBERO/README.md).
- **LIBERO-Plus** (camera, robot initial state, language, lighting, background,
  sensor noise, object layout): [`examples/LIBERO_Plus/README.md`](examples/LIBERO_Plus/README.md).

## Reproducing the experiments

Every pre-training config is followed by LIBERO fine-tuning and evaluation on
LIBERO / LIBERO-Plus; the report averages each number over three seeds
(`--seed`, with a distinct `--run_id` per run).

| Study | Pre-training | Fine-tuning |
|---|---|---|
| Visual target × flow matching / regression | `configs/pretrain/visual_target/` | `configs/finetune/visual_target/` (same name) |
| One vs. four future frames | `configs/pretrain/future_frames/` | `configs/finetune/future_frames/` (same name) |
| Ego4D curation (raw, Gate 1, Gate 1 + 2 vs. robot-only) | `configs/pretrain/ego4d_curation/`, `configs/pretrain/robot_only.yaml` | `configs/finetune/libero.yaml` (16k and 20k checkpoints) |
| Egocentric data scaling (0 / 150 / 450 / 1,500 h) | `configs/pretrain/robot_only.yaml`, `configs/pretrain/data_scaling/` | `configs/finetune/libero.yaml` |
| Egocentric data diversity (200K pairs) | `configs/pretrain/data_diversity/` | `configs/finetune/libero.yaml` |
| Re-initializing vs. continuing the action expert | `configs/pretrain/cotrain_openaoe.yaml` | `configs/finetune/libero.yaml` vs. `--trainer.transfer_policy full --trainer.transfer_retain_gen_model false` |

## Acknowledgements

This code base is built on [StarVLA](https://github.com/starVLA/starVLA), which
in turn builds on [LeRobot](https://github.com/huggingface/lerobot),
[GR00T](https://github.com/NVIDIA/Isaac-GR00T), [Qwen](https://github.com/QwenLM)
and [DeepSpeed](https://github.com/deepspeedai/DeepSpeed). We thank the authors
of [DROID](https://droid-dataset.github.io/), [Ego4D](https://ego4d-data.org/),
Open-AoE, EgoSmith, [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
and LIBERO-Plus for releasing their data and tools.

## Citation

```bibtex
@article{long2026egoguidance,
  title   = {Ego Guidance: Enhancing Robot Policy Learning with Action-Free Human Egocentric Videos},
  author  = {Long, Lin and Ma, Tianren and Chu, Zhong and Wu, Di and Cao, Maosong and Ye, Qixiang and Zhao, Junbo and Zhang, Shanghang},
  journal = {arXiv preprint},
  year    = {2026}
}

@article{community2026starvla,
  title   = {StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing},
  author  = {Community, StarVLA},
  journal = {arXiv preprint arXiv:2604.05014},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE). See also [LEGAL.md](LEGAL.md).

# LIBERO

Fine-tuning data and evaluation for the four LIBERO suites: Spatial, Object, Goal and Long (`libero_10`).
One policy is trained on all four suites and evaluated on each of them.

## Fine-tuning data

```bash
bash examples/LIBERO/data_preparation.sh            # -> playground/Datasets/LEROBOT_LIBERO_DATA
# or download elsewhere and symlink it there:
bash examples/LIBERO/data_preparation.sh /data/LEROBOT_LIBERO_DATA
```

The script downloads the four `IPEC-COMMUNITY/libero_*_no_noops_1.0.0_lerobot` datasets (LeRobot v2.0) and
copies [`train_files/modality.json`](train_files/modality.json) into each `meta/` directory. The data
registry (`libero_all` mixture, front + wrist cameras, 7-D delta end-effector actions) is in
[`train_files/data_registry/data_config.py`](train_files/data_registry/data_config.py). Fine-tune with:

```bash
bash scripts/train.sh configs/finetune/libero.yaml \
    --trainer.pretrained_checkpoint playground/Checkpoints/<pretrain_run_id>/checkpoints/steps_20000_pytorch_model.pt
```

## Evaluation

Evaluation uses two processes that talk over WebSocket:

- the **policy server** (`deployment/model_server/server_policy.py`, starVLA environment) loads the
  checkpoint together with the `config.yaml` and `dataset_statistics.json` of its run directory and returns
  **un-normalized** action chunks (`action_horizon` actions, 8 in the released recipes);
- the **simulator client** ([`eval_files/eval_libero.py`](eval_files/eval_libero.py), LIBERO environment)
  renders the front and wrist cameras at 256x256, rotates both by 180 degrees (as in the training data),
  sends them with the instruction and the checkpoint's per-view role prompts, and executes each chunk
  open-loop (a new chunk every `action_horizon` steps, no ensembling). The gripper output (1 = open) is binarized at 0.5
  and mapped to LIBERO's convention (-1 = open, +1 = close). Proprioceptive state is not sent (the
  checkpoints are trained with `include_state: false`).

Protocol: 50 episodes per task with LIBERO's fixed initial states, seed 7, 10 no-op steps before the
policy starts, and a step limit of 220 / 280 / 300 / 520 policy steps for Spatial / Object / Goal / Long.

### 1. Install the LIBERO environment

Use a separate environment from starVLA:

```bash
conda create -n libero python=3.10 -y
conda activate libero
bash examples/LIBERO/eval_files/install_libero.sh
```

This clones LIBERO into `playground/LIBERO`, installs it with its requirements plus the client
dependencies (`websockets`, `msgpack`, `imageio`, CPU-only `torch` for loading the initial states), and
writes LIBERO's `config.yaml` to `playground/libero_config/libero/` (the evaluation scripts point
`LIBERO_CONFIG_PATH` there).

### 2. Run (two terminals, from the repository root)

```bash
# Terminal 1, starVLA environment: start the policy server
CKPT=playground/Checkpoints/<run_id>/checkpoints/steps_20000_pytorch_model.pt \
    bash examples/LIBERO/eval_files/run_policy_server.sh

# Terminal 2, LIBERO environment: roll out all four suites
CKPT=playground/Checkpoints/<run_id>/checkpoints/steps_20000_pytorch_model.pt \
    bash examples/LIBERO/eval_files/eval_libero.sh
```

Both scripts take their settings from environment variables documented at the top of each file:
`GPU_ID`, `PORT` (10093), `USE_BF16` (1) and `UNNORM_KEY` for the server; `PORT`, `SUITES`, `NUM_TRIALS`,
`VIDEOS_PER_TASK` and `OUT_DIR` for the client. For a quick smoke test, use
`NUM_TRIALS=1 EVAL_ARGS="--max-tasks-per-suite 1" OUT_DIR=playground/eval_smoke` (a results directory only
accepts one protocol, so keep smoke tests separate).

For several GPUs, `examples/LIBERO_Plus/eval_files/run_eval.sh` starts the servers and parallel simulator
workers itself and also runs standard LIBERO (`BENCHMARK=libero`); see the
[LIBERO-Plus README](../LIBERO_Plus/README.md).

### 3. Results

Results are written to `OUT_DIR`, by default `<run_dir>/eval/libero/<checkpoint name>/`:

| Path | Content |
|---|---|
| `episodes/*.json` | outcome of every episode, written as soon as it finishes |
| `eval_config.json` | protocol and checkpoint of this directory (a mismatching re-run is refused) |
| `videos/<suite>/task<i>_ep<j>_<success\|failure>.mp4` | front-camera videos of the first `VIDEOS_PER_TASK` episodes per task |
| `summary.md`, `summary.json` | success rate per suite and the average over suites |
| `eval.log` | client log |

Re-running the same command resumes an interrupted evaluation (finished episodes are kept). The summary is
produced at the end of `eval_libero.sh`; to recompute it at any time:

```bash
python examples/LIBERO/eval_files/summarize_results.py <OUT_DIR>
```

Only tasks with all their episodes recorded are counted; incomplete tasks are listed.

### Notes

- **Rendering.** The client renders headless with EGL (`MUJOCO_GL=egl`). Select the GPU with
  `CUDA_VISIBLE_DEVICES=<i> MUJOCO_EGL_DEVICE_ID=<i>`. Without EGL, `MUJOCO_GL=osmesa` renders on the CPU
  (needs `libosmesa6-dev`); it is slower and its images are not bit-identical to EGL, so success rates can
  differ slightly. The reported numbers use EGL.
- **PyTorch >= 2.6** in the LIBERO environment refuses to load LIBERO's initial-state files
  (`torch.load` defaults to `weights_only=True`); use the pinned 2.4.1 or set
  `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.
- **Proxies.** The WebSocket client removes `HTTP(S)_PROXY` / `ALL_PROXY` from its environment before
  connecting, so the connection to the policy server is never routed through an HTTP proxy.
- Flow-matching sampling on the server is not seeded, so repeated evaluations of one checkpoint vary
  slightly.

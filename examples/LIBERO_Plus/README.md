# LIBERO-Plus

[LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) is a robustness benchmark built on the four LIBERO
suites. It perturbs every task along seven dimensions (camera viewpoint, robot initial state, language
instruction, lighting, background texture, sensor noise, object layout), giving 10,030 tasks
(Spatial 2,402, Object 2,518, Goal 2,591, Long 2,519) that are each evaluated with **one** episode. We
report the success rate per perturbation dimension and overall.

The policy server and the simulator client are the same as for [LIBERO](../LIBERO/README.md)
(`deployment/model_server/server_policy.py` and `examples/LIBERO/eval_files/eval_libero.py`, with the same
observation format and action post-processing). LIBERO-Plus is a drop-in fork of the `libero` package, so
it needs its own environment.

## 1. Install the LIBERO-Plus environment

```bash
# ImageMagick's MagickWand library, used by LIBERO-Plus's sensor-noise perturbations through `wand`
sudo apt-get install -y libmagickwand-dev libexpat1 libfontconfig1-dev

conda create -n libero_plus python=3.10 -y
conda activate libero_plus
bash examples/LIBERO_Plus/eval_files/install_libero_plus.sh
```

The script

1. clones LIBERO-Plus into `playground/LIBERO-plus` and installs it with its requirements, `wand`,
   `scikit-image` and the client dependencies (numpy kept at LIBERO's 1.22.4);
2. assembles `libero/libero/assets/`: the LIBERO-Plus `assets.zip`
   ([Hugging Face](https://huggingface.co/datasets/Sylvest/LIBERO-plus), 6.4 GB, saved to
   `playground/Datasets/LIBERO-plus/`) does not contain all standard LIBERO assets
   (e.g. `turbosquid_objects/`, `stable_scanned_objects/`, `textures/`, `wall.xml`), so the standard LIBERO
   assets are copied first (from `playground/LIBERO`, cloned if missing) and the archive is extracted over
   them;
3. writes LIBERO-Plus's `config.yaml` to `playground/libero_config/libero_plus/` (separate from standard
   LIBERO's, since both packages are called `libero`);
4. checks that `wand` finds MagickWand and that the task counts are those of LIBERO-Plus.

Without root access, ImageMagick can also come from conda (`conda install -c conda-forge imagemagick`,
then `export MAGICK_HOME=$CONDA_PREFIX`).

## 2. Run

`run_eval.sh` runs everything on one machine: it starts `SERVERS_PER_GPU` policy servers on each GPU of
`GPUS`, starts `WORKERS_PER_SERVER` simulator workers per server (each worker renders with EGL on its
server's GPU and processes every M-th task), waits for them, stops the servers and writes the summary.
Launch it from the **starVLA** environment and point `LIBERO_PYTHON` at the LIBERO-Plus environment:

```bash
conda activate egoguidance   # the training environment (see the top-level README)
LIBERO_PYTHON=$(conda run -n libero_plus which python) \
GPUS="0 1 2 3 4 5 6 7" WORKERS_PER_SERVER=4 \
CKPT=playground/Checkpoints/<run_id>/checkpoints/steps_20000_pytorch_model.pt \
    bash examples/LIBERO_Plus/eval_files/run_eval.sh
```

Smoke test (two tasks per suite): add `EVAL_ARGS="--max-tasks-per-suite 2" OUT_DIR=playground/eval_smoke`
(use a separate `OUT_DIR`, since a results directory only accepts one protocol). The same launcher evaluates
standard LIBERO in parallel with `BENCHMARK=libero` and `LIBERO_PYTHON` from the LIBERO environment
(50 episodes per task, split into units of `EPISODE_CHUNK=10` episodes).

Other variables (documented at the top of the script): `BASE_PORT` (10093; server `i` listens on
`BASE_PORT + i`), `SUITES`, `VIDEOS_PER_TASK` (0), `UNITS_PER_PROCESS` (8), `MAX_RESTARTS` (20),
`MUJOCO_GL` (`egl` or `osmesa`), `USE_BF16` (1), `UNNORM_KEY`, `OUT_DIR`. Each simulator worker needs
about one CPU core; the BLAS/OpenMP thread count of every process is set to `nproc / (servers + workers)`.

## 3. Results

Results are written to `OUT_DIR`, by default `<run_dir>/eval/libero_plus/<checkpoint name>/`:

| Path | Content |
|---|---|
| `episodes/*.json` | outcome of every task, written as soon as it finishes |
| `eval_config.json` | protocol, checkpoint and `task_classification.json` path (a mismatching re-run is refused) |
| `summary.md`, `summary.json` | success rate per dimension, per suite and overall |
| `logs/server_<i>.log`, `logs/worker_<j>.log` | process logs |

Interrupted or partially failed runs are resumed by running the same command again. To see the progress or
recompute the summary at any time:

```bash
python examples/LIBERO/eval_files/summarize_results.py <OUT_DIR>
```

The dimension of each task comes from LIBERO-Plus's `libero/libero/benchmark/task_classification.json`
(1-based ids; the path is recorded in `eval_config.json`, or pass `--task-classification`). The "Total"
column is the success rate over all completed tasks.

## Notes

- **Worker restarts.** Every LIBERO-Plus task builds a new simulator, and renderer memory is not fully
  released when an environment is closed. Workers therefore exit after `UNITS_PER_PROCESS` tasks and are
  restarted by the launcher; crashed workers are restarted as well (finished tasks are kept).
- **Slow sensor-noise tasks.** The glass-blur perturbation of LIBERO-Plus moves pixels in a Python loop and
  takes about a second per frame, so these tasks run much slower than the others. We evaluate with the
  upstream implementation unchanged. (The evaluations in the report used a faster re-implementation that
  was verified to be pixel- and random-state-identical under NumPy 1.22; it only saves time and is not
  included here.)
- **Rendering.** EGL device numbering can differ from CUDA numbering; this only changes which GPU renders.
  `MUJOCO_GL=osmesa` renders on the CPU (needs `libosmesa6-dev`); its images are not bit-identical to EGL,
  so success rates can differ slightly. The reported numbers use EGL.
- **Randomness.** NumPy is seeded once per worker process (seed 7), and several perturbations draw from it
  (e.g. motion-blur angle, fog), so the exact perturbation noise depends on how tasks are split among
  workers; flow-matching sampling on the server is not seeded either. Expect small run-to-run variation.
- See the [LIBERO README](../LIBERO/README.md#notes) for PyTorch >= 2.6 and proxy notes.

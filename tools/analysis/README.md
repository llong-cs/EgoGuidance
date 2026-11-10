# Training-log analysis

Every run writes its outputs to `<run_root_dir>/<run_id>/` (default `run_root_dir`:
`playground/Checkpoints`). Per-step metrics can be read from two places there.

## `metrics.jsonl` (simplest)

If W&B is not used (not installed, no API key, `WANDB_MODE=disabled` or
`trainer.disable_tracking: true`), the trainers append every logged step to
`<run_dir>/metrics.jsonl` (`starVLA/training/trainer_utils/tracking.py`). Each line is one
JSON object, `{"step": n, ...}`, holding the same metrics as the `Step <n>, Loss: {...}` log
line. It is written every `trainer.logging_frequency` steps and includes `val/action_mse` on
evaluation steps that are also logging steps (every evaluation step with the released configs).

```python
import json

rows = [json.loads(line) for line in open("playground/Checkpoints/<run_id>/metrics.jsonl")]
val = [(r["step"], r["val/action_mse"]) for r in rows if "val/action_mse" in r]
```

## `extract_metrics.py` (from the training logs)

`extract_metrics.py` works whichever tracker was used, and also recovers the wall time of
every step. It needs only the Python standard library.

```bash
python tools/analysis/extract_metrics.py playground/Checkpoints out/metrics
```

The first argument is searched recursively for the node-0 logs that `scripts/train.sh`
writes, `<run_dir>/logs/train_node0_<timestamp>.log`. The run id is the name of the run
directory. All node-0 logs of a run (one per launch, e.g. after a resume) are read in launch
order, and a later launch overrides an earlier one for the same step. For every run:

| File | Content |
|---|---|
| `<name>.csv` | One row per logged step: every numeric metric of the `Step <n>, Loss: {...}` log line (`action_loss`, `gen_loss`, `video_gen_loss`, `total_loss`, learning rates, and `val/action_mse` as in `metrics.jsonl`), plus `step_time_s`. |
| `<name>__steptime.csv` | Wall time of every step, from the tqdm progress bar. Use this for throughput and stall statistics, because periodic stalls rarely fall on logged steps. |
| `summary.csv` | One row per run: last logged losses, `val/action_mse`, progress, and median / mean step time (first 10 % of steps excluded). |

`<name>` is the run directory's path below the search root, joined with `__`. When the
search root is the `run_root_dir`, this is just the run id.

To plot a metric:

```python
import csv
import matplotlib.pyplot as plt

rows = [r for r in csv.DictReader(open("out/metrics/<run_id>.csv")) if r["val/action_mse"]]
plt.plot([int(r["step"]) for r in rows], [float(r["val/action_mse"]) for r in rows])
plt.xlabel("step"); plt.ylabel("val/action_mse"); plt.show()
```

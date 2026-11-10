#!/usr/bin/env python3
"""Extract per-step training metrics from StarVLA training logs into CSV files.

    python tools/analysis/extract_metrics.py <run_root> <out_dir>

``scripts/train.sh`` writes the log of node 0 to
``<run_root_dir>/<run_id>/logs/train_node0_<YYYYmmdd_HHMMSS>.log`` (one file per launch, so
a resumed run has several). This tool searches ``<run_root>`` recursively for
``logs/train_node0_*.log``, treats the directory holding ``logs/`` as the run directory
(its name is the run id), reads all of that run's node-0 logs in launch order (a later
launch overrides an earlier one for the same step), and parses two sources:

* the periodic ``Step <n>, Loss: {...}`` records (every ``trainer.logging_frequency``
  steps): every numeric metric of the logged dict (losses, learning rates, timing, and
  ``val/*`` on evaluation steps);
* the tqdm progress snapshots, from which the wall time of every step is derived. A step's
  time is taken from its FIRST appearance: tqdm refreshes on a timer, so one step can emit
  several snapshots and consecutive snapshots do not delimit whole steps.

Outputs, per run (``<name>`` is the run directory's path below ``<run_root>`` joined with
``__``; for the standard layout ``<run_root> = <run_root_dir>`` this is just the run id):

* ``<out_dir>/<name>.csv``: one row per logged step, plus ``step_time_s`` where known;
* ``<out_dir>/<name>__steptime.csv``: the complete per-step wall-time series. Periodic
  stalls (e.g. dataloader worker recycling) rarely coincide with logged steps, so stall
  statistics must come from this file rather than from the logged-step CSV;
* ``<out_dir>/summary.csv``: last logged values and step-time statistics of every run.

When W&B is not used, the trainers also append the same per-step metrics to
``<run_dir>/metrics.jsonl`` (``starVLA/training/trainer_utils/tracking.py``); that file
needs no parsing but has no per-step wall time.

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import statistics
import sys
from pathlib import Path

STEP_MARKER_RE = re.compile(r"Step (\d+), Loss:")
TQDM_RE = re.compile(r"(\d+)/(\d+) \[(\d+:\d+(?::\d+)?)<")
# Rich's right-aligned source location (e.g. "train_starvla.py:1963") at the end of a log line.
SOURCE_LOCATION_RE = re.compile(r"\s*[A-Za-z_][A-Za-z0-9_]*\.py:\d+\s*$")
METRIC_RE = re.compile(r"['\"]([A-Za-z0-9_/]+)['\"]\s*:\s*([-+0-9.eE]+|nan|inf)")
SUMMARY_KEYS = ("action_loss", "gen_loss", "video_gen_loss", "total_loss", "val/action_mse")
MAX_RECORD_LINES = 60


def to_sec(t: str) -> int:
    """tqdm elapsed time ``[H:]MM:SS`` -> seconds."""
    p = [int(x) for x in t.split(":")]
    return p[0] * 3600 + p[1] * 60 + p[2] if len(p) == 3 else p[0] * 60 + p[1]


def parse_metrics_blob(blob: str) -> dict[str, float]:
    """Pull ``'key': number`` pairs out of a logged dict repr.

    The rich handler hard-wraps the dict across lines at whitespace (sometimes between a
    value and the following comma), so the caller joins continuation lines with a single
    space; joining with a space rather than concatenating never merges two tokens.
    """
    flat = re.sub(r"\s+", " ", blob)
    out = {}
    for k, v in METRIC_RE.findall(flat):
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


def parse_step_rows(txt: str):
    """Yield ``(step, metrics)`` for every ``Step <n>, Loss: {...}`` record.

    The dict may start on the marker line or on the following line, and may continue over
    several wrapped lines until its closing brace.
    """
    lines = txt.split("\n")
    for i, line in enumerate(lines):
        m = STEP_MARKER_RE.search(line)
        if not m:
            continue
        buf, depth, started = [], 0, False
        for s in [line[m.end() :], *lines[i + 1 : i + MAX_RECORD_LINES]]:
            s = SOURCE_LOCATION_RE.sub("", s.strip())
            if not s:
                continue
            buf.append(s)
            depth += s.count("{") - s.count("}")
            if "{" in s:
                started = True
            if started and depth <= 0:
                break
        if buf:
            d = parse_metrics_blob(" ".join(buf))
            if d:
                yield int(m.group(1)), d


def total_steps(txt: str) -> int:
    """The run's ``max_train_steps``, read off the largest tqdm denominator.

    A run can show more than one progress bar (evaluation loops have their own); the
    training bar is always the longest.
    """
    tots = [int(t) for _, t, _ in TQDM_RE.findall(txt)]
    return max(tots) if tots else 0


def steptimes(txt: str, total: int) -> dict[int, int]:
    """Per-step wall time in seconds, keyed by step.

    Only snapshots of the training bar (denominator == ``total``) count, and each step is
    timed by its FIRST appearance. Only consecutive step numbers are differenced: a gap means
    snapshots were lost, and differencing across it would attribute several steps to one.
    """
    first: dict[int, int] = {}
    for a, tot, b in TQDM_RE.findall(txt):
        if tot != str(total):
            continue
        s, t = int(a), to_sec(b)
        if s not in first or t < first[s]:
            first[s] = t
    ks = sorted(first)
    return {k1: first[k1] - first[k0] for k0, k1 in itertools.pairwise(ks) if k1 - k0 == 1}


def discover_runs(root: Path) -> list[tuple[str, str, list[Path]]]:
    """``(name, run_id, node-0 logs in launch order)`` for every run directory below ``root``."""
    by_run: dict[Path, list[Path]] = {}
    for log in root.rglob("logs/train_node0_*.log"):
        by_run.setdefault(log.parent.parent, []).append(log)
    runs = []
    for run_dir in sorted(by_run):
        parts = run_dir.relative_to(root).parts or (run_dir.name,)
        # The file names end in a launch timestamp, so name order is launch order.
        runs.append(("__".join(parts), run_dir.name, sorted(by_run[run_dir], key=lambda p: p.name)))
    return runs


def extract_run(log_paths: list[Path], out_dir: Path, name: str, run_id: str | None = None) -> dict | None:
    """Write the two per-run CSV files; returns the run's summary row (None if nothing was logged).

    Each log is parsed on its own (a resumed launch restarts the tqdm clock), and a later log
    overrides an earlier one for the same step.
    """
    total = 0
    st: dict[int, int] = {}
    by_step: dict[int, dict] = {}
    for log_path in log_paths:
        txt = log_path.read_text(errors="ignore").replace("\r", "\n")
        log_total = total_steps(txt)
        total = max(total, log_total)
        st.update(steptimes(txt, log_total))
        for step, d in parse_step_rows(txt):
            by_step[step] = dict(d, step=step)
    if not by_step:
        return None
    rows = [by_step[k] for k in sorted(by_step)]
    for r in rows:
        if r["step"] in st:
            r["step_time_s"] = st[r["step"]]
    cols = ["step", "step_time_s", *sorted({k for r in rows for k in r} - {"step", "step_time_s"})]
    path = out_dir / f"{name}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    with (out_dir / f"{name}__steptime.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "step_time_s"])
        for k in sorted(st):
            w.writerow([k, st[k]])
    # Step-time statistics skip the first 10% of steps (start-up, compilation, warm-up).
    sts = [st[k] for k in sorted(st)]
    sts = sts[int(len(sts) * 0.1) :] or sts
    last = rows[-1]
    summary = {
        "run": name,
        "run_id": run_id or name,
        "logs": len(log_paths),
        "rows": len(rows),
        "last_step": last["step"],
        "total_steps": total,
        "pct": round(100 * last["step"] / total, 1) if total else "",
        "step_time_median": round(statistics.median(sts), 2) if sts else "",
        "step_time_mean": round(statistics.mean(sts), 2) if sts else "",
    }
    summary.update({k: last.get(k, "") for k in SUMMARY_KEYS})
    summary["csv"] = path.name
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Training logs -> per-step metric CSV files.")
    p.add_argument(
        "run_root",
        type=Path,
        help="Directory searched recursively for <run_dir>/logs/train_node0_*.log (e.g. playground/Checkpoints).",
    )
    p.add_argument("out_dir", type=Path, help="Output directory for the CSV files.")
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, run_id, log_paths in discover_runs(args.run_root):
        row = extract_run(log_paths, args.out_dir, name, run_id)
        if row is not None:
            summary.append(row)

    if not summary:
        print("no runs parsed")
        return 1
    spath = args.out_dir / "summary.csv"
    with spath.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"\n{len(summary)} runs -> {args.out_dir}/  (summary: {spath})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

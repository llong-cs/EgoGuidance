# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Merge per-episode results written by ``eval_libero.py`` and report success rates.

Reads ``<out_dir>/episodes/*.json`` (any number of workers / episode chunks; an episode recorded
twice is counted once) and ``<out_dir>/eval_config.json``. Prints and writes
``<out_dir>/summary.json`` and ``<out_dir>/summary.md`` with

  - the success rate per suite and the average over suites (LIBERO), and
  - for LIBERO-Plus, the success rate per perturbation dimension and overall, using the
    benchmark's ``task_classification.json`` (recorded in ``eval_config.json`` by the rollout
    client, or passed with ``--task-classification``).

Only tasks whose ``num_trials_per_task`` episodes are all recorded are counted; incomplete
tasks are reported so that the evaluation can be resumed. Standard library only.

Usage::

    python examples/LIBERO/eval_files/summarize_results.py <out_dir> [--task-classification FILE]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SUITE_ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]
SUITE_LABEL = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "Long",
    "libero_90": "LIBERO-90",
}
STANDARD_TASK_COUNT = {"libero_spatial": 10, "libero_object": 10, "libero_goal": 10, "libero_10": 10, "libero_90": 90}
# LIBERO-Plus perturbation dimensions: (category in task_classification.json, short label).
DIMENSIONS = [
    ("Camera Viewpoints", "Camera"),
    ("Robot Initial States", "Robot"),
    ("Language Instructions", "Language"),
    ("Light Conditions", "Light"),
    ("Background Textures", "Background"),
    ("Sensor Noise", "Noise"),
    ("Objects Layout", "Layout"),
]


def load_task_results(out_dir: Path, num_trials: int) -> tuple[dict, dict]:
    """Merge unit files into per-task results.

    Returns ``(complete, incomplete)``: ``complete[(suite, task_id)] = (successes, episodes)`` for tasks
    with all ``num_trials`` episodes recorded, and ``incomplete[(suite, task_id)] = episodes recorded``.
    """
    outcomes = defaultdict(dict)  # (suite, task_id) -> {episode: success}
    for path in sorted((out_dir / "episodes").glob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            print(f"warning: skipping unreadable {path}", file=sys.stderr)
            continue
        key = (rec["suite"], int(rec["task_id"]))
        for ep in rec.get("episodes", []):
            outcomes[key].setdefault(int(ep["episode"]), bool(ep["success"]))

    complete, incomplete = {}, {}
    for key, eps in outcomes.items():
        trials = {e: s for e, s in eps.items() if e < num_trials}
        if len(trials) == num_trials:
            complete[key] = (sum(trials.values()), num_trials)
        else:
            incomplete[key] = len(trials)
    return complete, incomplete


def _rate(successes: int, episodes: int) -> float:
    return 100.0 * successes / episodes if episodes else float("nan")


def summarize(out_dir: Path, task_classification: Path | None = None) -> dict:
    config = json.loads((out_dir / "eval_config.json").read_text())
    num_trials = int(config["num_trials_per_task"])
    complete, incomplete = load_task_results(out_dir, num_trials)
    evaluated_suites = {suite for suite, _ in complete} | {suite for suite, _ in incomplete}

    per_suite = defaultdict(lambda: [0, 0, 0])  # successes, episodes, tasks
    for (suite, _), (s, e) in complete.items():
        per_suite[suite][0] += s
        per_suite[suite][1] += e
        per_suite[suite][2] += 1
    suites = sorted(per_suite, key=lambda x: SUITE_ORDER.index(x) if x in SUITE_ORDER else len(SUITE_ORDER))
    total_s = sum(per_suite[x][0] for x in suites)
    total_e = sum(per_suite[x][1] for x in suites)
    summary = {
        "out_dir": str(out_dir),
        "benchmark": config.get("benchmark"),
        "checkpoint": config.get("checkpoint"),
        "num_trials_per_task": num_trials,
        "complete_tasks": len(complete),
        "incomplete_tasks": sorted(f"{s}:{t}" for s, t in incomplete),
        "per_suite": {
            x: {
                "successes": per_suite[x][0],
                "episodes": per_suite[x][1],
                "tasks": per_suite[x][2],
                "success_rate": _rate(*per_suite[x][:2]),
            }
            for x in suites
        },
        "suite_average": (sum(_rate(*per_suite[x][:2]) for x in suites) / len(suites)) if suites else float("nan"),
        "overall": {"successes": total_s, "episodes": total_e, "success_rate": _rate(total_s, total_e)},
    }

    if config.get("benchmark") == "libero_plus":
        path = task_classification or (
            Path(config["task_classification"]) if config.get("task_classification") else None
        )
        if path is None or not path.is_file():
            raise SystemExit("LIBERO-Plus results need task_classification.json; pass --task-classification.")
        # task_classification.json: {suite: [{"id": 1-based task id, "category": ..., ...}, ...]}
        category = {
            (suite, int(item["id"]) - 1): item["category"]
            for suite, items in json.loads(path.read_text()).items()
            for item in items
        }
        per_dim = defaultdict(lambda: [0, 0])
        unmapped = []
        for key, (s, e) in complete.items():
            if key not in category:
                unmapped.append(f"{key[0]}:{key[1]}")
                continue
            per_dim[category[key]][0] += s
            per_dim[category[key]][1] += e
        summary["task_classification"] = str(path)
        summary["per_dimension"] = {
            name: {"successes": per_dim[name][0], "episodes": per_dim[name][1], "success_rate": _rate(*per_dim[name])}
            for name, _ in DIMENSIONS
        }
        extra = sorted(set(per_dim) - {name for name, _ in DIMENSIONS})
        if extra:
            summary["other_categories"] = extra
        summary["unmapped_tasks"] = sorted(unmapped)
        summary["expected_tasks"] = sum(1 for suite, _ in category if suite in evaluated_suites)
    else:
        summary["expected_tasks"] = sum(STANDARD_TASK_COUNT.get(suite, 0) for suite in evaluated_suites)
    return summary


def to_markdown(summary: dict) -> str:
    lines = [f"# Results: {summary['out_dir']}", ""]
    lines.append(f"- checkpoint: `{summary['checkpoint']}`")
    lines.append(f"- trials per task: {summary['num_trials_per_task']}")
    n_inc = len(summary["incomplete_tasks"])
    lines.append(
        f"- complete tasks: {summary['complete_tasks']} of {summary['expected_tasks']} in the evaluated suites"
        + (f" ({n_inc} started but incomplete)" if n_inc else "")
    )
    lines.append("")
    if summary.get("per_dimension"):
        lines.append("| " + " | ".join(label for _, label in DIMENSIONS) + " | Total |")
        lines.append("|" + "---|" * (len(DIMENSIONS) + 1))
        cells = [f"{summary['per_dimension'][name]['success_rate']:.1f}" for name, _ in DIMENSIONS]
        lines.append("| " + " | ".join(cells) + f" | {summary['overall']['success_rate']:.1f} |")
        lines.append("")
    suites = list(summary["per_suite"])
    lines.append("| " + " | ".join(SUITE_LABEL.get(x, x) for x in suites) + " | Avg |")
    lines.append("|" + "---|" * (len(suites) + 1))
    cells = [f"{summary['per_suite'][x]['success_rate']:.1f}" for x in suites]
    lines.append("| " + " | ".join(cells) + f" | {summary['suite_average']:.1f} |")
    lines.append("")
    lines.append("| suite | tasks | successes / episodes | SR (%) |")
    lines.append("|---|---|---|---|")
    for x in suites:
        r = summary["per_suite"][x]
        lines.append(f"| {x} | {r['tasks']} | {r['successes']} / {r['episodes']} | {r['success_rate']:.1f} |")
    if summary.get("per_dimension"):
        lines.append("")
        lines.append("| dimension | successes / episodes | SR (%) |")
        lines.append("|---|---|---|")
        for name, _ in DIMENSIONS:
            r = summary["per_dimension"][name]
            lines.append(f"| {name} | {r['successes']} / {r['episodes']} | {r['success_rate']:.1f} |")
        if summary.get("unmapped_tasks"):
            lines.append(f"\n{len(summary['unmapped_tasks'])} completed tasks have no category and are excluded.")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir", type=Path, help="The --out-dir passed to eval_libero.py.")
    p.add_argument("--task-classification", type=Path, default=None, help="LIBERO-Plus task_classification.json.")
    args = p.parse_args(argv)

    summary = summarize(args.out_dir, args.task_classification)
    markdown = to_markdown(summary)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out_dir / "summary.md").write_text(markdown)
    print(markdown)
    if summary["incomplete_tasks"]:
        print(
            f"{len(summary['incomplete_tasks'])} tasks are incomplete; re-run the evaluation with the same output "
            "directory to finish them.",
            file=sys.stderr,
        )
    print(f"Wrote {args.out_dir / 'summary.json'} and {args.out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

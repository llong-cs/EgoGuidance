"""CPU-only tests for tools/analysis/extract_metrics.py."""

import csv

import pytest

TOOL = "tools/analysis/extract_metrics.py"

# Shape of a rich-formatted training log: the dict wraps over several lines and the source
# location is right-aligned on the marker line; tqdm snapshots are interleaved.
LOG = """\
[11/09 10:00:00] INFO     | >> Step 20, Loss:                                      train_starvla.py:2244
                          {'action_loss': 0.5, 'gen_loss':
                          1.25, 'video_gen_loss': 2.0e-1, 'total_loss': 0.95,
                          'learning_rate/vlm': 3e-05}
  1%|          | 20/200 [00:40<05:00,  2.00s/it]\r  1%|          | 21/200 [00:42<05:00,  2.00s/it]
  1%|          | 21/200 [00:43<05:00,  2.00s/it]\r  1%|          | 22/200 [00:45<05:00,  2.00s/it]
eval:  50%|#####     | 1/2 [00:01<00:01,  1.00s/it]
  1%|          | 24/200 [00:50<05:00,  2.00s/it]
[11/09 10:01:00] INFO     | >> Step 40, Loss: {'action_loss': 0.25, 'val/action_mse': 0.125}
"""


def test_parse_step_rows_handles_wrapped_and_single_line_records(load_by_path):
    em = load_by_path(TOOL)
    rows = dict(em.parse_step_rows(LOG))
    assert rows[20] == {
        "action_loss": 0.5,
        "gen_loss": 1.25,
        "video_gen_loss": 0.2,
        "total_loss": 0.95,
        "learning_rate/vlm": 3e-05,
    }
    assert rows[40] == {"action_loss": 0.25, "val/action_mse": 0.125}


def test_steptimes_use_first_appearance_and_skip_gaps(load_by_path):
    em = load_by_path(TOOL)
    txt = LOG.replace("\r", "\n")
    assert em.total_steps(txt) == 200
    # 20 -> 40 s, 21 -> 42 s (first appearance), 22 -> 45 s; 24 follows a gap and is not timed.
    assert em.steptimes(txt, 200) == {21: 2, 22: 3}
    assert em.to_sec("1:02:03") == 3723


def test_main_discovers_runs_and_writes_csv(load_by_path, tmp_path):
    em = load_by_path(TOOL)
    # scripts/train.sh layout: <run_root_dir>/<run_id>/logs/train_node<rank>_<timestamp>.log
    logs = tmp_path / "runs" / "my_run" / "logs"
    logs.mkdir(parents=True)
    (logs / "train_node0_20261109_100000.log").write_text(LOG)
    (logs / "train_node1_20261109_100000.log").write_text("not parsed: only node 0 logs are read")
    out = tmp_path / "csv"
    assert em.main([str(tmp_path / "runs"), str(out)]) == 0

    with (out / "my_run.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [r["step"] for r in rows] == ["20", "40"]
    assert rows[1]["val/action_mse"] == "0.125"
    with (out / "summary.csv").open() as f:
        summary = list(csv.DictReader(f))
    assert len(summary) == 1
    assert summary[0]["run"] == summary[0]["run_id"] == "my_run"
    assert summary[0]["last_step"] == "40"
    assert float(summary[0]["pct"]) == pytest.approx(20.0)
    assert (out / "my_run__steptime.csv").read_text().splitlines()[0] == "step,step_time_s"


def test_resumed_run_merges_launches_in_order(load_by_path, tmp_path):
    em = load_by_path(TOOL)
    logs = tmp_path / "nested" / "resumed_run" / "logs"
    logs.mkdir(parents=True)
    (logs / "train_node0_20261109_100000.log").write_text(LOG)
    # The resumed launch restarts the tqdm clock at its initial step and re-logs step 40.
    resumed = (
        " 20%|##        | 40/200 [00:00<05:00,  2.00s/it]\n"
        " 20%|##        | 41/200 [00:04<05:00,  2.00s/it]\n"
        "[11/09 12:00:00] INFO     | >> Step 40, Loss: {'action_loss': 0.2}\n"
        "[11/09 12:01:00] INFO     | >> Step 60, Loss: {'action_loss': 0.1}\n"
    )
    (logs / "train_node0_20261109_120000.log").write_text(resumed)
    runs = em.discover_runs(tmp_path)
    assert [(name, run_id, [p.name for p in paths]) for name, run_id, paths in runs] == [
        (
            "nested__resumed_run",
            "resumed_run",
            ["train_node0_20261109_100000.log", "train_node0_20261109_120000.log"],
        )
    ]
    row = em.extract_run(runs[0][2], tmp_path, runs[0][0], runs[0][1])
    assert row["logs"] == 2 and row["last_step"] == 60 and row["action_loss"] == 0.1
    with (tmp_path / "nested__resumed_run.csv").open() as f:
        rows = {r["step"]: r for r in csv.DictReader(f)}
    assert rows["40"]["action_loss"] == "0.2"  # the later launch wins
    assert rows["20"]["gen_loss"] == "1.25"  # steps only in the first launch are kept
    with (tmp_path / "nested__resumed_run__steptime.csv").open() as f:
        times = {int(r["step"]): int(r["step_time_s"]) for r in csv.DictReader(f)}
    assert times == {21: 2, 22: 3, 41: 4}  # no step is timed across the restart


def test_main_reports_when_nothing_is_parsed(load_by_path, tmp_path):
    em = load_by_path(TOOL)
    assert em.main([str(tmp_path), str(tmp_path / "csv")]) == 1

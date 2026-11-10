# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""LIBERO / LIBERO-Plus rollout client.

Connects to a running policy server (``deployment/model_server/server_policy.py``) and rolls out
episodes in the LIBERO simulator. The same script evaluates standard LIBERO and LIBERO-Plus (a
drop-in fork of the ``libero`` package); which one runs depends on the ``libero`` package installed
in the current environment, and ``--benchmark`` checks that it is the expected one.

Work is split into *units* ``(suite, task_id, [ep_lo, ep_hi))``. Every unit writes
``<out_dir>/episodes/<suite>_task<id>_ep<lo>-<hi>.json`` after each episode, so an interrupted run
resumes at episode granularity when it is started again with the same ``--out-dir``. Several
processes can share one ``--out-dir`` with ``--worker-index i --num-workers M``; each one takes every
M-th unit. ``summarize_results.py`` merges the per-episode outcomes into success rates.

Example (single process, all four suites, run from the repository root)::

    PYTHONPATH=. python examples/LIBERO/eval_files/eval_libero.py \\
        --port 10093 --out-dir playground/Checkpoints/<run_id>/eval/libero/steps_20000_pytorch_model
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import os
import pathlib
import sys
import time

import imageio
import numpy as np

# Print Python stacks on native crashes (SIGSEGV / SIGABRT in the renderer); enabled before the
# simulator imports so that crashes during import are covered too.
faulthandler.enable()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

from examples.LIBERO.eval_files.model2libero_interface import ModelClient  # noqa: E402

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]  # no motion, gripper open
LIBERO_ENV_RESOLUTION = 256  # resolution used to render the training data
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

# Per-suite episode budget (policy steps), set slightly above the longest training demonstration.
MAX_STEPS = {
    "libero_spatial": 220,  # longest training demo has 193 steps
    "libero_object": 280,  # longest training demo has 254 steps
    "libero_goal": 300,  # longest training demo has 270 steps
    "libero_10": 520,  # longest training demo has 505 steps
    "libero_90": 400,  # longest training demo has 373 steps
}
DEFAULT_TRIALS_PER_TASK = {"libero": 50, "libero_plus": 1}

# Settings that must be identical for every episode written to one --out-dir.
_GUARDED_CONFIG_KEYS = (
    "benchmark",
    "checkpoint",
    "num_trials_per_task",
    "num_steps_wait",
    "seed",
    "resolution",
    "mujoco_gl",
)
EXIT_RECYCLE = 3  # exit code asking the launcher to restart this worker (see --max-units-per-process)


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    """Map the policy's gripper output (1 = open, 0 = closed) to LIBERO's command (-1 = open, +1 = close)."""
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


def _get_libero_env(task, resolution: int, seed: int):
    """Build the off-screen LIBERO environment for ``task``; returns ``(env, task_description)``."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        # LIBERO-Plus string-matches perturbation markers in the file name, so pass a str, not a Path.
        bddl_file_name=str(task_bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)  # the seed affects object positions even with a fixed initial state
    return env, task_description


def run_episode(
    env,
    client: ModelClient,
    task_description: str,
    initial_state,
    max_steps: int,
    num_steps_wait: int,
    record_frames: bool = False,
):
    """Roll out one episode. Returns ``(success, num_policy_steps, front_frames)``."""
    client.reset(task_description=task_description)
    env.reset()
    obs = env.set_init_state(initial_state)

    t, step, done = 0, 0, False
    frames = []
    while t < max_steps + num_steps_wait:
        # Do nothing for the first few steps: the simulator drops the objects and they need to settle.
        if t < num_steps_wait:
            obs, _reward, done, _info = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        # Rotate both camera images by 180 degrees to match the training data.
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        if record_frames:
            frames.append(img)

        # Proprioceptive state is not sent: the released checkpoints are trained with include_state=false.
        action = np.asarray(client.step([img, wrist_img], str(task_description), step), dtype=np.float32)
        delta_action = np.concatenate([action[:3], action[3:6], _binarize_gripper_open(action[6:7])], axis=0)

        obs, _reward, done, _info = env.step(delta_action.tolist())
        if done:
            break
        t += 1
        step += 1
    return bool(done), step, frames


def _atomic_write_json(path: pathlib.Path, obj: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _write_video(path: pathlib.Path, frames) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    imageio.mimwrite(str(tmp), [np.asarray(x) for x in frames], fps=10)
    os.replace(tmp, path)


def _check_eval_config(out_dir: pathlib.Path, config: dict) -> None:
    """Create ``eval_config.json`` or verify that it matches this run (no mixing of settings/checkpoints)."""
    path = out_dir / "eval_config.json"
    try:
        with open(path, "x") as f:
            json.dump(config, f, indent=2)
        return
    except FileExistsError:
        pass
    for _ in range(50):  # another worker may be writing it right now
        try:
            existing = json.loads(path.read_text())
            break
        except json.JSONDecodeError:
            time.sleep(0.1)
    else:
        raise RuntimeError(f"Cannot parse {path}.")
    diff = {k: (existing.get(k), config.get(k)) for k in _GUARDED_CONFIG_KEYS if existing.get(k) != config.get(k)}
    if diff:
        raise SystemExit(
            f"{path} was written with different settings ({{key: (existing, current)}}: {diff}). "
            "Use a new --out-dir for a different checkpoint or protocol."
        )


def _completed_episodes(episodes_dir: pathlib.Path) -> set:
    """Return ``{(suite, task_id, episode)}`` already recorded in any unit file of this run."""
    done = set()
    for path in episodes_dir.glob("*.json"):
        try:
            rec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue  # a half-written file from a crash; its episodes are re-run
        for ep in rec.get("episodes", []):
            done.add((rec["suite"], int(rec["task_id"]), int(ep["episode"])))
    return done


def _build_units(task_suites: dict, num_trials: int, episode_chunk: int, max_tasks_per_suite: int) -> list:
    chunk = num_trials if episode_chunk <= 0 else min(episode_chunk, num_trials)
    units = []
    for suite_name, suite in task_suites.items():
        n_tasks = suite.n_tasks if max_tasks_per_suite <= 0 else min(max_tasks_per_suite, suite.n_tasks)
        for task_id in range(n_tasks):
            for lo in range(0, num_trials, chunk):
                units.append((suite_name, task_id, lo, min(lo + chunk, num_trials)))
    return units


def run_unit(client, suite, suite_name, task_id, ep_lo, ep_hi, done, args) -> int:
    """Run the missing episodes of one unit; returns the number of episodes executed."""
    todo = [e for e in range(ep_lo, ep_hi) if (suite_name, task_id, e) not in done]
    if not todo:
        return 0
    task = suite.get_task(task_id)
    initial_states = suite.get_task_init_states(task_id)
    if ep_hi > len(initial_states):
        raise ValueError(
            f"{suite_name}:{task_id} has {len(initial_states)} initial states, fewer than "
            f"--num-trials-per-task={args.num_trials_per_task}."
        )

    unit_path = args.out_dir / "episodes" / f"{suite_name}_task{task_id}_ep{ep_lo}-{ep_hi}.json"
    record = {"suite": suite_name, "task_id": task_id, "task": None, "ep_lo": ep_lo, "ep_hi": ep_hi, "episodes": []}
    if unit_path.exists():
        try:
            record["episodes"] = json.loads(unit_path.read_text()).get("episodes", [])
        except (OSError, json.JSONDecodeError):
            pass

    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    record["task"] = str(task_description)
    try:
        for episode_idx in todo:
            save_video = episode_idx < args.videos_per_task
            success, steps, frames = run_episode(
                env,
                client,
                task_description,
                initial_states[episode_idx],
                MAX_STEPS[suite_name],
                args.num_steps_wait,
                record_frames=save_video,
            )
            record["episodes"].append({"episode": episode_idx, "success": success, "steps": steps})
            _atomic_write_json(unit_path, record)
            if save_video and frames:
                suffix = "success" if success else "failure"
                _write_video(
                    args.out_dir / "videos" / suite_name / f"task{task_id}_ep{episode_idx}_{suffix}.mp4", frames
                )
            n_ok = sum(e["success"] for e in record["episodes"])
            logging.info(
                "[%s:%d] episode %d %s (unit %d/%d successes) | %s",
                suite_name,
                task_id,
                episode_idx,
                "success" if success else "failure",
                n_ok,
                len(record["episodes"]),
                task_description,
            )
    finally:
        env.close()  # release the renderer's GL context before the next unit builds a new env
    return len(todo)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=10093)
    p.add_argument("--out-dir", type=pathlib.Path, required=True, help="Results directory (resumable).")
    p.add_argument(
        "--benchmark",
        choices=sorted(DEFAULT_TRIALS_PER_TASK),
        default="libero",
        help="Expected benchmark; checked against the installed libero package.",
    )
    p.add_argument("--suites", nargs="+", default=list(SUITES), choices=sorted(MAX_STEPS))
    p.add_argument(
        "--num-trials-per-task",
        type=int,
        default=None,
        help="Episodes per task (default: 50 for LIBERO, 1 for LIBERO-Plus).",
    )
    p.add_argument("--num-steps-wait", type=int, default=10, help="No-op steps while objects settle.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--unnorm-key", default=None, help="dataset_statistics.json key (default: server's default).")
    p.add_argument("--episode-chunk", type=int, default=0, help="Episodes per unit; 0 = one unit per task.")
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--max-tasks-per-suite", type=int, default=0, help="Only the first N tasks per suite (smoke test).")
    p.add_argument(
        "--max-units-per-process",
        type=int,
        default=0,
        help=f"Exit with code {EXIT_RECYCLE} after N executed units so a launcher can restart the process "
        "(frees renderer memory on long LIBERO-Plus runs); 0 = never.",
    )
    p.add_argument("--videos-per-task", type=int, default=1, help="Save videos of the first N episodes of each task.")
    args = p.parse_args(argv)
    if args.num_trials_per_task is None:
        args.num_trials_per_task = DEFAULT_TRIALS_PER_TASK[args.benchmark]
    if not 0 <= args.worker_index < args.num_workers:
        p.error("--worker-index must be in [0, --num-workers).")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s | %(message)s",
        datefmt="%m/%d %H:%M:%S",
        force=True,
    )
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suites = {name: benchmark_dict[name]() for name in args.suites}
    # Standard LIBERO suites have 10 tasks; LIBERO-Plus suites have thousands of perturbed tasks.
    is_plus = any(suite.n_tasks > 10 for name, suite in task_suites.items() if name != "libero_90")
    if is_plus != (args.benchmark == "libero_plus"):
        counts = {name: suite.n_tasks for name, suite in task_suites.items()}
        raise SystemExit(
            f"--benchmark {args.benchmark}, but the installed libero package ({benchmark.__file__}) has task counts "
            f"{counts}. Activate the environment of the matching benchmark."
        )

    client = ModelClient(host=args.host, port=args.port, unnorm_key=args.unnorm_key)
    ckpt_path = client.server_metadata.get("ckpt_path")
    classification = pathlib.Path(get_libero_path("benchmark_root")) / "benchmark" / "task_classification.json"
    (args.out_dir / "episodes").mkdir(parents=True, exist_ok=True)
    _check_eval_config(
        args.out_dir,
        {
            "benchmark": args.benchmark,
            "checkpoint": os.path.realpath(ckpt_path) if ckpt_path else None,
            "num_trials_per_task": args.num_trials_per_task,
            "num_steps_wait": args.num_steps_wait,
            "seed": args.seed,
            "resolution": LIBERO_ENV_RESOLUTION,
            "mujoco_gl": os.environ.get("MUJOCO_GL", ""),
            "task_classification": str(classification) if classification.is_file() else None,
        },
    )

    units = _build_units(task_suites, args.num_trials_per_task, args.episode_chunk, args.max_tasks_per_suite)
    my_units = units[args.worker_index :: args.num_workers]
    done = _completed_episodes(args.out_dir / "episodes")
    logging.info(
        "Worker %d/%d: %d of %d units (%d episodes already recorded in %s)",
        args.worker_index,
        args.num_workers,
        len(my_units),
        len(units),
        len(done),
        args.out_dir,
    )

    executed, failed = 0, 0
    for i, (suite_name, task_id, ep_lo, ep_hi) in enumerate(my_units):
        try:
            n = run_unit(client, task_suites[suite_name], suite_name, task_id, ep_lo, ep_hi, done, args)
        except Exception:
            # Keep going with the other units; the failed one has no complete record and is retried on restart.
            logging.exception("[%s:%d] episodes [%d, %d) failed", suite_name, task_id, ep_lo, ep_hi)
            failed += 1
            continue
        if n:
            executed += 1
        if args.max_units_per_process and executed >= args.max_units_per_process and i + 1 < len(my_units):
            logging.info(
                "Executed %d units; exiting with code %d so the launcher restarts this worker.", executed, EXIT_RECYCLE
            )
            client.close()
            return EXIT_RECYCLE

    client.close()
    logging.info("Worker %d done: %d units executed, %d failed.", args.worker_index, executed, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

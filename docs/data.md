# Data

All configs reference data and weights relative to `playground/` (git-ignored).
This page describes what each data stream expects on disk. The repository
contains the training code only; preparing the egocentric clips and captions is
done beforehand.

```
playground/
├── Pretrained_models/
│   ├── Qwen3.5-2B/                      # backbone VLM (the folder name must contain "Qwen3.5")
│   ├── siglip2-so400m-patch16-256/      # default generation target
│   ├── dinov2-giant/                    # visual-target study only
│   ├── vjepa2-vitg-fpc64-256/
│   ├── lingbot-vision-vit-giant/
│   └── Wan2.2-TI2V-5B-Diffusers/
├── Datasets/
│   ├── DROID/                           # LeRobot v3, contains success/
│   ├── LEROBOT_LIBERO_DATA/             # LeRobot LIBERO, mixture libero_all
│   ├── Open-AoE/
│   │   ├── videos/                      # source videos
│   │   ├── index.jsonl                  # all released clips with their step descriptions
│   │   └── diversity/<subset>.jsonl     # data-diversity study (200K pairs each)
│   └── Ego4D/
│       ├── clips/                       # uniform 5 s clips
│       └── index_{raw,gate1,gate1_gate2}.jsonl
└── Checkpoints/                         # training outputs (<run_root_dir>/<run_id>/)
```

## Robot data

**DROID** ([nvidia/Cosmos3-DROID](https://huggingface.co/datasets/nvidia/Cosmos3-DROID),
LeRobot v3). Point `datasets.vla_data.data_root_dir` at the folder that contains
`success/`; the mixture `droid_success` reads the successful episodes. The
episodes are curated as in the report when the dataset is indexed
(`datasets.vla_data.episode_filter`, see
`starVLA/dataloader/gr00t_lerobot/episode_filter.py`): an episode is dropped if
none of its task strings is non-empty, if its action statistics are missing or
non-finite, or if none of its action dimensions changes. Both checks read the
LeRobot episode metadata (`meta/episodes/*/*.parquet`), so no pass over the frame
data is needed. The loader prints how many episodes were kept.

Observations: `exterior_1` (primary global view, also the generation target) and
the wrist camera. Actions: 6-D cartesian velocity + 1-D gripper position.

**LIBERO**: `bash examples/LIBERO/data_preparation.sh` downloads the four
LeRobot-format suites to `playground/Datasets/LEROBOT_LIBERO_DATA`; see
[`examples/LIBERO/README.md`](../examples/LIBERO/README.md).

## Egocentric video: `index.jsonl`

An egocentric stream (`dataset_py: video_datasets`) is described by one JSON
object per line. Two record formats are supported; one index must not mix them.
Fields not listed here are ignored.

| Field | Required | Meaning |
|---|---|---|
| `video` | yes | Path of the mp4, relative to `datasets.video_data.data_root_dir` (default: the directory of `index_jsonl`) or absolute. |
| `split` | no | The record is used if it equals `datasets.video_data.split` (default `train`) or is missing. |
| `step_description` | no | Caption used as the language instruction (field name: `datasets.video_data.text_key`). |

**Clip records** (pre-cut clip files, e.g. the 5 s Ego4D clips):

| Field | Required | Meaning |
|---|---|---|
| `total_frames` | yes | Number of decodable frames in the clip. The observation frame is drawn uniformly in `[0, total_frames - Δ)`, where Δ is the generation offset; shorter clips are skipped. |

**Interval records** (windows of longer source videos, e.g. Open-AoE):

| Field | Required | Meaning |
|---|---|---|
| `valid_start_s`, `valid_end_s` | yes | Interval in seconds of the source video; the observation and its future target both stay inside it. |
| `sampling_weight_s` | no | Relative probability of drawing the record (default: the sampleable span `valid_end_s - valid_start_s - τ`). |

```json
{"video": "ego4d_000123_clip007.mp4", "total_frames": 150, "split": "train", "step_description": "Cut the onion on the board."}
{"video": "sample_0001/raw_video.mp4", "valid_start_s": 12.0, "valid_end_s": 14.0, "split": "train", "step_description": "Pick up the cup."}
```

**Timing.** The generation target is aligned with the end of the action chunk,
a<sub>t+H</sub>: it lies τ = H / f<sub>robot</sub> ahead (8 DROID frames at 15 Hz =
0.533 s in pre-training; 8 LIBERO frames at 20 Hz = 0.4 s in fine-tuning), i.e.
Δ = ⌊τ·f + ½⌋ frames of a stream recorded at f = `dataset_fps` frames per second:
16 frames of a 30 fps egocentric video. With `num_future_frames: T`, frame k lies
at k·τ/T seconds.

**Open-AoE** clips and step descriptions are used as released, without the
Ego4D gates. The data-scaling configs train on a seeded fraction of
`index.jsonl` (`datasets.video_data.episode_fraction`); the data-diversity
configs point `index_jsonl` at a fixed-size subset (200K clip–description pairs)
that is balanced across scene, object or interaction clusters.

**Ego4D** videos are cut into uniform, non-overlapping 5 s clips, and two
cumulative quality gates select the curation levels used by
`configs/pretrain/ego4d_curation/` (all three indexes refer to the same clip
files):

- `raw`: every clip.
- `gate1`: Gate 1 (motion quality). Corner detection and optical-flow fitting check
  sampled points of the clip; the clip passes if the flow check passes for at
  least 85% of the evaluable points, for at least 3 points, with no more than 2
  consecutive failures.
- `gate1_gate2`: Gate 1 and Gate 2 (visible hands). A hand detector runs on every
  sampled frame; a box counts if its confidence is at least 0.30, its area is 2–50%
  of the image and it intersects the region [0.075, 0.075, 0.925, 1.0]. The clip
  passes if at least 10% of the frames (and at least one) contain a valid hand,
  with no more than 5 consecutive frames without one.

Each clip's `step_description` is one imperative English sentence generated from
15 frames sampled at 10-frame intervals.

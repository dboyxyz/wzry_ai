# Screen-Based Honor of Kings AI on Jetson

A visual game-agent project connecting an Android phone to an NVIDIA Jetson
Orin. The agent observes screen images, predicts actions, sends touch commands,
and records new trajectories for offline training.

The project combines a ResNet-101 visual encoder, a Transformer action policy,
a three-class ResNet state classifier, and temporal HUD OCR. State recognition
uses visual evidence for `normal`, `dead`, and `tower_attack`; changes in gold,
kills, and deaths provide event labels for reward preparation.

This repository is an English reorganization of the local Jetson project.
Datasets, generated reports, archived experiments, and
application-portfolio assets are kept outside version control.

For the motivation, individual contributions, results, and limitations, start
with [the project overview](docs/PROJECT_OVERVIEW.md).

## Project layout

| File | Purpose |
| --- | --- |
| `run_agent.py` | Screen capture, policy inference, touch control, manual takeover, recording |
| `prepare_features.py` | Precompute frozen ResNet-101 features for each run |
| `train_policy.py` | Offline reward-weighted action-policy training |
| `train_status.py` | Train/evaluate the three-class visual state model, grouped by run |
| `hud_ocr.py` | Numeric OCR and temporal gold/K/D event inference |
| `hybrid_rewards.py` | Combine visual states and OCR events into per-frame rewards |
| `review_states.py` | Generate paginated per-image HTML review pages |
| `convert_legacy.py` | Convert original data schemas and checkpoint parameter names |
| `policy_model.py`, `layers.py`, `sublayers.py`, `embeddings.py` | Transformer architecture |
| `linux_capture.py`, `linux_control.py` | X11 capture and scrcpy touch control |
| `config/*.json` | Stable action IDs, touch scripts, legacy migration mapping |

## Environment

The original system was developed on Jetson AGX Orin with an iQOO 10 running
Android 16. Use Python 3.10 or newer, an X11 desktop, ADB, scrcpy, and an Android
phone with USB debugging enabled. Install PyTorch and torchvision builds that
match your JetPack/CUDA environment, then install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Use the repository root as the working directory. This local checkout includes
three models in `weights/`: `policy.pt` (the currently deployed epoch-100 policy),
`status_best.pt` (the best three-state classifier), and
`resnet101-cd907fc2.pth` (the ImageNet feature encoder). Runtime and preprocessing
load the local encoder first, avoiding an initial download. The weights directory
is Git-ignored; a source-only clone needs these files supplied separately.

Default model paths are anchored to this project's `weights/` directory,
independent of the shell working directory. `run_agent.py` defaults to
`policy.pt`; state review and policy rewards default to `status_best.pt`;
the feature encoder uses `resnet101-cd907fc2.pth`. Visual evaluation with
`--eval-only` also uses `status_best.pt` unless `--resume` is specified.
New visual-model training still starts from its selected ImageNet backbone
unless a resume checkpoint is explicitly requested.

## Model downloads

Download the two project checkpoints and save them under `weights/` with the
filenames below. These files are ready for the English codebase and do not
need legacy conversion.

| Model | Download | Destination |
| --- | --- | --- |
| Action policy (epoch 100) | [policy.pt on Google Drive](https://drive.google.com/file/d/1puIuK1AzQGFqrV5oC_Fay2OEbV5Uiz1w/view?usp=drive_link) | `weights/policy.pt` |
| Three-class visual state classifier | [status_best.pt on Google Drive](https://drive.google.com/file/d/1mTXAdfmTz05bK9km_jG2A0HFwwIg3nTx/view?usp=drive_link) | `weights/status_best.pt` |

The ResNet-101 ImageNet encoder is loaded from
`weights/resnet101-cd907fc2.pth` when present. Otherwise, torchvision downloads
its pretrained weights automatically on first use; an Internet connection is
needed for that initial download.

## Data and model formats

```text
data/
  runs/
    RUN_ID/
      0.jpg
      1.jpg
      _actions.jsonl
      features.npz
  status/
    labels.jsonl
    RUN_ID_FRAME.jpg
weights/
  policy.pt
  status_best.pt
  resnet101-cd907fc2.pth
```

Each action-record line uses this schema:

```json
{"frame_id":"0","movement":"right","skill":"attack","control_mode":0}
```

`keep-move` means no new movement command; preprocessing carries forward the
previous movement. Action names combine movement and skill using one underscore,
for example `up-right_skill1`. Preserve `config/action_to_id.json`: its 130 IDs
are part of checkpoint compatibility. Token 128 is the historical start token.

`features.npz` contains `features` (N x 73728), `actions` (N), and
`control_modes` (N). The control-mode field is a legacy recording flag, **not**
an episode termination signal; do not use it as PPO `done`.

New recordings store the current frame's AI-mode flag (`0` enabled, `1`
disabled) and the resolved movement direction. AI-enabled frames can still
contain manual overrides. Legacy recordings retain their original mode flag,
which may lag by one frame; it cannot measure manual takeover frequency.

Status labels are JSON Lines, such as `{"RUN_ID_120":"tower_attack"}`.
Offline augmented image keys use `__aug_` or `__aug2_`. Validation uses only
original images from held-out runs, keeping augmentations with their source run.

## Migrate existing local artifacts

Use new output paths; the converter refuses to overwrite them:

```bash
python3 convert_legacy.py runs /path/to/original/runs data/runs
python3 convert_legacy.py labels /path/to/original/labels.json data/status/labels.jsonl
python3 convert_legacy.py policy /path/to/original/policy.pt weights/policy.pt
python3 convert_legacy.py status /path/to/original/best.pt weights/status_best.pt
```

The runs converter copies JPG frames and translates action records. It rebuilds
no features: run preprocessing next. Copy status images separately into
`data/status/` with their original filenames. The policy converter changes two
parameter-name prefixes while preserving tensor values and action IDs. The
status converter preserves model tensors and translates its ordered label list.
`config/legacy_mapping.json` stores Unicode-escaped legacy names exclusively for
backward compatibility. It is not a second runtime action vocabulary.

## Run and record

Start a scrcpy display window, then adjust the capture rectangle to its game
content (the historical capture size is 960 x 432):

```bash
scrcpy --window-title wzry_ai
python3 run_agent.py --device cuda:0 --weights-dir weights \
  --model-name policy.pt --top 65 --left 60 --width 960 --height 432 --save-data
```

AI control starts enabled. Press `i` to toggle it, `Esc` to stop. `W/A/S/D`
control movement, `q` stops movement, the up arrow attacks, `j/k/l` cast skills,
`f` recalls, `g` heals, and `h` triggers the summoner skill. Manual input can
override AI control. The model history uses the executed command after manual
overrides. Recording requires `--save-data`.

Touch coordinates in `config/touch_scripts.json` are device/layout specific.
The capture and touch-control video streams use different resolutions; adjust
their coordinates together for a different phone or UI layout. The runtime
preserves the legacy width/height ordering of policy image preprocessing for
compatibility with existing policy weights.

## Inspect states before training

```bash
python3 review_states.py --all-runs --device cuda:0 \
  --state-weights weights/status_best.pt --output-dir outputs/state_review
```

Open `outputs/state_review/index.html`. Each run has paginated images, visual
class probabilities, OCR gold/K/D/A values, temporal events, and final reward
labels. `--reuse` reuses completed `rewards.json` files; use a new output folder
after changing model weights, thresholds, or source images. Pages reference
source images through relative paths, so retain the data directory when sharing
the report. State review does not require feature NPZ files.

OCR selects the highest-confidence result from the raw crop and a 4x cubic
variant, with a default minimum confidence of 0.1. Temporal inference retains
the tested extra-digit gold correction and suppresses gold events close to K/D
events. Missing frames split sequences. Inspect results manually: spending can
hide gold gains and OCR counter errors can generate false events.

## Prepare features and train the policy

```bash
python3 prepare_features.py --data-dir data/runs --device cuda:0 \
  --batch-size 4 --feat-dtype float16

python3 train_policy.py --all-runs --data-dir data/runs --device cuda:0 \
  --policy-weights weights/policy.pt --state-weights weights/status_best.pt \
  --epochs 10 --save-every 1 --reward-alignment next-frame
```

`--all-runs` includes every run directory and checks all feature caches;
`--run-id RUN_A RUN_B` selects a subset. `next-frame` associates the next
observation reward with the action taken after the current screenshot. Use
`same-frame` only when that matches the recording convention. For a smoke test,
add `--max-frames 20 --epochs 1 --max-updates 1`. `--prepare-only` generates
rewards without training and supports CPU.

Rewards are prepared once per invocation and reused across epochs. The current
local reward scale is preserved in `hybrid_rewards.REWARDS`; review it before
training. Event priority is killed, hero kill, gold event, then the visual state.
Checkpoints and logs go to a new timestamped `hybrid_policy_runs/` directory.

The policy objective is reward-weighted log likelihood on recorded actions,
not standard on-policy PPO. In particular, signed negative weights can drive
the objective down without improving gameplay, so lower training loss is not
a checkpoint-quality metric. Evaluate actual gameplay and action distributions
before selecting a policy. The current architecture also multiplies action
embeddings by zero; retaining action history currently preserves the interface
but does not make that history influence the logits. The value head has no
value-loss training in this workflow. Online rollouts with old log probabilities,
GAE, valid terminal flags, and PPO clipping remain future work.

## Train and evaluate the visual state model

```bash
python3 train_status.py --mode full --label-json data/status/labels.jsonl \
  --image-dir data/status --backbone resnet50 --image-width 384 --image-height 192 \
  --device cuda:0 --require-cuda --include-augmented --epochs 30

python3 train_status.py --eval-only --eval-all --resume weights/status_best.pt \
  --label-json /path/to/test/labels.jsonl --image-dir /path/to/test/images \
  --export-errors-dir outputs/status_errors --device cuda:0
```

The trainer supports balanced sampling, grouped validation, mixed precision,
early stopping, epoch history, and selection by validation macro F1. Use the
same backbone and input size as the checkpoint when evaluating; specify those
options explicitly for nondefault checkpoints.

## Verification

```bash
python3 -m compileall -q .
python3 -m unittest discover -s tests -v
```

The CPU tests cover action ID round trips, causal policy masks, feature/action
alignment, legacy label conversion, temporal OCR edge cases, and HTML export.
Repository export was also checked against the original policy: after converting
parameter prefixes, logits and value outputs matched exactly in a small model.
The converted three-state checkpoint was loaded and used on four real source
images to generate English review pages. A new live phone session was not run
as part of the repository cleanup.

# 🧾 Bill Detection System (YOLO)

Real-time detection of a **hand holding a bill** (the `hand_bill` class) with a single
Ultralytics YOLO model. Built for CCTV cameras, but works on images, videos, built-in
webcams, and Android phone cameras (DroidCam / IP Webcam).

```
Input frame ──► [hand_bill YOLO] ──► hand-holding-bill boxes ──► annotated output
```

---

## 📑 Table of Contents

- [✨ Features](#-features)
- [🚀 Quick start](#-quick-start)
- [📷 Usage](#-usage)
  - [Images](#images)
  - [Video](#video)
  - [Webcam / CCTV](#webcam--cctv)
  - [Android phone (DroidCam)](#android-phone-droidcam)
  - [Web test UI](#web-test-ui-browser--images--rtspcctv-streams)
- [⚙️ CLI options](#%EF%B8%8F-cli-options-mainpy)
- [🧠 Models](#-models)
- [📊 Benchmarks](#-benchmarks)
- [🏋️ Training](#%EF%B8%8F-training)
- [🗂️ Datasets](#-datasets)
- [📁 Project structure](#-project-structure)
- [📄 License](#-license)

---

## ✨ Features

- **One model, one class** — no hand + denomination pipeline; each box *is* a "hand holding a bill".
- **Confidence-colored boxes** — 🟢 green ≥ 0.6, 🟠 orange < 0.6, exact score on each label.
- **Focused boxes** — every box is tightened toward the bill (`--box-inset`), so full-frame detections don't look sloppy.
- **"BILL DETECTED xN" banner** — colored strip across the top when a bill is found (color = best confidence that frame).
- **Works everywhere** — image, folder, video, built-in webcam, CCTV RTSP/HTTP stream, DroidCam, IP Webcam.
- **Multi-scale inference (opt-in)** — runs at **416 + 640** and NMS-merges results. Most accurate mode: catches every bill the single scales do *plus* phone-photo boxes each scale misses (~2–3× slower, so live streams stay single-scale).
- **Web test UI** — zero-dependency drag-and-drop browser interface for images *and* live RTSP/CCTV streams. Includes a **high-precision mode** (two models must agree), **temporal confirmation** (bill must be seen on 2 consecutive frames), and a **bill filter** with three independent checks: **shape**, **hand nearby**, and **white paper**.

---

## 🚀 Quick start

```bash
# one-time setup
python3 -m venv .venv
./.venv/bin/pip install -r examples/YOLO-Hand-Bill-Detection/requirements.txt

# try it on the bundled test images with the default model
./.venv/bin/python examples/YOLO-Hand-Bill-Detection/main.py \
  --weights models/hand_bill_detector_v4.pt --source test_images/ --save
```

---

## 📷 Usage

> All commands run from the repo root (`bill-detection-git/`).

### Images

```bash
# single image
./.venv/bin/python examples/YOLO-Hand-Bill-Detection/main.py \
  --weights models/hand_bill_detector.pt \
  --source path/to/image.jpg --save

# whole folder of images
./.venv/bin/python examples/YOLO-Hand-Bill-Detection/main.py \
  --weights models/hand_bill_detector_v2.pt \
  --source test_images/ --save --output-dir runs/image_test

# most accurate: merge 416 + 640 (slower, but catches phone photos the
# single scales miss)
./.venv/bin/python examples/YOLO-Hand-Bill-Detection/main.py \
  --weights models/hand_bill_detector_v4.pt --source test_images/ --save --multi-scale

# popup window while running (press q to quit)
./.venv/bin/python examples/YOLO-Hand-Bill-Detection/main.py \
  --weights models/hand_bill_detector.pt --source test_images/ --show
```

### Video

```bash
./.venv/bin/python examples/YOLO-Hand-Bill-Detection/main.py \
  --weights models/hand_bill_detector.pt \
  --source path/to/clip.mp4 --save --output-dir runs/video_test
```

### Webcam / CCTV

```bash
# built-in webcam
./.venv/bin/python examples/YOLO-Hand-Bill-Detection/main.py \
  --weights models/hand_bill_detector.pt --source 0 --show

# CCTV RTSP stream
./.venv/bin/python examples/YOLO-Hand-Bill-Detection/main.py \
  --weights models/hand_bill_detector.pt \
  --source 'rtsp://user:pass@192.168.0.50:554/stream1' --show
```

### Android phone (DroidCam)

```bash
# 1. install the DroidCam app on the phone, start it (note the Wi-Fi IP)
# 2. one-shot setup + live detection (loads driver, waits for phone, verifies frames):
./droidcam_start.sh
# or step by step:
sudo ./examples/YOLO-Hand-Bill-Detection/setup_droidcam.sh   # loads v4l2loopback driver
droidcam-cli <phone-ip> 4747 /dev/video2 &
./.venv/bin/python examples/YOLO-Hand-Bill-Detection/live_detect.py \
  --url /dev/video2 --weights models/hand_bill_detector_v2.pt
```

### Web test UI (browser: images + RTSP/CCTV streams)

```bash
./.venv/bin/python examples/YOLO-Hand-Bill-Detection/test_ui.py
# then open http://127.0.0.1:8000
```

- **Images:** drag & drop a photo, pick a model + threshold, hit **Detect**.
- **Live:** paste an RTSP/HTTP stream URL (`rtsp://user:pass@ip:554/stream1`) — or a local video
  file path — into the RTSP field and hit **Start**. Save camera URLs (**＋ Save**) to keep a
  quick-pick list across sessions. The annotated stream appears live (MJPEG).

Zero new dependencies (Python standard library), same drawing pipeline as the CLI scripts.
For the most accurate image tests, tick **Multi-scale — merge 416 + 640** (live streams stay
single-scale for smoothness).

---

## ⚙️ CLI options (main.py)

| Flag | Default | Description |
|---|---|---|
| `--weights` | (required) | Trained `hand_bill` weights |
| `--source` | `0` | Image, folder, video, URL, webcam index |
| `--conf` | `0.25` | Confidence threshold |
| `--imgsz` | `640` | Inference size |
| `--multi-scale` | off | Run at 416 **and** 640 and merge — most accurate, ~2–3× slower |
| `--box-inset` | `0.10` | How much to shrink boxes toward their center (0 = off) |
| `--shape-filter` / `--no-shape-filter` | on | Keep only elongated (bill-shaped) detections |
| `--hand-check` / `--no-hand-check` | on | Require skin (a hand) overlapping or next to the bill; auto-disabled on very dark frames (night/IR) |
| `--white-check` / `--no-white-check` | on | Require white (desaturated, bright) bill paper inside the box |
| `--device` | auto | `cpu`, `0`, `mps` |
| `--save` / `--show` | off | Save outputs / show live window |
| `--output-dir` | `runs/cctv` | Where outputs go |

---

## 🧠 Models

| File | Notes |
|---|---|
| `models/hand_bill_detector_v4.pt` (v4) | **Default.** Finetune of v2 on the newer dataset; **two classes** (`hand_bill` + `no_bill`). Catches everything v2 does **plus** phone-photo styles; zero false alarms on the test set |
| `models/hand_bill_detector_v2.pt` (v2) | High precision; misses phone-photo frames (001.jpg, Media.jpeg, test_bill3.png) |
| `models/hand_bill_detector.pt` (v1) | High recall; catches more (also more false alarms) |
| `models/hand_bill_detector_last.pt` | Last checkpoint from training; behaves like v1 |
| `models/hand_bill_detector_v3.pt` (v3) | Experiment @512; catches phone photos only at low confidence — not recommended |
| `models/hand_bill_detector_v3_last.pt` | v3 last checkpoint (not recommended) |

**v4 is the default model** in the UI and for live streams — it is the accuracy winner, promoted
from the `local_runs/detect/local_runs/bill_hand-5` run. Only class-0 (`hand_bill`) boxes are
reported; the `no_bill` class the model learned keeps false alarms at zero.

---

## 📊 Benchmarks

### Labeled test set (conf 0.25)

| Model | Filter | Accuracy | Precision | Recall |
|---|---|---|---|---|
| v4 (default) | ON | **100%** | **100%** | **100%** |
| v2 | ON | 100% | 100% | 100% |
| v2 | OFF | 83% | 73% | 100% |
| v1 / last | — | 44% | 44% | 100% |

### Phone-photo recall (boxes kept by the filter; top confidence in parentheses)

| Image | v2 | v4 @416 | v4 @640 | v4 multi-scale |
|---|---|---|---|---|
| 001.jpg | 0 | 2 (0.63) | 3 (0.58) | **4 (0.63)** |
| Media.jpeg | 0 | 4 (0.54) | 4 (0.38) | **5 (0.54)** |
| test_bill3.png | 0 | 0 | 0 | 0 |
| test_bill5.png | 1 (0.72) | 1 (0.40) | 2 (0.40) | **2 (0.40)** |

**Multi-scale is never worse than the best single scale** and adds boxes on the hard
phone photos — that's why it's the recommended setting in the UI for image testing.

### The bill filter

On by default, three independent checks (all toggleable in the UI / CLI):

1. **Shape** — elongated (bill-shaped) boxes pass; **near-square boxes pass only when a hand is
   present**, because a hand + note together fill a near-square region (this fixed real bills
   being dropped). Tiny slivers and near-full-frame boxes are always rejected.
2. **Hand nearby** — skin detected in YCrCb space around the box; auto-disables on very dark
   frames so night/IR CCTV is unaffected.
3. **White paper** — bills are white/desaturated with dark print; strongly colored or near-black
   boxes are rejected; also auto-disables on dark IR frames.

The filter scores **100% precision and 100% recall on the labeled test set** (8/8 bills caught,
10/10 clean images ignored) — the best result measured so far, with or without multi-scale.
Of the phone photos, `001.jpg` and `Media.jpeg` are now caught at every scale with v4;
`test_bill3.png` still returns 0 boxes because the *model* never fires on it — only retraining
with that photo style fixes that (multi-scale can only merge boxes the model actually produces).

Training-validation metrics (yolo11n @416, 100 epochs): mAP50 **67%**, mAP50-95 32%.

---

## 🏋️ Training

```bash
yolo detect train data=bill_hand_dataset/data.yaml model=yolo11n.pt epochs=100 imgsz=416
```

Colab notebooks for GPU training: `examples/YOLO-Hand-Bill-Detection/train_colab.ipynb`
and `train_bill_hand_v2_colab_git_repo_ml.ipynb`.

---

## 🗂️ Datasets

| Dataset | Size | Notes |
|---|---|---|
| `bill_hand_dataset/` | main | Original labeled dataset (tracked) |
| `bill_hand_dataset_v5/` | 281 train / 13 val | v4 dataset + phone-photo styles (`001.jpg`, `Media.jpeg`, `test_bill3.png`, `test_bill5.png`) with v1 pseudo-labels, so a fine-tune teaches the model those styles |
| `bill_hand_dataset_v6/` | 278 train / 13 val | Latest iteration |

Auto-built datasets are generated by `build_v3_dataset.py` / `build_v5_dataset.py`
(v3/v4 datasets are gitignored as regenerable).

---

## 📁 Project structure

```
models/                        trained weights (.pt)
test_images/                   labeled test photos (with_bill_* / without_bill_*) + previews
bill_hand_dataset*/            labeled datasets + training data
build_v3_dataset.py            build the v3 dataset
build_v5_dataset.py            build the v5 dataset (adds phone-photo pseudo-labels)
examples/YOLO-Hand-Bill-Detection/
  main.py                      image / video / webcam / CCTV detection
  live_detect.py               live streams (DroidCam, IP Webcam, CCTV)
  ipcam_test.py                IP Webcam phone stream
  test_ui.py + test_ui.html    web UI for testing images + live streams
  compare_models.py            side-by-side model comparison
  droidcam_start.sh            one-shot DroidCam setup + detection
  setup_droidcam.sh            sudo driver setup for DroidCam
  train_colab.ipynb            GPU training notebook
```

---

## 📄 License

Ultralytics is AGPL-3.0; see the Ultralytics license for commercial terms.

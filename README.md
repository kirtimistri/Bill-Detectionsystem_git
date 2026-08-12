# 🧾 Bill Detection System (YOLO)

Real-time detection of a **hand holding a bill** (the `hand_bill` class) using a single
Ultralytics YOLO model. Built for CCTV cameras, but usable on images, videos, built-in
webcams, and Android phone cameras (DroidCam / IP Webcam).

```
Input frame ──► [hand_bill YOLO] ──► hand-holding-bill boxes ──► annotated output
```

## ✨ Features

- **One model, one class** — no hand + denomination pipeline; each box *is* a "hand holding a bill".
- **Confidence-colored boxes** — 🟢 green ≥ 0.6, 🟠 orange < 0.6, with the exact score on each label.
- **Focused boxes** — every box is tightened toward the bill (`--box-inset`), so full-frame detections don't look sloppy.
- **"BILL DETECTED xN" banner** — a colored strip across the top when a bill is found (color = best confidence that frame).
- **Works everywhere** — single image, folder of images, video file, built-in webcam, CCTV RTSP/HTTP stream, DroidCam, IP Webcam.
- **Web test UI** — zero-dependency drag-and-drop interface for testing images in the browser.

## 🧠 Models

| File | Notes |
|---|---|
| `models/hand_bill_detector.pt` (v1) | High recall; catches more (also more false alarms) |
| `models/hand_bill_detector_v2.pt` (v2) | High precision; zero false alarms on the test set, but misses some real-world frames |
| `models/hand_bill_detector_last.pt` | Last checkpoint from training; behaves like v1 |

Measured on the labeled test set (10 with-bill + 10 without-bill photos, conf 0.25):

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| v2 | **90%** | 100% | 80% |
| v1 / last | 50% | 50% | 100% |

Training-validation metrics (yolo11n @416, 100 epochs): mAP50 **67%**, mAP50-95 32%.

## 🚀 Quick start

```bash
# one-time setup
python3 -m venv .venv
./.venv/bin/pip install -r examples/YOLO-Hand-Bill-Detection/requirements.txt
```

## 📷 Usage

All commands run from the repo root (`bill-detection-git/`).

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

### Web test UI (browser, images only)

```bash
./.venv/bin/python examples/YOLO-Hand-Bill-Detection/test_ui.py
# then open http://127.0.0.1:8000 — drag & drop an image, pick a model + threshold, hit Detect.
```

Zero new dependencies (Python standard library). Uses the same drawing pipeline as the CLI scripts,
so results match the CCTV/live use case.

## ⚙️ Options (main.py)

| Flag | Default | Description |
|---|---|---|
| `--weights` | (required) | Trained `hand_bill` weights |
| `--source` | `0` | Image, folder, video, URL, webcam index |
| `--conf` | `0.25` | Confidence threshold |
| `--imgsz` | `640` | Inference size |
| `--box-inset` | `0.10` | How much to shrink boxes toward their center (0 = off) |
| `--device` | auto | `cpu`, `0`, `mps` |
| `--save` / `--show` | off | Save outputs / show live window |
| `--output-dir` | `runs/cctv` | Where outputs go |

## 🏋️ Training

```bash
yolo detect train data=bill_hand_dataset/data.yaml model=yolo11n.pt epochs=100 imgsz=416
```

Colab notebooks for GPU training: `examples/YOLO-Hand-Bill-Detection/train_colab.ipynb`.

## 📁 Project structure

```
models/                        trained weights (.pt)
test_images/                   labeled test photos (with_bill_* / without_bill_*) + previews
bill_hand_dataset/             dataset, labels, training scripts
examples/YOLO-Hand-Bill-Detection/
  main.py                      image / video / webcam / CCTV detection
  live_detect.py               live streams (DroidCam, IP Webcam, CCTV)
  ipcam_test.py                IP Webcam phone stream
  test_ui.py                   web UI for testing images
  droidcam_start.sh            one-shot DroidCam setup + detection
  setup_droidcam.sh            sudo driver setup for DroidCam
```

## 📄 License

Ultralytics is AGPL-3.0; see the Ultralytics license for commercial terms.

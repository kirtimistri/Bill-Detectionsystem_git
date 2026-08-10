# YOLO Hand → Bill Two-Stage Detection

Two-stage detection pipeline built with Ultralytics YOLO: it first detects **hands** in the full frame,
**crops** each hand region, and then runs a second model that detects **bills/currency inside the crop only**.

<p align="center">
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" width="50%" alt="Ultralytics YOLO">
</p>

## Why two stages?

Running the bill detector on a small hand crop instead of the full frame:

- **Reduces false positives** – the bill model only ever sees a close-up of a hand, so background clutter
  (posters, screens, product packaging) can't trigger detections.
- **Improves small-object accuracy** – the bill is effectively upscaled inside the crop, which is exactly
  where small-object detectors struggle on full frames.
- **Is faster for the second stage** – inference on small crops is cheaper than on the full frame.

## Pipeline

1. **Stage 1 – Hand detection:** `hand_model(frame)` → hand boxes in frame coordinates.
2. **Crop:** each hand box is expanded by `--margin` (default 10%) and clipped to the frame.
3. **Stage 2 – Bill detection:** all hand crops are batched through `bill_model(crops)` in one inference call.
4. **Map back:** bill boxes are translated from crop coordinates to frame coordinates and drawn on the output.

```
Input frame ──► [Hand YOLO] ──► hand boxes ──► crop each hand ──► [Bill YOLO] ──► bill boxes ──► annotate frame
```

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py --hand-weights path/to/hand.pt --bill-weights path/to/bill.pt --source 0
```

| Argument          | Default          | Description                                                        |
| ----------------- | ---------------- | ------------------------------------------------------------------ |
| `--hand-weights`  | (required)       | Stage-1 YOLO model that detects hands                              |
| `--bill-weights`  | (required)       | Stage-2 YOLO model that detects bills/currency                     |
| `--source`        | `0`              | Image, directory, video, URL or webcam index (`0`)                 |
| `--device`        | `""`             | `cpu`, `0`, `mps`, or empty for auto-detection                     |
| `--hand-conf`     | `0.25`           | Confidence threshold for hand detection                            |
| `--bill-conf`     | `0.25`           | Confidence threshold for bill detection                            |
| `--margin`        | `0.1`            | Extra crop size around the hand box (fraction of box size)         |
| `--imgsz`         | `640`            | Inference size for both models                                     |
| `--hand-classes`  | `None`           | Optional stage-1 class ids treated as hands, e.g. `--hand-classes 0 1` |
| `--save`          | `False`          | Save annotated images or an annotated video                        |
| `--save-crops`    | `False`          | Also save hand/bill crops to the output directory                  |
| `--show`          | `False`          | Display the annotated frame in a window (press `q` to quit)        |
| `--output-dir`    | `runs/two_stage` | Where saved outputs are written                                    |

### Examples

```bash
# Live webcam
python main.py --hand-weights hand.pt --bill-weights bill.pt --source 0 --show

# Single image
python main.py --hand-weights hand.pt --bill-weights bill.pt --source path/to/image.jpg --save

# Folder of images
python main.py --hand-weights hand.pt --bill-weights bill.pt --source path/to/images/ --save

# Video, saving the annotated result
python main.py --hand-weights hand.pt --bill-weights bill.pt --source path/to/video.mp4 --save

# Use only class ids 0 and 1 from the hand model, wider crop margin, stricter bill threshold
python main.py --hand-weights hand.pt --bill-weights bill.pt --source 0 --hand-classes 0 1 --margin 0.15 --bill-conf 0.5
```

## Getting the models

The script does not ship with hand or bill weights — both are required arguments.

- **Hand model:** train a YOLO model on a hand dataset (e.g. [EgoHands](https://www.kaggle.com/datasets/kylegibson/egohands),
  [Hand Gesture Recognition](https://universe.roboflow.com/), or hand-object interaction datasets), or fine-tune
  `yolo11n.pt` on your own labeled hand images. Export/obtain the `.pt` weights and pass the path.
- **Bill model:** train on a currency dataset (e.g. Roboflow Universe "currency detection" / "money detector"
  projects) in the same way. If your model has multiple classes (e.g. `10`, `20`, `50`, `100`, `USD`, `EUR`),
  class names from the model are shown on the boxes automatically.

If your hand model was trained on more than just hands (e.g. hands + faces), use `--hand-classes` to select the
class ids that correspond to hands.

## Output

- Annotated frames show the **hand box (blue)**, the **expanded stage-2 crop region (orange)**, and **bill boxes
  (green)** with their class name and confidence.
- With `--save`, images are written to `runs/two_stage/` and videos to `runs/two_stage/<source>.mp4`.
- With `--save-crops`, the actual hand and bill crops are written to `runs/two_stage/crops/` for inspection.

## License

Ultralytics offers two licensing options to accommodate diverse use cases:

- **AGPL-3.0 License**: This [OSI-approved](https://opensource.org/license/agpl-v3/) open-source license is ideal
  for students and enthusiasts, promoting open collaboration and knowledge sharing. See the
  [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file for details.
- **Enterprise License**: Designed for businesses, this license permits commercial utilization of Ultralytics
  software. Contact [Ultralytics Licensing](https://ultralytics.com/license) for more details.

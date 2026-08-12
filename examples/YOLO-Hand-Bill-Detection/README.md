# YOLO CCTV Hand → Bill Detection

Single-stage detection built with Ultralytics YOLO: one model directly detects a **hand holding a bill**
(the `hand_bill` class). No money-denomination or two-stage pipeline logic.

## Why single-stage?

- **Direct answer** – each detection is already "a hand holding a bill", which is exactly the CCTV question.
- **Simpler** – no hand crop + denomination model, no margin tuning, one model to maintain.
- **Faster** – one inference call per frame instead of hand detection plus per-crop bill detection.

## Pipeline

```
Input CCTV frame ──► [hand_bill YOLO] ──► hand-holding-bill boxes ──► annotate frame
```

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py --weights path/to/best.pt --source 0
```

| Argument       | Default         | Description                                            |
| -------------- | --------------- | ------------------------------------------------------ |
| `--weights`    | (required)      | YOLO model trained on the `hand_bill` class            |
| `--source`     | `0`             | Image, directory, video, URL or webcam index (`0`)     |
| `--device`     | `""`            | `cpu`, `0`, `mps`, or empty for auto-detection         |
| `--conf`       | `0.25`          | Confidence threshold for detections                    |
| `--imgsz`      | `640`           | Inference size (pixels)                                |
| `--save`       | `False`         | Save annotated images or an annotated video            |
| `--show`       | `False`         | Display the annotated frame in a window (press `q`)    |
| `--output-dir` | `runs/cctv`     | Where saved outputs are written                        |

### Examples

```bash
# CCTV video, saving the annotated result
python main.py --weights ../../../models/hand_bill_detector.pt --source path/to/video.mp4 --save

# Live webcam
python main.py --weights ../../../models/hand_bill_detector.pt --source 0 --show

# Single image
python main.py --weights ../../../models/hand_bill_detector.pt --source path/to/image.jpg --save

# Folder of images
python main.py --weights ../../../models/hand_bill_detector.pt --source path/to/images/ --save
```

## Getting the model

Train a YOLO model on a `hand_bill` dataset (images where a hand is holding a bill, boxed around the bill):

```bash
yolo detect train data=bill_hand_dataset/data.yaml model=yolo11n.pt epochs=100 imgsz=416
```

## Output

- Annotated frames show **green hand-holding-bill boxes** with confidence.
- With `--save`, images are written to `runs/cctv/` and videos to `runs/cctv/<source>.mp4`.
- A summary line reports how many frames contained a hand holding a bill.

## License

Ultralytics offers two licensing options to accommodate diverse use cases:

- **AGPL-3.0 License**: This [OSI-approved](https://opensource.org/license/agpl-v3/) open-source license is ideal
  for students and enthusiasts, promoting open collaboration and knowledge sharing. See the
  [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file for details.
- **Enterprise License**: Designed for businesses, this license permits commercial utilization of Ultralytics
  software. Contact [Ultralytics Licensing](https://ultralytics.com/license) for more details.

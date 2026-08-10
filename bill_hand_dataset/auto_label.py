import os
import sys

from PIL import Image
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "raw")
MODEL = os.path.join(BASE, "models", "yolo11m_finetuned.pt")
OUT = os.path.join(BASE, "auto_labels")
CONF = 0.05
IOU = 0.5


def main():
    os.makedirs(OUT, exist_ok=True)
    model = YOLO(MODEL)

    labeled = 0
    unlabeled = []
    boxes_total = 0
    stems = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(RAW)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    for stem in stems:
        img_path = next(
            os.path.join(RAW, stem + ext)
            for ext in (".jpg", ".jpeg", ".png")
            if os.path.exists(os.path.join(RAW, stem + ext))
        )
        with Image.open(img_path) as im:
            W, H = im.size
        result = model(img_path, conf=CONF, iou=IOU, verbose=False)[0]
        boxes = result.boxes
        n = 0 if boxes is None else len(boxes)
        lines = []
        if boxes is not None and n:
            xyxy = boxes.xyxy.numpy()
            for x1, y1, x2, y2 in xyxy:
                x1 = max(0.0, min(W, float(x1)))
                y1 = max(0.0, min(H, float(y1)))
                x2 = max(0.0, min(W, float(x2)))
                y2 = max(0.0, min(H, float(y2)))
                if x2 <= x1 or y2 <= y1:
                    continue
                xc = (x1 + x2) / 2 / W
                yc = (y1 + y2) / 2 / H
                bw = (x2 - x1) / W
                bh = (y2 - y1) / H
                lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            boxes_total += len(lines)
        with open(os.path.join(OUT, stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        if lines:
            labeled += 1
        else:
            unlabeled.append(stem + ".jpg")

    print(f"labeled images: {labeled}/{len(stems)}, boxes total: {boxes_total}")
    print("images with NO detection (needs manual review):")
    for name in unlabeled:
        print(f"  raw/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

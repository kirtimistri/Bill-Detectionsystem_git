import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "bill_hand_dataset" / "yolo"
TEST_IMAGES = ROOT / "test_images"
DST = ROOT / "bill_hand_dataset_v3"
V2 = ROOT / "models" / "hand_bill_detector_v2.pt"
CONF = 0.5
IMG_SIZE = 640

for split in ("train", "val"):
    for sub in ("images", "labels"):
        (DST / sub / split).mkdir(parents=True, exist_ok=True)


def copy_phone(phone_split, yolo_split):
    src_imgs = SRC / "images" / phone_split
    src_lbls = SRC / "labels" / phone_split
    for img in sorted(src_imgs.glob("*.*")):
        stem = img.stem
        lbl = src_lbls / f"{stem}.txt"
        if not lbl.exists():
            continue
        shutil.copy2(img, DST / "images" / yolo_split / img.name)
        shutil.copy2(lbl, DST / "labels" / yolo_split / f"{stem}.txt")


def main():
    from PIL import Image

    copy_phone("train", "train")
    copy_phone("val", "val")

    model = YOLO(V2)
    count = {"with": 0, "without": 0}
    for img in sorted(TEST_IMAGES.glob("*.jpg")):
        name = img.name.lower()
        if name.startswith("with_bill"):
            im = Image.open(img)
            res = model.predict(img, conf=CONF, iou=0.5, imgsz=IMG_SIZE, verbose=False)[0]
            out_name = DST / "images" / "train" / f"cctv_{img.name}"
            shutil.copy2(img, out_name)
            lines = []
            for box in res.boxes:
                if int(box.cls) != 0:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                W, H = im.size
                cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                bw, bh = (x2 - x1) / W, (y2 - y1) / H
                lines.append(f"0 {cx:.6f} {cy:.6f} {max(0.0, min(1.0, bw)):.6f} {max(0.0, min(1.0, bh)):.6f}")
            (DST / "labels" / "train" / f"{out_name.name}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
            print(f"{img.name}: {len(lines)} pseudo boxes")
            count["with"] += 1
        elif name.startswith("without_bill"):
            out_name = DST / "images" / "train" / f"cctv_{img.name}"
            shutil.copy2(img, out_name)
            (DST / "labels" / "train" / f"{out_name.name}.txt").write_text("")
            count["without"] += 1

    n_train = len(list((DST / "images" / "train").glob("*.*")))
    n_val = len(list((DST / "images" / "val").glob("*.*")))
    n_boxes = sum(
        1
        for f in (DST / "labels" / "train").glob("*.txt")
        for line in f.read_text().splitlines()
        if line.strip()
    )
    print(f"train images: {n_train}, val images: {n_val}, total boxes: {n_boxes}")
    print(f"cctv with_bill: {count['with']}, without_bill: {count['without']}")

    (DST / "data.yaml").write_text(
        f"path: {DST}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        "  0: hand_bill\n"
    )
    print("data.yaml written to", DST / "data.yaml")


if __name__ == "__main__":
    main()

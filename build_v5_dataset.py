"""Build bill_hand_dataset_v5: the v4 dataset plus the phone-photo styles the
model misses (001.jpg, Media.jpeg, test_bill3.png, test_bill5.png), so a
fine-tune teaches the model those styles.

The phone photos get class-0 (hand_bill) pseudo-labels from v1 (the high-recall
model that does fire on them): every candidate box that passes the bill filter
(shape + hand + white) is kept, tightened, and normalized to YOLO format.
User confirmed these boxes look right (test_images/previews/label_confirm_*.png).
"""
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "bill_hand_dataset_v4"
DST = ROOT / "bill_hand_dataset_v5"
TEST = ROOT / "test_images"
V1 = ROOT / "models" / "hand_bill_detector.pt"
CONF = 0.3
MAX_BOXES = 3  # the number of candidates shown in the confirmation previews

sys_path = None
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "examples" / "YOLO-Hand-Bill-Detection"))
from main import bill_ok, tighten_box  # noqa: E402

import argparse  # noqa: E402

PHONE = ["001.jpg", "Media.jpeg", "test_bill3.png", "test_bill5.png"]
# prefix avoids colliding with the base dataset's own phone_*.jpg images
PHONE_PREFIX = "phone_test_"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("photos", nargs="*", default=None,
                        help="subset of phone photos to add (default: all 4)")
    parser.add_argument("--out", default="bill_hand_dataset_v5", help="output dataset dir")
    args = parser.parse_args()
    DST = ROOT / args.out
    phones = args.photos or PHONE
    if DST.exists():
        import shutil as _s
        _s.rmtree(DST)
    # 1) copy the v4 dataset wholesale (keeps its 2-class structure)
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            (DST / sub / split).mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        for img in (SRC / "images" / split).glob("*.*"):
            shutil.copy2(img, DST / "images" / split / img.name)
        for lbl in (SRC / "labels" / split).glob("*.txt"):
            shutil.copy2(lbl, DST / "labels" / split / lbl.name)
    # drop stale ultralytics label caches so they rebuild for the new files
    for cache in (DST / "labels" / "train" / "train.cache",
                  DST / "labels" / "val" / "val.cache"):
        if cache.exists():
            cache.unlink()

    # 2) add the phone photos with v1 pseudo-labels (class 0 = hand_bill)
    model = YOLO(V1)
    for name in phones:
        img_path = TEST / name
        im = __import__("cv2").imread(str(img_path))
        H, W = im.shape[:2]
        res = model.predict(img_path, imgsz=640, conf=CONF, verbose=False)[0]
        cands = []
        if res.boxes is not None and len(res.boxes):
            for b, c, cl in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                                res.boxes.cls.cpu().numpy()):
                box = [float(v) for v in b]
                if int(cl) != 0:
                    continue
                if not bill_ok(im, box, True, True, True):
                    continue
                cands.append((float(c), tighten_box(box, 0.25, float(W * H), (H, W))))
        cands.sort(reverse=True)
        lines = []
        for _, tb in cands[:MAX_BOXES]:
            x1, y1, x2, y2 = tb
            cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            bw, bh = (x2 - x1) / W, (y2 - y1) / H
            lines.append(f"0 {cx:.6f} {cy:.6f} {max(0.0, min(1.0, bw)):.6f} {max(0.0, min(1.0, bh)):.6f}")
        out_img = DST / "images" / "train" / f"{PHONE_PREFIX}{name}"
        if out_img.exists():
            raise SystemExit(f"collision: {out_img} already exists — refusing to overwrite")
        shutil.copy2(img_path, out_img)
        # ultralytics matches labels by image *stem*: x.jpg -> x.txt
        (DST / "labels" / "train" / f"{out_img.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        print(f"{name}: {len(lines)} label box(es) -> {out_img.name}")

    # 3) data.yaml (same 2 classes as v4)
    n_train = len(list((DST / "images" / "train").glob("*.*")))
    n_val = len(list((DST / "images" / "val").glob("*.*")))
    n_boxes = sum(
        1 for f in (DST / "labels" / "train").glob("*.txt")
        for line in f.read_text().splitlines() if line.strip()
    )
    (DST / "data.yaml").write_text(
        f"path: {args.out}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        "  0: hand_bill\n"
        "  1: no_bill\n"
    )
    print(f"{args.out}: {n_train} train imgs, {n_val} val imgs, {n_boxes} boxes -> {DST / 'data.yaml'}")


if __name__ == "__main__":
    main()

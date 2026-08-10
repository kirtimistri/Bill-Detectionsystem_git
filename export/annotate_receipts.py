"""Receipt/paper-slip candidate detector for the hand-bill dataset.

The previous auto-annotator used a banknote (currency) detector, which is the
wrong object for receipts. This script finds WHITE PAPER covered in TEXT:
receipts are typically the brightest, most edge-dense region of a photo (vs
hands, clothing, walls). It is a *candidate* generator - the output gallery
exists for human review.

Input : export/work_yolo/yolo/  (extracted hand_bill_colab.zip)
Output: export/receipt_review/
          labels/   candidate YOLO label files (class 0 = receipt)
          previews/ images with green boxes + status
          report.json
          index.html review gallery
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

DATA = Path(__file__).resolve().parent / "work_yolo" / "yolo"
OUT = Path(__file__).resolve().parent / "receipt_review"

IMG_DIRS = [DATA / "images" / "train", DATA / "images" / "val"]
LAB_DIRS = [DATA / "labels" / "train", DATA / "labels" / "val"]
MAX_BOXES = 2
MIN_AREA_FRAC = 0.02
MAX_AREA_FRAC = 0.97
BRIGHT_POS = 0.55          # brightness threshold position between 50th and 98th percentile
TEXT_THR = 0.15 * 255      # texture (edge density) threshold


def detect_candidates(bgr: np.ndarray) -> list[dict]:
    """Return ranked candidate boxes for one image."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Brightness: receipts are usually the brightest large region.
    lo, hi = np.percentile(blur, 50), np.percentile(blur, 98)
    thr = lo + BRIGHT_POS * (hi - lo)
    bright = (blur > thr).astype(np.uint8)

    # Texture: receipts are covered in small text -> dense edges.
    edges = cv2.Canny(blur, 60, 160)
    text = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    texture = cv2.GaussianBlur(text.astype(np.float32), (15, 15), 0)

    # Paper = bright AND textured.
    paper = ((bright == 1) & (texture > TEXT_THR)).astype(np.uint8)
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(paper, 8)
    img_area = h * w
    cands = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < MIN_AREA_FRAC * img_area or area > MAX_AREA_FRAC * img_area:
            continue
        mask = (labels == i).astype(np.uint8)
        inside = cv2.mean(gray, mask)[0]
        big = cv2.dilate(mask, np.ones((21, 21), np.uint8))
        ring = big - mask
        outside = cv2.mean(gray, ring)[0] if ring.sum() > 0 else inside
        contrast = inside - outside
        t_in = cv2.mean(texture, mask)[0]
        # rough solidity via largest contour of the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solidity = 0.0
        if contours:
            c = max(contours, key=cv2.contourArea)
            solidity = cv2.contourArea(c) / max(1.0, cv2.arcLength(c, True))
        score = np.log1p(area) * max(0.0, contrast) * t_in * max(0.2, min(1.0, solidity))
        cands.append({
            "x": int(x), "y": int(y), "w": int(bw), "h": int(bh),
            "area": int(area), "contrast": float(contrast),
            "texture": float(t_in), "solidity": float(solidity), "score": float(score),
        })
    cands.sort(key=lambda c: c["score"], reverse=True)
    return cands[:MAX_BOXES]


def to_yolo(c: dict, w: int, h: int) -> tuple[float, float, float, float]:
    cx = (c["x"] + c["w"] / 2) / w
    cy = (c["y"] + c["h"] / 2) / h
    bw = c["w"] / w
    bh = c["h"] / h
    cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
    bw, bh = max(0.0, min(1.0, bw)), max(0.0, min(1.0, bh))
    return cx, cy, bw, bh


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "labels").mkdir(parents=True)
    (OUT / "previews").mkdir(parents=True)

    report = {}
    cards = []
    total = ok = low = none = 0

    for img_dir, lab_dir in zip(IMG_DIRS, LAB_DIRS):
        for img_path in sorted(img_dir.glob("*.jpg")):
            stem = img_path.stem
            total += 1
            bgr = cv2.imread(str(img_path))
            h, w = bgr.shape[:2]
            cands = detect_candidates(bgr)

            # threshold on a normalized score
            best = cands[0] if cands else None
            norm = (best["score"] / (np.log1p(0.02 * h * w))) if best else 0.0

            lines, status, score = [], "NONE", 0.0
            if best and norm >= 90:
                status = "DETECTED"
            elif best and norm >= 35:
                status = "LOW_CONF"
            elif best:
                status = "LOW_CONF"
            if best:
                score = round(norm, 1)
                if norm >= 35:
                    for c in cands:
                        cx, cy, bw, bh = to_yolo(c, w, h)
                        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            if status == "DETECTED":
                ok += 1
            elif status == "LOW_CONF":
                low += 1
            else:
                none += 1

            # write label
            lab_path = OUT / "labels" / f"{stem}.txt"
            if lines:
                lab_path.write_text("".join(lines), encoding="utf-8")
            else:
                lab_path.write_text("", encoding="utf-8")

            # annotated preview
            frame = bgr.copy()
            for c in cands:
                cv2.rectangle(frame, (c["x"], c["y"]), (c["x"] + c["w"], c["y"] + c["h"]),
                              (0, 200, 0) if norm >= 35 else (0, 120, 255), 2)
            preview = OUT / "previews" / f"{stem}.jpg"
            cv2.imwrite(str(preview), frame)

            color = "#4caf50" if status == "DETECTED" else ("#ff9800" if status == "LOW_CONF" else "#e53935")
            cards.append(
                f'<div class="card {status.lower()}"><img src="previews/{stem}.jpg">'
                f'<div class="meta"><b>{stem}</b>'
                f'<span style="color:{color}">{status} · score {score}</span></div></div>'
            )
            report[stem] = {
                "split": img_dir.parent.name, "status": status, "score": score,
                "boxes": len(lines),
                "cands": cands,
            }

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Receipt auto-fix review</title><style>
body {{ font-family: system-ui, sans-serif; background: #111; color: #eee; margin: 20px; }}
h1 {{ font-size: 18px; }} .stats {{ color: #9db; margin-bottom: 14px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 12px; }}
.card {{ background: #1c1c1c; border-radius: 8px; overflow: hidden; border: 1px solid #333; }}
.card img {{ width: 100%; display: block; }}
.card.none img {{ opacity: .5; }}
.meta {{ padding: 8px 10px; font-size: 11px; display: flex; justify-content: space-between; gap: 6px; }}
.meta b {{ font-size: 11px; max-width: 55%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
</style></head><body>
<h1>Receipt candidate auto-fix — review before retraining</h1>
<div class="stats">{total} images · <span style="color:#4caf50">{ok} DETECTED</span> ·
<span style="color:#ff9800">{low} LOW_CONF</span> · <span style="color:#e53935">{none} NO_DETECTION</span></div>
<p style="color:#aaa; font-size:12px">Green = receipt candidate box. Check the box actually covers the paper receipt.
Drop LOW_CONF / NO_DETECTION images that don't contain a receipt.</p>
<div class="grid">{''.join(cards)}</div></body></html>"""
    (OUT / "index.html").write_text(html, encoding="utf-8")
    (OUT / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"total={total}  DETECTED={ok}  LOW_CONF={low}  NO_DETECTION={none}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

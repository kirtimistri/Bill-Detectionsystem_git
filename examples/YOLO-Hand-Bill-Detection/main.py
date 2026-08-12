# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Single-stage hand -> bill (currency) detection for CCTV.

Detects a hand holding a bill directly with one YOLO model trained on the
``hand_bill`` class. No money-denomination or two-stage pipeline logic.

Usage:
    # CCTV video file
    python main.py --weights ../../../models/hand_bill_detector.pt --source path/to/video.mp4 --save

    # Webcam
    python main.py --weights local_runs/detect/bill_hand/weights/best.pt --source 0

    # Image / directory
    python main.py --weights local_runs/detect/bill_hand/weights/best.pt --source path/to/images/ --save
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".ts", ".mpeg", ".mpg"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
HIGH_CONF = 0.6


def box_color(conf: float) -> tuple[int, int, int]:
    """Return BGR color for a detection: green for high confidence, orange for low."""
    return (0, 200, 0) if conf >= HIGH_CONF else (0, 165, 255)


def label_txt_color(conf: float) -> tuple[int, int, int]:
    """Black text on orange (low-confidence) labels for readability, white on green."""
    return (0, 0, 0) if conf < HIGH_CONF else (255, 255, 255)


def bill_shape_ok(box, frame_shape, min_ratio: float = 1.5, max_ratio: float = 6.0) -> bool:
    """Keep detections shaped like a bill: elongated (short in width, long in
    length), aspect ratio ~1.5–6. Near-square patches and extreme slivers are
    typically false alarms. A box cut off by the image edge has an unreliable
    ratio (the bill may just be entering the frame), so it is always kept;
    a box spanning nearly the whole frame is always rejected.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return False
    if frame_shape is not None:
        W, H = float(frame_shape[1]), float(frame_shape[0])
        if W > 0 and H > 0:
            if min(w, h) / max(W, H) < 0.04:
                return False  # tiny sliver (even at the edge) — a false alarm
            if (w * h) / (W * H) >= 0.85:
                return False  # near full-frame "detection" — a false alarm
            if x1 <= 2 or x2 >= W - 2 or y1 <= 2 or y2 >= H - 2:
                return True  # substantial box cut off by the edge — keep it
    ratio = max(w, h) / min(w, h)
    return min_ratio <= ratio <= max_ratio


def bill_ok(frame, box, shape_filter: bool = True, hand_check: bool = True, white_check: bool = True) -> bool:
    """Combined "hand holding bill" verdict used by every pipeline.

    A box counts when it is clearly bill-shaped (elongated), OR — because a hand
    and note together fill a near-square region — when it is near-square but a
    hand is physically present. Hard junk is always rejected with the filter on:
    tiny slivers, near-full-frame boxes, and extreme slivers. Substantial
    edge-cut boxes (bill entering the frame) are always kept. Each check
    (shape / hand / white paper) can be switched off independently.
    """
    if not shape_filter and not hand_check and not white_check:
        return True
    if shape_filter:
        x1, y1, x2, y2 = (float(v) for v in box)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return False
        W, H = float(frame.shape[1]), float(frame.shape[0])
        if W > 0 and H > 0:
            if min(w, h) / max(W, H) < 0.04:
                return False  # tiny sliver
            if (w * h) / (W * H) >= 0.85:
                return False  # near full-frame
            if x1 <= 2 or x2 >= W - 2 or y1 <= 2 or y2 >= H - 2:
                return True  # substantial edge-cut box — can't judge, keep
            ratio = max(w, h) / min(w, h)
            if 1.5 <= ratio <= 6.0:
                pass  # clearly bill-shaped
            elif ratio > 6.0:
                return False  # extreme sliver
            elif not hand_check or not hand_near_box(frame, box):
                return False  # near-square only counts with a hand present
    elif hand_check and not hand_near_box(frame, box):
        return False
    if white_check and not bill_whiteness_ok(frame, box):
        return False
    return True


def hand_near_box(frame, box, margin: float = 0.6, skin_frac: float = 0.02) -> bool:
    """True when skin-colored pixels (a hand) appear inside or near the bill box.

    The class is "hand holding bill", so a bill-shaped box with NO hand close by
    is probably a false alarm (a receipt, a sign, paper). Skin is detected in
    YCrCb space (robust to lighting), within the box expanded by ``margin`` x
    its size — a hand overlapping or holding the bill always lands there.
    Returns True (lenient) if there is no usable color info in the region.
    """
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    m = max(int(max(w, h) * margin), 15)
    rx1, ry1 = max(0, x1 - m), max(0, y1 - m)
    rx2, ry2 = min(W, x2 + m), min(H, y2 + m)
    if rx2 - rx1 < 8 or ry2 - ry1 < 8:
        return True  # region clipped away — can't judge, stay lenient
    region = frame[ry1:ry2, rx1:rx2]
    try:
        ycrcb = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
    except cv2.error:
        return True
    Y, Cr, Cb = ycrcb[:, :, 0].astype(int), ycrcb[:, :, 1].astype(int), ycrcb[:, :, 2].astype(int)
    if float(Y.mean()) < 40:
        return True  # too dark to judge skin (night/IR CCTV) — don't drop detections
    skin = (Y > 60) & (Cr >= 133) & (Cr <= 175) & (Cb >= 77) & (Cb <= 127)
    frac = float(skin.mean())
    return frac >= skin_frac


def bill_whiteness_ok(frame, box, max_sat: float = 0.45, min_bright: float = 0.28, patch: float = 0.5) -> bool:
    """True when the box looks like a bill: white paper with black text.

    Bills are white/light with dark print, so the region is mostly desaturated
    and bright. Strongly colored or near-black boxes are usually false alarms
    (objects, clothing, signage). Only the central ``patch`` fraction of the box
    is sampled to avoid the hand/edges dominating. Returns True (lenient) when
    the box has no usable color info (very dark IR CCTV) so detections there
    are not dropped.
    """
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    pw, ph = max(1, w * patch / 2), max(1, h * patch / 2)
    rx1, ry1 = max(0, int(cx - pw)), max(0, int(cy - ph))
    rx2, ry2 = min(W, int(cx + pw)), min(H, int(cy + ph))
    if rx2 - rx1 < 4 or ry2 - ry1 < 4:
        return True  # clipped away - can't judge, stay lenient
    region = frame[ry1:ry2, rx1:rx2]
    try:
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    except cv2.error:
        return True
    sat = hsv[:, :, 1].astype(int)
    val = hsv[:, :, 2].astype(int)
    if float(val.mean()) < 12:
        return True  # too dark to judge (night/IR CCTV) - don't drop detections
    return float(sat.mean()) / 255.0 <= max_sat and float(val.mean()) / 255.0 >= min_bright


def box_iou(a, b) -> float:
    """Intersection-over-union of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms_boxes(boxes, iou_thr: float = 0.5):
    """Greedy non-max suppression; keeps higher-confidence detections first.

    Accepts (box, conf) or (box, conf, cls) tuples — extra fields ride along
    with their box through the suppression.
    """
    kept = []
    for item in sorted(boxes, key=lambda it: -it[1]):
        b = item[0]
        if all(box_iou(b, k[0]) < iou_thr for k in kept):
            kept.append(item)
    return kept


def predict_merged(model, frame, conf: float, device: str = "") -> list:
    """Run ``model`` at 416 and 640 and NMS-merge the hand_bill (class-0) boxes.

    Multi-scale inference is measurably more accurate than either single scale
    (benchmark: 8/8 with-bill AND the phone photos single scales miss, 0/10
    false alarms) at ~2x the cost — so it is opt-in and streams stay
    single-scale for speed. Returns [(box, conf, cls), ...] with cls always 0.
    """
    merged = []
    for size in (416, 640):
        r = model.predict(frame, imgsz=size, conf=conf, device=device, verbose=False)[0]
        if r.boxes is None or not len(r.boxes):
            continue
        for b, c, cl in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                            r.boxes.cls.cpu().numpy()):
            if int(cl) != 0:
                continue  # only the hand_bill class (v4 also predicts a no_bill class)
            merged.append(([float(v) for v in b], float(c), int(cl)))
    return nms_boxes(merged)


def tighten_box(box, inset: float, frame_area: float, frame_shape=None) -> tuple[float, float, float, float]:
    """Shrink a detection box toward its center so it hugs the bill.

    Bigger (looser) boxes shrink proportionally more, keeping detections focused
    instead of spanning the whole frame. ``inset`` is the base fraction shrunk
    from each side; ``frame_area`` scales it up for full-frame detections.

    ``frame_shape`` (h, w) makes the shrink edge-aware: a side that already
    touches the image boundary is NOT pulled inward, so a bill at the edge of
    the frame keeps its box (previously the box drifted away from the bill).
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    if frame_area <= 0 or w <= 0 or h <= 0:
        return x1, y1, x2, y2
    area_frac = (w * h) / frame_area
    # sloppy full-frame boxes get extra tightening, but never shrink more than
    # 35% per side so a high slider value can't collapse a box into a sliver
    k = min(inset * (1.0 + area_frac), 0.35)
    dx, dy = w * k, h * k
    if frame_shape is not None:
        # room to shrink on each side before hitting the image edge; zero for
        # sides already at the boundary so those stay put
        W, H = float(frame_shape[1]), float(frame_shape[0])
        sx1 = x1 if x1 > 2 else 0.0
        sx2 = (W - x2) if x2 < W - 2 else 0.0
        sy1 = y1 if y1 > 2 else 0.0
        sy2 = (H - y2) if y2 < H - 2 else 0.0
    else:  # backward-compatible symmetric behavior
        sx1 = sx2 = dx
        sy1 = sy2 = dy
    nx1, ny1 = x1 + min(dx, sx1), y1 + min(dy, sy1)
    nx2, ny2 = x2 - min(dx, sx2), y2 - min(dy, sy2)
    if nx2 - nx1 < 8 or ny2 - ny1 < 8:  # keep a minimum visible box size
        return x1, y1, x2, y2
    return nx1, ny1, nx2, ny2


class HandBillDetector:
    """Detect hands holding bills directly with a single YOLO model.

    Attributes:
        model (YOLO): YOLO model trained on the ``hand_bill`` class.
        device (str): Device used for inference, e.g. "", "cpu", "0", "mps".
        conf (float): Confidence threshold for detections.
        imgsz (int): Inference image size used by the model.
    """

    def __init__(self, weights: str, device: str = "", conf: float = 0.25, imgsz: int = 640, box_inset: float = 0.15, shape_filter: bool = True, hand_check: bool = True, white_check: bool = True, multi_scale: bool = False) -> None:
        """Initialize the hand_bill detection model and configuration."""
        self.model = YOLO(weights)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.box_inset = box_inset
        self.shape_filter = shape_filter
        self.hand_check = hand_check
        self.white_check = white_check
        self.multi_scale = multi_scale
        # Multi-scale runs its own 416+640 inference per frame WHILE the outer
        # stream generator is open — ultralytics hangs if the same instance is
        # called re-entrantly, so the merged passes use a separate instance.
        self._ms_model = YOLO(weights) if multi_scale else None

    def _merged_boxes(self, frame) -> list:
        """Run the model at 416 and 640 and NMS-merge the class-0 detections.

        Multi-scale inference is measurably more accurate (8/8 with-bill AND
        phone-photo detections, vs either single scale) at ~2x the cost.
        Uses the dedicated ``_ms_model`` instance so this never re-enters the
        outer stream predictor (which would hang ultralytics).
        """
        return predict_merged(self._ms_model or self.model, frame, self.conf, self.device)


    @staticmethod
    def is_video_source(source: str | int) -> bool:
        """Return True if ``source`` is a video file, webcam index, or stream/URL (not a still image)."""
        s = str(source)
        suffix = Path(s).suffix.lower()
        if s.isdigit():
            return True
        if suffix in VIDEO_SUFFIXES:
            return True
        if suffix in IMAGE_SUFFIXES:
            return False
        return s.lower().startswith(("http://", "https://", "rtsp://", "rtmp://", "udp://"))

    def predict(
        self,
        source: str | int,
        save: bool = False,
        show: bool = False,
        output_dir: str = "runs/cctv",
    ) -> None:
        """Run hand_bill detection over ``source`` and optionally display/save the annotated output.

        Args:
            source (str | int): Image path, directory, video path, URL or webcam index (e.g. 0).
            save (bool): Save annotated outputs (images or a video depending on the source type).
            show (bool): Display annotated frames in a window (press 'q' to quit).
            output_dir (str): Directory (images) or base path (video) for saved outputs.
        """
        save_dir = Path(output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        video_writer = None
        video_path = None
        if save and self.is_video_source(source):
            cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            video_path = save_dir / f"{Path(str(source)).stem or 'output'}.mp4"
            if frame_w and frame_h:
                video_writer = cv2.VideoWriter(
                    str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h)
                )
                print(f"Saving annotated video to {video_path}")

        total_frames = 0
        total_hand_bills = 0
        frames_with_bill = 0

        # multi-scale does its own 416+640 inference per frame, so the outer
        # stream pass is only used as a fast (320px) frame loader
        outer_imgsz = 320 if self.multi_scale else self.imgsz
        for result in self.model.predict(
            source,
            stream=True,
            imgsz=outer_imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        ):
            total_frames += 1
            frame = result.orig_img
            if self.multi_scale:
                boxes = self._merged_boxes(frame)
            else:
                boxes = []
                if result.boxes is not None and len(result.boxes):
                    boxes = [(b, float(c), int(cl)) for b, c, cl in zip(
                        result.boxes.xyxy.cpu().numpy(),
                        result.boxes.conf.cpu().numpy(),
                        result.boxes.cls.cpu().numpy())]

            annotator = Annotator(frame, line_width=3, pil=False)
            frame_area = frame.shape[0] * frame.shape[1]
            frame_hits = 0
            kept_conf = []
            for box, conf, cls in boxes:
                if int(cls) != 0:
                    continue  # only the hand_bill class (v4 also predicts a no_bill class)
                if not bill_ok(frame, box, self.shape_filter, self.hand_check, self.white_check):
                    continue  # not a hand holding a bill -> false alarm
                kept_conf.append(float(conf))
                label = f"{self.model.names.get(int(cls), 'hand_bill')} {float(conf):.2f}"
                annotator.box_label(
                    tighten_box(box, self.box_inset, frame_area, frame.shape),
                    label,
                    color=box_color(float(conf)),
                    txt_color=label_txt_color(float(conf)),
                )
                frame_hits += 1
                total_hand_bills += 1

            if frame_hits:
                frames_with_bill += 1
                # banner across the top, colored by the frame's best confidence
                max_conf = max(kept_conf)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), box_color(max_conf), -1)
                cv2.putText(
                    frame,
                    f"BILL DETECTED x{frame_hits}",
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

            if video_writer is None and video_path is not None:  # capture opened but metadata was unavailable
                h, w = frame.shape[:2]
                video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                print(f"Saving annotated video to {video_path}")

            if video_writer is not None:
                video_writer.write(frame)
            elif save:
                out_path = save_dir / f"{Path(result.path).stem}.jpg"
                cv2.imwrite(str(out_path), frame)

            if show:
                cv2.imshow("CCTV Hand-Bill Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if video_writer is not None:
            video_writer.release()
        if show:
            cv2.destroyAllWindows()
        print(
            f"Processed {total_frames} frame(s): hand holding bill detected in "
            f"{frames_with_bill} frame(s), {total_hand_bills} total detection(s)."
        )


def parse_opt() -> argparse.Namespace:
    """Parse command-line arguments for single-stage hand_bill detection."""
    parser = argparse.ArgumentParser(description="CCTV hand -> bill detection with Ultralytics YOLO")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="YOLO model path trained on the hand_bill class (e.g. models/hand_bill_detector.pt)",
    )
    parser.add_argument("--source", type=str, default="0", help="image path, directory, video path or webcam index")
    parser.add_argument("--device", default="", help='cuda device, i.e. 0 or 0,1,2,3 or cpu/mps, "" for auto-detection')
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold for detections")
    parser.add_argument("--imgsz", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--multi-scale", dest="multi_scale", action="store_true", help="run inference at 416 AND 640 and merge the results — most accurate, ~2-3x slower (default off; streams stay single-scale)")
    parser.add_argument("--box-inset", type=float, default=0.15, help="base fraction to shrink each box toward its center; larger boxes shrink more (0 = off, e.g. 0.2 = tighter/smaller)")
    parser.add_argument("--shape-filter", dest="shape_filter", action="store_true", default=True, help="keep only bill-shaped (elongated) detections; kills most false alarms (default)")
    parser.add_argument("--no-shape-filter", dest="shape_filter", action="store_false", help="disable the bill-shape (elongation) filter")
    parser.add_argument("--hand-check", dest="hand_check", action="store_true", default=True, help="require skin (a hand) near the bill box (default)")
    parser.add_argument("--no-hand-check", dest="hand_check", action="store_false", help="disable the hand-nearby (skin) check")
    parser.add_argument("--white-check", dest="white_check", action="store_true", default=True, help="require white (desaturated, bright) bill paper inside the box (default)")
    parser.add_argument("--no-white-check", dest="white_check", action="store_false", help="disable the whiteness check")
    parser.add_argument("--save", action="store_true", help="save annotated outputs")
    parser.add_argument("--show", action="store_true", help="display annotated frames in a window")
    parser.add_argument("--output-dir", type=str, default="runs/cctv", help="directory for saved outputs")
    return parser.parse_args()


def main(opt: argparse.Namespace) -> None:
    """Run single-stage hand_bill detection with parsed arguments."""
    detector = HandBillDetector(
        weights=opt.weights,
        device=opt.device,
        conf=opt.conf,
        imgsz=opt.imgsz,
        box_inset=opt.box_inset,
        shape_filter=opt.shape_filter,
        hand_check=opt.hand_check,
        white_check=opt.white_check,
        multi_scale=opt.multi_scale,
    )
    detector.predict(
        source=opt.source,
        save=opt.save,
        show=opt.show,
        output_dir=opt.output_dir,
    )


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Two-stage hand -> bill (currency) detection.

Pipeline:
    Stage 1: Detect hands in the full frame with a hand-detection YOLO model.
    Stage 2: Crop each detected hand region (with an optional margin) and run a
        second YOLO model that detects bills/currency *inside the crop only*.

The bill boxes are mapped back to original frame coordinates and drawn on the
frame together with the hand boxes. Images, directories, videos and webcam are
all supported through the Ultralytics source argument.

Usage:
    # Webcam
    python main.py --hand-weights hand.pt --bill-weights bill.pt --source 0

    # Image / directory / video
    python main.py --hand-weights hand.pt --bill-weights bill.pt --source path/to/bus.jpg
    python main.py --hand-weights hand.pt --bill-weights bill.pt --source path/to/images/
    python main.py --hand-weights hand.pt --bill-weights bill.pt --source path/to/video.mp4 --save
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".ts", ".mpeg", ".mpg"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}


class TwoStageBillDetector:
    """Two-stage detector: detect hands, crop the hand region, then detect bills inside the crop.

    Attributes:
        hand_model (YOLO): Stage-1 YOLO model that detects hands.
        bill_model (YOLO): Stage-2 YOLO model that detects bills/currency.
        device (str): Device used for inference, e.g. "", "cpu", "0", "mps".
        hand_conf (float): Confidence threshold for stage-1 (hand) detections.
        bill_conf (float): Confidence threshold for stage-2 (bill) detections.
        margin (float): Fraction of the hand box size added around the crop so the bill is fully captured.
        imgsz (int): Inference image size used by both models.
    """

    def __init__(
        self,
        hand_weights: str,
        bill_weights: str,
        device: str = "",
        hand_conf: float = 0.25,
        bill_conf: float = 0.25,
        margin: float = 0.1,
        imgsz: int = 640,
    ) -> None:
        """Initialize both YOLO models and the detection configuration."""
        self.hand_model = YOLO(hand_weights)
        self.bill_model = YOLO(bill_weights)
        self.device = device
        self.hand_conf = hand_conf
        self.bill_conf = bill_conf
        self.margin = margin
        self.imgsz = imgsz

    def crop_hand_region(self, frame: np.ndarray, box: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int]]:
        """Expand a hand box by ``margin``, clip it to the frame, and return the crop plus its top-left offset.

        Args:
            frame (np.ndarray): Full BGR frame.
            box (np.ndarray): Hand box as [x1, y1, x2, y2] in frame coordinates.

        Returns:
            (tuple[np.ndarray | None, tuple[int, int]]): The cropped hand region and the (x, y) offset of the crop
                within the frame, or (None, (0, 0)) if the box collapses after clipping.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box.astype(float)
        bw, bh = x2 - x1, y2 - y1
        mx, my = bw * self.margin, bh * self.margin  # margin in pixels per axis

        x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
        x2, y2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
        if x2 <= x1 or y2 <= y1:
            return None, (0, 0)
        return frame[y1:y2, x1:x2], (x1, y1)

    def detect_bills_in_hands(self, frame: np.ndarray, hand_boxes: np.ndarray) -> list[list[np.ndarray]]:
        """Run stage-2 bill detection on every hand crop, returning bill boxes in frame coordinates.

        Args:
            frame (np.ndarray): Full BGR frame.
            hand_boxes (np.ndarray): Array of hand boxes as [x1, y1, x2, y2], shape (N, 4).

        Returns:
            (list[list[np.ndarray]]): For each hand, a list of bill boxes [x1, y1, x2, y2] mapped back to frame
                coordinates. Bill confidence and class ids are appended as the 5th and 6th values.
        """
        crops, offsets, hand_indices = [], [], []
        for hand_idx, box in enumerate(hand_boxes):
            crop, offset = self.crop_hand_region(frame, box)
            if crop is not None:
                crops.append(crop)
                offsets.append(offset)
                hand_indices.append(hand_idx)

        bills_per_hand: list[list[np.ndarray]] = [[] for _ in hand_boxes]
        if not crops:
            return bills_per_hand

        # Batch all hand crops through the bill model in a single inference call.
        bill_results = self.bill_model.predict(
            crops, imgsz=self.imgsz, conf=self.bill_conf, device=self.device, verbose=False
        )
        for bill_result, hand_idx, (x1, y1) in zip(bill_results, hand_indices, offsets):
            if bill_result.boxes is None or not len(bill_result.boxes):
                continue
            for bill_box, bill_conf, bill_cls in zip(
                bill_result.boxes.xyxy.cpu().numpy(),
                bill_result.boxes.conf.cpu().numpy(),
                bill_result.boxes.cls.cpu().numpy(),
            ):
                # Map the bill box from crop coordinates back to frame coordinates.
                bills_per_hand[hand_idx].append(
                    np.array([*bill_box + (x1, y1, x1, y1), bill_conf, bill_cls], dtype=float)
                )

        return bills_per_hand

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
        save_crops: bool = False,
        show: bool = False,
        output_dir: str = "runs/two_stage",
        hand_classes: list[int] | None = None,
    ) -> None:
        """Run the two-stage pipeline over ``source`` and optionally display/save the annotated output.

        Args:
            source (str | int): Image path, directory, video path, URL or webcam index (e.g. 0).
            save (bool): Save annotated outputs (images or a video depending on the source type).
            save_crops (bool): Also save the hand and bill crops to ``output_dir/crops``.
            show (bool): Display annotated frames in a window (press 'q' to quit).
            output_dir (str): Directory (images) or base path (video) for saved outputs.
            hand_classes (list[int] | None): Optional filter of stage-1 class ids treated as hands.
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

        hand_colors = [(255, 0, 0), (255, 165, 0)]  # hand box + expanded crop box
        bill_color = (0, 200, 0)
        total_hands = 0
        total_bills = 0
        total_frames = 0

        for result in self.hand_model.predict(
            source,
            stream=True,
            imgsz=self.imgsz,
            conf=self.hand_conf,
            device=self.device,
            classes=hand_classes,
            verbose=False,
        ):
            total_frames += 1
            frame = result.orig_img

            annotator = Annotator(frame, line_width=2, font_size=10, pil=False)
            hand_boxes = (
                result.boxes.xyxy.cpu().numpy() if result.boxes is not None and len(result.boxes) else np.empty((0, 4))
            )

            for i, box in enumerate(hand_boxes):
                hand_name = result.names[int(result.boxes.cls[i])]
                hand_conf = float(result.boxes.conf[i])
                annotator.box_label(box, f"{hand_name} {hand_conf:.2f}", color=hand_colors[0])
                total_hands += 1

                crop, (cx, cy) = self.crop_hand_region(frame, box)
                if crop is None:
                    continue
                # Draw the expanded crop region that stage 2 will inspect.
                annotator.box_label((cx, cy, cx + crop.shape[1], cy + crop.shape[0]), color=hand_colors[1])

                if save_crops:
                    crop_dir = save_dir / "crops" / "hand"
                    crop_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(crop_dir / f"frame{total_frames}_hand{i}.jpg"), crop)

            bills_per_hand = self.detect_bills_in_hands(frame, hand_boxes)
            for bills in bills_per_hand:
                for bill in bills:
                    box_xyxy = bill[:4]
                    conf = float(bill[4])
                    cls = int(bill[5]) if len(bill) > 5 else 0
                    label = f"{self.bill_model.names.get(cls, 'bill')} {conf:.2f}"
                    annotator.box_label(box_xyxy, label, color=bill_color)
                    total_bills += 1

                    if save_crops:
                        crop_dir = save_dir / "crops" / "bill"
                        crop_dir.mkdir(parents=True, exist_ok=True)
                        x1, y1, x2, y2 = map(int, box_xyxy)
                        cv2.imwrite(
                            str(crop_dir / f"frame{total_frames}_bill{total_bills}.jpg"),
                            frame[max(0, y1) : y2, max(0, x1) : x2],
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
                cv2.imshow("Two-Stage Hand -> Bill Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if video_writer is not None:
            video_writer.release()
        if show:
            cv2.destroyAllWindows()
        print(f"Processed {total_frames} frame(s): {total_hands} hand(s), {total_bills} bill(s) detected.")


def parse_opt() -> argparse.Namespace:
    """Parse command-line arguments for the two-stage hand -> bill detection pipeline."""
    parser = argparse.ArgumentParser(description="Two-stage hand -> bill detection with Ultralytics YOLO")
    parser.add_argument(
        "--hand-weights",
        type=str,
        required=True,
        help="stage-1 hand detection YOLO model path (e.g. a .pt trained on a hand dataset)",
    )
    parser.add_argument(
        "--bill-weights",
        type=str,
        required=True,
        help="stage-2 bill/currency detection YOLO model path (e.g. a .pt trained on a currency dataset)",
    )
    parser.add_argument(
        "--source", type=str, default="0", help="image path, directory, video path or webcam index (default: 0)"
    )
    parser.add_argument("--device", default="", help='cuda device, i.e. 0 or 0,1,2,3 or cpu/mps, "" for auto-detection')
    parser.add_argument("--hand-conf", type=float, default=0.25, help="confidence threshold for hand detection")
    parser.add_argument("--bill-conf", type=float, default=0.25, help="confidence threshold for bill detection")
    parser.add_argument(
        "--margin",
        type=float,
        default=0.1,
        help="fraction of the hand box size added around the crop so the bill is fully captured",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="inference size (pixels) for both models")
    parser.add_argument("--save", action="store_true", help="save annotated outputs")
    parser.add_argument("--save-crops", action="store_true", help="also save hand and bill crops to the output dir")
    parser.add_argument("--show", action="store_true", help="display annotated frames in a window")
    parser.add_argument("--output-dir", type=str, default="runs/two_stage", help="directory for saved outputs")
    parser.add_argument(
        "--hand-classes",
        nargs="+",
        type=int,
        default=None,
        help="optional stage-1 class ids treated as hands, e.g. --hand-classes 0 1",
    )
    return parser.parse_args()


def main(opt: argparse.Namespace) -> None:
    """Run the two-stage hand -> bill detection pipeline with parsed arguments."""
    detector = TwoStageBillDetector(
        hand_weights=opt.hand_weights,
        bill_weights=opt.bill_weights,
        device=opt.device,
        hand_conf=opt.hand_conf,
        bill_conf=opt.bill_conf,
        margin=opt.margin,
        imgsz=opt.imgsz,
    )
    detector.predict(
        source=opt.source,
        save=opt.save,
        save_crops=opt.save_crops,
        show=opt.show,
        output_dir=opt.output_dir,
        hand_classes=opt.hand_classes,
    )


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

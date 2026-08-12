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


class HandBillDetector:
    """Detect hands holding bills directly with a single YOLO model.

    Attributes:
        model (YOLO): YOLO model trained on the ``hand_bill`` class.
        device (str): Device used for inference, e.g. "", "cpu", "0", "mps".
        conf (float): Confidence threshold for detections.
        imgsz (int): Inference image size used by the model.
    """

    def __init__(self, weights: str, device: str = "", conf: float = 0.25, imgsz: int = 640) -> None:
        """Initialize the hand_bill detection model and configuration."""
        self.model = YOLO(weights)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

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

        for result in self.model.predict(
            source,
            stream=True,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        ):
            total_frames += 1
            frame = result.orig_img
            boxes = result.boxes if result.boxes is not None else None

            annotator = Annotator(frame, line_width=3, pil=False)
            frame_hits = 0
            if boxes is not None and len(boxes):
                for box, conf, cls in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()):
                    label = f"{self.model.names.get(int(cls), 'hand_bill')} {float(conf):.2f}"
                    annotator.box_label(box, label, color=box_color(float(conf)), txt_color=label_txt_color(float(conf)))
                    frame_hits += 1
                    total_hand_bills += 1

            if frame_hits:
                frames_with_bill += 1
                # banner across the top, colored by the frame's best confidence
                max_conf = max(float(c) for c in boxes.conf.cpu().numpy())
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

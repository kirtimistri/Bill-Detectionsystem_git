#!/usr/bin/env python3
"""
Real-time hand-bill detection for CCTV AND Android phone cameras (DroidCam / IP Webcam).

Works with any source OpenCV can open:
  - CCTV RTSP/HTTP stream   --url rtsp://user:pass@192.168.0.50:554/stream1
  - CCTV video file         --url path/to/clip.mp4
  - DroidCam (virtual cam)  --url /dev/video2   (after droidcam-cli is running)
  - DroidCam MJPEG stream   --url http://<phone-ip>:4747/video
  - IP Webcam MJPEG         --url http://<phone-ip>:8080/video
  - built-in webcam         --url 0

Keys: q = quit | s = save snapshot

DroidCam setup (phone + PC on same Wi-Fi):
  1. Install "DroidCam" app on the phone, start it.
  2. On PC:  sudo ./install-client && sudo ./install-video   (from the droidcam folder)
  3. Start virtual camera:  droidcam-cli <phone-ip> 4747 /dev/video2 &
  4. Run:  python live_detect.py --url /dev/video2 --weights models/hand_bill_detector_v2.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

HIGH_CONF = 0.6


def box_color(conf: float) -> tuple[int, int, int]:
    """Return BGR color for a detection: green for high confidence, orange for low."""
    return (0, 200, 0) if conf >= HIGH_CONF else (0, 165, 255)


def label_txt_color(conf: float) -> tuple[int, int, int]:
    """Black text on orange (low-confidence) labels for readability, white on green."""
    return (0, 0, 0) if conf < HIGH_CONF else (255, 255, 255)


class LiveDetector:
    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 640, device: str = "") -> None:
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def run(self, url: str, save: bool = False, out_dir: str = "runs/live") -> None:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open source: {url}\n"
                "- For CCTV: check the RTSP/HTTP URL and network.\n"
                "- For DroidCam: is droidcam-cli running? (droidcam-cli <phone-ip> 4747 /dev/video2)\n"
                "- For IP Webcam: did you tap 'Start server' in the app?"
            )
        print(f"Source: {url} ({cap.getBackendName()} backend)")

        save_dir = Path(out_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        frame_count = 0
        prev_time = time.time()
        fps = 0.0
        total_det = 0
        frames_hit = 0

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame dropped / stream ended, retrying...")
                time.sleep(0.3)
                continue

            frame_count += 1
            results = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
            boxes = results.boxes

            annotator = Annotator(frame, line_width=2, font_size=10, pil=False)
            frame_hits = 0
            if boxes is not None and len(boxes):
                for box, c, cls in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()):
                    label = f"{self.model.names.get(int(cls), 'obj')} {float(c):.2f}"
                    annotator.box_label(box, label, color=box_color(float(c)), txt_color=label_txt_color(float(c)))
                    frame_hits += 1
                total_det += frame_hits
                frames_hit += 1

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / (now - prev_time)) if now > prev_time else fps
            prev_time = now

            cv2.putText(
                frame,
                f"FPS {fps:.1f} | hits {frames_hit}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            if save:
                cv2.imwrite(str(save_dir / f"frame_{frame_count:05d}.jpg"), frame)

            cv2.imshow("Hand-Bill Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                cv2.imwrite(str(save_dir / f"snapshot_{int(time.time())}.jpg"), frame)
                print("Saved snapshot")

        cap.release()
        cv2.destroyAllWindows()
        print(
            f"Processed {frame_count} frame(s): detection in {frames_hit} frame(s), "
            f"{total_det} total detection(s). Avg FPS {fps:.1f}"
        )


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live hand-bill detection from CCTV or Android phone camera")
    parser.add_argument("--url", required=True, help="source: RTSP/HTTP URL, video file, /dev/videoN, or webcam index")
    parser.add_argument("--weights", required=True, help="trained hand_bill YOLO weights, e.g. best.pt")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--device", default="", help='cuda device, e.g. 0, or "cpu"')
    parser.add_argument("--save", action="store_true", help="save every annotated frame")
    parser.add_argument("--out-dir", default="runs/live", help="where saved frames go")
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    LiveDetector(weights=opt.weights, conf=opt.conf, imgsz=opt.imgsz, device=opt.device).run(
        url=opt.url, save=opt.save, out_dir=opt.out_dir
    )

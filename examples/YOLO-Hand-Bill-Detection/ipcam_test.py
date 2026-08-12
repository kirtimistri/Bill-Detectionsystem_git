#!/usr/bin/env python3
"""
Real-time hand-bill detection using an Android phone as an IP camera.

Works with the free "IP Webcam" Android app (Pavel Khlebovich):
  1. Install IP Webcam on your phone.
  2. Connect phone and PC to the SAME Wi-Fi network.
  3. Open IP Webcam, tap "Start server" (bottom).
  4. Note the URL shown, e.g. http://192.168.1.42:8080
  5. Run this script:
       python ipcam_test.py --url http://192.168.1.42:8080/video --weights ../../../models/hand_bill_detector.pt
  6. Press 'q' in the window to quit. Press 's' to save the current frame.

To find the phone's IP: the app prints it, or use `ipconfig`/`ip addr` / scan your router.

Optional: `--save` writes every annotated frame to runs/cctv_ipcam/.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

HIGH_CONF = 0.6
BOX_INSET = 0.1


def box_color(conf: float) -> tuple[int, int, int]:
    """Return BGR color for a detection: green for high confidence, orange for low."""
    return (0, 200, 0) if conf >= HIGH_CONF else (0, 165, 255)


def label_txt_color(conf: float) -> tuple[int, int, int]:
    """Black text on orange (low-confidence) labels for readability, white on green."""
    return (0, 0, 0) if conf < HIGH_CONF else (255, 255, 255)


def tighten_box(box, inset: float, frame_area: float):
    """Shrink a detection box toward its center; bigger boxes shrink more."""
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    if frame_area <= 0 or w <= 0 or h <= 0:
        return box
    k = inset * (1.0 + (w * h) / frame_area)
    nx1, ny1 = x1 + w * k, y1 + h * k
    nx2, ny2 = x2 - w * k, y2 - h * k
    if nx2 - nx1 < 8 or ny2 - ny1 < 8:
        return box
    return nx1, ny1, nx2, ny2


class IPCamDetector:
    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 640, device: str = "") -> None:
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def run(self, url: str, save: bool = False, out_dir: str = "runs/cctv_ipcam") -> None:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open stream {url}. Is IP Webcam running and are both devices on the same Wi-Fi?"
            )
        print(f"Streaming from {url}  (press 'q' to quit, 's' to save frame)")

        save_dir = Path(out_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"OpenCV backends: {cap.getBackendName()} | resolution {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

        frame_count = 0
        prev_time = time.time()
        fps = 0.0
        detections = 0
        frames_with_bill = 0

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Stream ended / frame dropped. Retrying...")
                time.sleep(0.5)
                continue

            frame_count += 1

            results = self.model.predict(
                frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False
            )[0]
            boxes = results.boxes

            annotator = Annotator(frame, line_width=2, font_size=10, pil=False)
            frame_area = frame.shape[0] * frame.shape[1]
            frame_hits = 0
            if boxes is not None and len(boxes):
                for box, c, cls in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()):
                    label = f"{self.model.names.get(int(cls), 'hand_bill')} {float(c):.2f}"
                    annotator.box_label(
                        tighten_box(box, BOX_INSET, frame_area),
                        label,
                        color=box_color(float(c)),
                        txt_color=label_txt_color(float(c)),
                    )
                    frame_hits += 1
                detections += frame_hits
                frames_with_bill += 1

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / (now - prev_time)) if now > prev_time else fps
            prev_time = now

            cv2.putText(
                frame,
                f"FPS {fps:.1f} | bill frames {frames_with_bill}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            if save:
                cv2.imwrite(str(save_dir / f"frame_{frame_count:05d}.jpg"), frame)

            cv2.imshow("IP Webcam - Hand Bill Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                cv2.imwrite(str(save_dir / f"snapshot_{int(time.time())}.jpg"), frame)
                print("Saved snapshot")

        cap.release()
        cv2.destroyAllWindows()
        print(
            f"Processed {frame_count} frame(s): hand holding bill in {frames_with_bill} frame(s), "
            f"{detections} total detection(s). Avg FPS {fps:.1f}"
        )


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time hand-bill detection from an IP Webcam phone stream")
    parser.add_argument("--url", default="http://192.168.1.100:8080/video", help="IP Webcam MJPEG stream URL")
    parser.add_argument("--weights", required=True, help="path to trained hand_bill YOLO weights, e.g. best.pt")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--device", default="", help='cuda device, e.g. 0, or "cpu"')
    parser.add_argument("--save", action="store_true", help="save every annotated frame to disk")
    parser.add_argument("--out-dir", default="runs/cctv_ipcam", help="where saved frames go")
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    IPCamDetector(weights=opt.weights, conf=opt.conf, imgsz=opt.imgsz, device=opt.device).run(
        url=opt.url, save=opt.save, out_dir=opt.out_dir
    )

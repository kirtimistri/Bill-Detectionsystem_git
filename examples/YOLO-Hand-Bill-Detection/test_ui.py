#!/usr/bin/env python3
"""Temporary web UI for testing the hand-bill detector on images.

Zero new dependencies (Python standard library only; reuses the model + drawing
logic already used by main.py, so results match the CCTV/live use case).

Usage:
    ./.venv/bin/python examples/YOLO-Hand-Bill-Detection/test_ui.py [--port 8000]

Then open http://127.0.0.1:8000 in your browser, drag an image in, and hit Detect.
"""

from __future__ import annotations

import argparse
import base64
import collections
import errno
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # reject anything larger than 25 MB
ALLOWED_IMGSZ = {320, 416, 640}
# v4 (a finetune of v2 on the newer dataset) is the accuracy winner: it catches
# everything v2 does AND the phone-photo styles v2 misses, with zero false alarms
# on the test set — so it is the default for both image detection and live streams
DEFAULT_MODEL = "hand_bill_detector_v4.pt"
PRECISION_MODEL = "hand_bill_detector_v4.pt"  # the trustworthy reference for confirmations

import cv2
import numpy as np

# --- reuse the drawing logic from main.py (same colors + box tightening) ------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from main import bill_ok, box_color, label_txt_color, predict_merged, tighten_box  # noqa: E402

REPO_ROOT = SCRIPT_DIR.parent.parent
MODELS_DIR = REPO_ROOT / "models"
# (dropdown label, file name, human note)
MODEL_CHOICES = [
    ("v4 (hand_bill_detector_v4.pt)", "hand_bill_detector_v4.pt", "best — catches phone photos too, zero false alarms (default)"),
    ("v2 (hand_bill_detector_v2.pt)", "hand_bill_detector_v2.pt", "high precision, misses some phone photos"),
    ("v1 (hand_bill_detector.pt)", "hand_bill_detector.pt", "catches more, but false-alarms often"),
    ("last (hand_bill_detector_last.pt)", "hand_bill_detector_last.pt", "similar to v1"),
]

PAGE_PATH = SCRIPT_DIR / "test_ui.html"


def _load_page() -> str:
    """Read the UI from the standalone test_ui.html next to this script."""
    try:
        return PAGE_PATH.read_text(encoding="utf-8")
    except OSError:
        return "<html><body><h1>test_ui.html not found next to test_ui.py</h1></body></html>"


def draw_label(frame, box, label: str, color, txt_color=(255, 255, 255)) -> None:
    """Draw a label INSIDE the top of a detection box, horizontally centered.

    ultralytics' Annotator puts the label ABOVE the box and can push it sideways
    when a box sits near an image edge — making the label look like it points the
    wrong way. Drawing inside the box keeps the label attached to its detection.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    box_w, box_h = x2 - x1, y2 - y1
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    th += 6
    lw = max(10, min(tw + 12, box_w - 4))  # clamp to box width (narrow boxes)
    lx = x1 + max(2, (box_w - lw) // 2)    # horizontally centered inside the box
    ly = y1 + 2
    lh = min(th, max(8, box_h - 4))
    cv2.rectangle(frame, (lx, ly), (lx + lw, ly + lh), color, -1, cv2.LINE_AA)
    cv2.putText(frame, label, (lx + 5, ly + lh - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, txt_color, 2, cv2.LINE_AA)


def annotate_frame(frame, result=None, inset=0.15, banner: bool = True, shape_filter: bool = True, hand_check: bool = True, white_check: bool = True, boxes=None, names=None) -> tuple:
    """Draw tightened confidence-colored boxes + (optional) banner on a frame.

    ``boxes`` overrides ``result.boxes`` and may be an ultralytics Boxes object
    or a plain [(box, conf, cls), ...] list (multi-scale merges); ``names``
    overrides ``result.names`` for label text. Shared by the image endpoint and
    the RTSP stream loop so both look identical.
    Returns (annotated_frame, dets) where dets is the JSON-able detection list.
    """
    frame = frame.copy()
    dets = []
    boxes = result.boxes if (boxes is None and result is not None) else boxes
    names = result.names if (names is None and result is not None) else names
    if names is None:
        names = {0: "hand_bill"}  # safe default when only ``boxes`` is given
    if boxes is not None and len(boxes):
        frame_area = frame.shape[0] * frame.shape[1]
        if isinstance(boxes, (list, tuple)):
            it = boxes  # [(box, conf, cls), ...] from multi-scale merging
        else:
            it = zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy())
        for box, c, cls in it:
            if int(cls) != 0:
                continue  # only the hand_bill class (v4 also predicts no_bill)
            if not bill_ok(frame, box, shape_filter, hand_check, white_check):
                continue  # not bill-shaped, no hand near the bill, or not white -> false alarm
            conf_v = float(c)
            label = f"{names.get(int(cls), 'hand_bill')} {conf_v:.2f}"
            tb = tighten_box(box, inset, frame_area, frame.shape)
            cv2.rectangle(frame, (int(tb[0]), int(tb[1])), (int(tb[2]), int(tb[3])),
                          box_color(conf_v), 3, cv2.LINE_AA)
            draw_label(frame, tb, label, box_color(conf_v), label_txt_color(conf_v))
            dets.append({
                "conf": round(conf_v, 3),
                "box": [round(float(v)) for v in tb],
                "color": "green" if conf_v >= 0.6 else "orange",
            })
        if dets and banner:  # banner, matching main.py
            max_conf = max(d["conf"] for d in dets)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), box_color(max_conf), -1)
            cv2.putText(frame, f"BILL DETECTED x{len(dets)}", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return frame, dets


def downscale(frame, max_w: int = 1280):
    """Cap the display width so large frames transfer/render quickly."""
    h, w = frame.shape[:2]
    if w > max_w:
        scale = max_w / w
        frame = cv2.resize(frame, (max_w, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return frame


def encode_jpg(frame, quality: int = 85) -> bytes:
    """JPEG-encode a frame; returns the raw bytes."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def _boxes_agree(a, b) -> bool:
    """Two detections refer to the same bill when the center of either box lies
    inside the other. Robust to one model outputting a loose box and the other a
    tight one, where raw IoU would wrongly reject the match."""
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return (b[0] <= acx <= b[2] and b[1] <= acy <= b[3]) or (
        a[0] <= bcx <= a[2] and a[1] <= bcy <= a[3]
    )


def _predict_boxes(detector, model, img, imgsz: int, conf: float) -> list:
    """Run one model on an image; returns [(box, confidence, class), ...] (raw xyxy)."""
    with detector._infer_lock:
        result = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
    boxes = []
    if result.boxes is not None and len(result.boxes):
        for box, c, cls in zip(result.boxes.xyxy.cpu().numpy(),
                               result.boxes.conf.cpu().numpy(),
                               result.boxes.cls.cpu().numpy()):
            boxes.append(([float(v) for v in box], float(c), int(cls)))
    return boxes


def annotate_ensemble(frame, main_boxes, partner_boxes, inset: float, tag: str, shape_filter: bool = True, hand_check: bool = True, white_check: bool = True) -> tuple:
    """Draw high-precision mode: green ✓ boxes where both models agree, orange ? otherwise."""
    frame = frame.copy()
    dets = []
    if main_boxes:
        frame_area = frame.shape[0] * frame.shape[1]
        for box, c, cls in main_boxes:
            if cls != 0:
                continue  # only the hand_bill class (v4 also predicts no_bill)
            if not bill_ok(frame, box, shape_filter, hand_check, white_check):
                continue
            # no partner boxes -> the selected model IS the precision model (all ✓)
            confirmed = (not partner_boxes) or any(_boxes_agree(box, pb) for pb, _, _ in partner_boxes)
            tb = tighten_box(box, inset, frame_area, frame.shape)
            mark = "✓" if confirmed else "?"
            color = (0, 200, 0) if confirmed else (0, 165, 255)
            cv2.rectangle(frame, (int(tb[0]), int(tb[1])), (int(tb[2]), int(tb[3])), color, 3, cv2.LINE_AA)
            draw_label(frame, tb, f"{tag} {c:.2f} {mark}", color,
                       (255, 255, 255) if confirmed else (0, 0, 0))
            dets.append({
                "conf": round(c, 3),
                "box": [round(float(v)) for v in tb],
                "color": "green" if confirmed else "orange",
                "confirmed": confirmed,
            })
        n_ok = sum(1 for d in dets if d["confirmed"])
        if n_ok:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 200, 0), -1)
            cv2.putText(frame, f"BILL CONFIRMED x{n_ok}", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
        elif dets:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 165, 255), -1)
            cv2.putText(frame, "Weak — not confirmed by both models", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return frame, dets


class Streamer:
    """Background thread that opens an RTSP/CCTV stream, runs detection on each
    frame, and keeps the latest annotated JPEG for the MJPEG endpoint.

    Also accepts local video files (useful for testing the pipeline before a
    camera is available).
    """

    def __init__(self, detector: DetectorServer) -> None:
        self.detector = detector
        self._lock = threading.Lock()
        self._frame: bytes | None = None  # latest annotated JPEG
        self._status = "idle"
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._placeholder_cache: tuple[str, bytes] | None = None
        self._history = collections.deque(maxlen=2)  # detections of the last 2 frames

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> str:
        with self._lock:
            return self._status

    def confirmed(self) -> bool:
        """True when the last 2 frames both had detections (temporal confirmation)."""
        with self._lock:
            return len(self._history) == 2 and all(self._history)

    def latest(self) -> tuple[bytes | None, str]:
        with self._lock:
            return self._frame, self._status

    def start(self, url: str, model_name: str, conf: float, imgsz: int, inset: float,
              shape_filter: bool = True, hand_check: bool = True, white_check: bool = True) -> None:
        """Stop any current stream and open a new one in a background thread."""
        self.stop()
        self._stop.clear()
        self._history.clear()
        self._set(None, f"Connecting to {url} …")
        self._thread = threading.Thread(
            target=self._loop, args=(url, model_name, conf, imgsz, inset, shape_filter, hand_check, white_check), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=4.0)
            self._thread = None
        self._history.clear()
        self._set(None, "Stream stopped.")

    def _set(self, frame: bytes | None, status: str) -> None:
        with self._lock:
            self._frame = frame
            self._status = status

    def placeholder(self, status: str) -> bytes:
        """A dark status frame shown while no camera frame is available."""
        if self._placeholder_cache is None or self._placeholder_cache[0] != status:
            img = np.full((720, 1280, 3), (23, 26, 31), np.uint8)
            cv2.putText(img, "CCTV / RTSP stream", (40, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        1.3, (88, 166, 255), 3, cv2.LINE_AA)
            for i, line in enumerate(status.splitlines()):
                cv2.putText(img, line, (40, 200 + i * 46), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (230, 237, 243), 2, cv2.LINE_AA)
            self._placeholder_cache = (status, encode_jpg(img, 85))
        return self._placeholder_cache[1]

    def _open(self, url: str):
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)
        return cap

    def _loop(self, url: str, model_name: str, conf: float, imgsz: int, inset: float,
              shape_filter: bool = True, hand_check: bool = True, white_check: bool = True) -> None:
        cap = None
        is_file = not url.lower().startswith(("rtsp://", "http://", "https://", "rtmp://"))
        try:
            model = self.detector.get_model(model_name)
            cap = self._open(url)
            # keep trying to open (a CCTV camera may be briefly offline); Stop cancels
            while not cap.isOpened():
                self._set(None, f"Could not open:\n{url}\nRetrying…")
                if self._stop.wait(2):
                    return
                cap.release()
                cap = self._open(url)
            frame_count = 0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    if is_file:
                        self._set(None, "Video finished (end of file).\nPress Start to replay.")
                        return
                    self._history.clear()  # don't carry a confirmed state across a reconnect
                    self._set(None, f"Lost connection to:\n{url}\nRetrying…")
                    if self._stop.wait(2):  # honor a Stop request while waiting
                        return
                    cap.release()
                    cap = self._open(url)
                    if not cap.isOpened():
                        self._set(None, f"Could not reconnect to:\n{url}")
                        return
                    continue
                with self.detector._infer_lock:
                    result = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
                # temporal confirmation: a bill must be seen on 2 consecutive frames
                # before the alarm triggers — single-frame blips are ignored
                has_dets = (result.boxes is not None and len(result.boxes) > 0 and
                            any(int(c) == 0 for c in result.boxes.cls.cpu().numpy()))
                self._history.append(has_dets)
                confirmed = len(self._history) == 2 and all(self._history)
                ann, dets = annotate_frame(frame, result, inset, banner=confirmed, shape_filter=shape_filter, hand_check=hand_check, white_check=white_check)
                ann = downscale(ann)
                frame_count += 1
                if confirmed:
                    status = f"Streaming · frame {frame_count} · BILL CONFIRMED x{len(dets)}"
                elif dets:
                    status = f"Streaming · frame {frame_count} · {len(dets)} detection(s) — confirming…"
                else:
                    status = f"Streaming · frame {frame_count}"
                # no fixed pacing: inference speed sets the frame rate (previously a
                # 1/12s sleep halved the fps and made the stream feel like it buffers)
                self._set(encode_jpg(ann, 75), status)
                time.sleep(1 / 25)  # ceiling ~25 fps so a fast GPU can't run away
        except Exception as exc:
            self._set(None, f"Stream error: {exc}")
        finally:
            if cap is not None:
                cap.release()



class DetectorServer:
    """Holds the loaded models so they persist across requests."""

    ALLOWED = {c[1] for c in MODEL_CHOICES}

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = Path(models_dir)
        self._models: dict[str, object] = {}
        self._lock = threading.Lock()  # guards model loading
        self._infer_lock = threading.Lock()  # serializes predict (shared model instance)

    def get_model(self, name: str):
        """Load (and cache) a model by file name; validates against the allowlist.

        A lock guarantees each model is loaded exactly once, even when the
        background preloader and the first request race.
        """
        if name not in self.ALLOWED:
            raise ValueError(f"Unknown model: {name!r}")
        with self._lock:
            if name not in self._models:
                from ultralytics import YOLO
                self._models[name] = YOLO(str(self.models_dir / name))
        return self._models[name]


class Handler(BaseHTTPRequestHandler):
    server_version = "BillDetectUI/1.0"

    # -- helpers -------------------------------------------------------------
    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: dict, status: int = 200) -> None:
        self._send(status, json.dumps(obj).encode(), "application/json")

    # -- routes --------------------------------------------------------------
    def do_GET(self):  # noqa: N802 (http.server naming)
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            page = _load_page().replace("__MODELS__", json.dumps([
                {"label": l, "file": f, "note": n} for l, f, n in MODEL_CHOICES
            ]))
            self._send(200, page.encode(), "text/html; charset=utf-8")
        elif path == "/stream.mjpeg":
            self._stream_mjpeg()
        elif path == "/stream/status":
            streamer = self.server.streamer
            self._json({"ok": True, "running": streamer.running, "status": streamer.status(),
                        "confirmed": streamer.confirmed()})
        else:
            self._json({"ok": False, "error": "Not found"}, 404)

    def do_POST(self):  # noqa: N802
        path = self.path.split("?")[0]
        try:
            if path == "/detect":
                self._detect()
            elif path == "/stream/start":
                self._stream_start()
            elif path == "/stream/stop":
                self._stream_stop()
            else:
                self._json({"ok": False, "error": "Not found"}, 404)
        except Exception as exc:  # surface any error to the UI
            print(f"[http] ERROR {path}: {exc}", flush=True)
            self._json({"ok": False, "error": str(exc)}, 500)

    def _detect(self) -> None:
        t_start = time.time()
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_UPLOAD_BYTES:
            return self._json({"ok": False, "error": "Image too large (max 25 MB)"}, 413)
        raw = self.rfile.read(length)
        if not raw:
            return self._json({"ok": False, "error": "No image received"}, 400)

        model_name = self.headers.get("X-Model", DEFAULT_MODEL)
        ensemble = self.headers.get("X-Ensemble", "0").lower() in ("1", "true", "yes")
        multi = self.headers.get("X-MultiScale", "0").lower() in ("1", "true", "yes")
        shape_filter = self.headers.get("X-Shape", "1").lower() in ("1", "true", "yes")
        hand_check = self.headers.get("X-Hand", "1").lower() in ("1", "true", "yes")
        white_check = self.headers.get("X-White", "1").lower() in ("1", "true", "yes")
        if ensemble and multi:  # ensemble already runs two models; multi-scale is redundant
            multi = False
        try:
            conf = float(self.headers.get("X-Conf", "0.25"))
            imgsz = int(self.headers.get("X-Imgsz", "416"))
            inset = float(self.headers.get("X-Inset", "0.15"))
        except ValueError:
            conf, imgsz, inset = 0.25, 416, 0.15
        conf = max(0.0, min(1.0, conf))
        inset = max(0.0, min(0.5, inset))
        if imgsz not in ALLOWED_IMGSZ:
            imgsz = 416

        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return self._json({"ok": False, "error": "Could not decode that file as an image"}, 400)

        model = self.server.detector.get_model(model_name)

        t0 = time.time()
        if ensemble:
            # high-precision mode: v2 (the trustworthy model) is the reference —
            # boxes it agrees with are shown ✓ green, everything else ? orange
            if model_name == PRECISION_MODEL:
                main_boxes = _predict_boxes(self.server.detector, model, img, imgsz, conf)
                partner_boxes = []  # selected model is v2 -> everything is confirmed
            else:
                partner_model = self.server.detector.get_model(PRECISION_MODEL)
                main_boxes = _predict_boxes(self.server.detector, model, img, imgsz, conf)
                partner_boxes = _predict_boxes(self.server.detector, partner_model, img, imgsz, conf)
                # only trust partner boxes of the hand_bill class that pass the
                # bill filter when confirming (a sliver shouldn't rubber-stamp a detection)
                partner_boxes = [pb for pb in partner_boxes if pb[2] == 0 and bill_ok(img, pb[0], True, True, True)]
            elapsed = time.time() - t0
            tag = {"hand_bill_detector.pt": "v1",
                   "hand_bill_detector_v2.pt": "v2",
                   "hand_bill_detector_last.pt": "last",
                   "hand_bill_detector_v4.pt": "v4"}.get(model_name, model_name)
            frame, dets = annotate_ensemble(img, main_boxes, partner_boxes, inset, tag, shape_filter, hand_check, white_check)
            confirmed_count = sum(1 for d in dets if d["confirmed"])
        else:
            if multi:
                with self.server.detector._infer_lock:
                    boxes = predict_merged(model, img, conf)
                frame, dets = annotate_frame(img, None, inset, boxes=boxes,
                                             names={0: "hand_bill"},
                                             shape_filter=shape_filter, hand_check=hand_check, white_check=white_check)
            else:
                with self.server.detector._infer_lock:
                    result = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
                frame, dets = annotate_frame(img, result, inset, shape_filter=shape_filter, hand_check=hand_check, white_check=white_check)
            elapsed = time.time() - t0
            confirmed_count = None
        frame = downscale(frame)
        self._json({
            "ok": True,
            "model": model_name,
            "conf": conf,
            "imgsz": imgsz,
            "count": len(dets),
            "confirmed_count": confirmed_count,
            "detections": dets,
            "elapsed": round(elapsed, 2),
            "image": base64.b64encode(encode_jpg(frame, 85)).decode(),
        })
        total = time.time() - t_start
        mode = "ensemble" if ensemble else ("multi" if multi else "single")
        shown_imgsz = "416+640" if multi else str(imgsz)
        print(
            f"[detect] {model_name} ({mode}) | imgsz={shown_imgsz} conf={conf} | "
            f"{len(dets)} box(es) | infer {elapsed * 1000:.0f} ms | "
            f"total (incl. upload+encode+response) {total * 1000:.0f} ms",
            flush=True,
        )

    # -- stream endpoints ----------------------------------------------------
    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            return json.loads(self.rfile.read(length).decode() or "{}")
        except json.JSONDecodeError:
            raise ValueError("Expected a JSON body") from None

    def _stream_start(self) -> None:
        try:
            body = self._read_json()
        except ValueError as exc:
            return self._json({"ok": False, "error": str(exc)}, 400)
        url = str(body.get("url", "")).strip()
        is_url = url.lower().startswith(("rtsp://", "http://", "https://", "rtmp://"))
        is_file = not is_url and url and Path(url).is_file()  # local video, handy for testing
        if not (is_url or is_file):
            return self._json({
                "ok": False,
                "error": "Enter an RTSP/HTTP stream URL (rtsp://user:pass@ip:554/stream1) or a local video path",
            }, 400)
        try:
            conf = max(0.0, min(1.0, float(body.get("conf", 0.25))))
            imgsz = int(body.get("imgsz", 416))
            inset = max(0.0, min(0.5, float(body.get("inset", 0.15))))
        except (TypeError, ValueError):
            conf, imgsz, inset = 0.25, 416, 0.15
        if imgsz not in ALLOWED_IMGSZ:
            imgsz = 416
        model_name = str(body.get("model", DEFAULT_MODEL))
        if model_name not in DetectorServer.ALLOWED:
            return self._json({"ok": False, "error": f"Unknown model: {model_name!r}"}, 400)
        shape_filter = str(body.get("shape", True)).lower() in ("1", "true", "yes")
        hand_check = str(body.get("hand", True)).lower() in ("1", "true", "yes")
        white_check = str(body.get("white", True)).lower() in ("1", "true", "yes")
        self.server.streamer.start(url, model_name, conf, imgsz, inset, shape_filter, hand_check, white_check)
        print(f"[stream] start {model_name} conf={conf} imgsz={imgsz} inset={inset} <- {url}", flush=True)
        self._json({"ok": True})

    def _stream_stop(self) -> None:
        self.server.streamer.stop()
        print("[stream] stop", flush=True)
        self._json({"ok": True})

    def _stream_mjpeg(self) -> None:
        """Long-lived multipart MJPEG feed: one JPEG per --frame boundary."""
        streamer = self.server.streamer
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                frame, status = streamer.latest()
                if frame is None:
                    frame = streamer.placeholder(status)
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # client closed the tab / stopped the stream

def main() -> None:
    parser = argparse.ArgumentParser(description="Temporary image-testing UI for the hand-bill detector")
    parser.add_argument("--port", type=int, default=8000, help="port to listen on (default 8000)")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    args = parser.parse_args()

    # threaded server so the long-lived MJPEG stream doesn't block other requests;
    # inference itself is serialized by the DetectorServer inference lock
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            print(
                f"Port {args.port} is already in use — another instance is running, "
                "or a previous one still holds the port.\n"
                f"  1) Just open http://{args.host}:{args.port} — the running instance works.\n"
                f"  2) Free the port and retry:  fuser -k {args.port}/tcp   (or:  ss -tlnp | grep {args.port})\n"
                f"  3) Use another port:         python test_ui.py --port {args.port + 1}",
                flush=True,
            )
            sys.exit(1)
        raise
    server.detector = DetectorServer(MODELS_DIR)
    server.streamer = Streamer(server.detector)
    print(f"Hand-Bill detector UI running at http://{args.host}:{args.port}", flush=True)
    # load + warm up the default model at startup so the first click is instant:
    # the ~1.2s one-time torch init is paid here instead of on the first image
    print(f"Models: {MODELS_DIR}  |  Ctrl+C to stop  |  warming up {DEFAULT_MODEL}…", flush=True)
    print("Streams: RTSP/HTTP CCTV URLs or local video files, shown live in the browser.", flush=True)
    try:
        model = server.detector.get_model(DEFAULT_MODEL)
        model.predict(np.zeros((320, 320, 3), np.uint8), imgsz=320, conf=0.5, verbose=False)
        print("Ready — detections are instant.", flush=True)
    except Exception as exc:
        print(f"Warmup failed ({exc}); model will load lazily on first use.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

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
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # reject anything larger than 25 MB

import cv2
import numpy as np

# --- reuse the drawing logic from main.py (same colors + box tightening) ------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from main import box_color, label_txt_color, tighten_box  # noqa: E402
from ultralytics.utils.plotting import Annotator

REPO_ROOT = SCRIPT_DIR.parent.parent
MODELS_DIR = REPO_ROOT / "models"
# (dropdown label, file name, human note)
MODEL_CHOICES = [
    ("v1 (hand_bill_detector.pt)", "hand_bill_detector.pt", "catches most test images"),
    ("v2 (hand_bill_detector_v2.pt)", "hand_bill_detector_v2.pt", "fewer false alarms"),
    ("last (hand_bill_detector_last.pt)", "hand_bill_detector_last.pt", "similar to v1"),
]

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hand-Bill Detector — Test UI</title>
<style>
  :root{--bg:#0d1117;--panel:#161b22;--panel2:#1c2128;--border:#30363d;--text:#e6edf3;
        --muted:#8b949e;--green:#2ea043;--orange:#f0883e;--accent:#58a6ff;--red:#f85149}
  *{box-sizing:border-box}
  body{margin:0;font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text)}
  header{display:flex;align-items:center;gap:12px;padding:14px 24px;border-bottom:1px solid var(--border);background:var(--panel)}
  .dot{width:11px;height:11px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green);animation:pulse 2s infinite}
  @keyframes pulse{50%{opacity:.45}}
  header h1{font-size:17px;margin:0}
  header .sub{color:var(--muted);font-size:13px}
  main{max-width:1140px;margin:22px auto;padding:0 22px;display:grid;grid-template-columns:330px 1fr;gap:22px}
  @media(max-width:860px){main{grid-template-columns:1fr}}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:18px}
  label{display:block;font-size:12px;color:var(--muted);margin:14px 0 6px;text-transform:uppercase;letter-spacing:.04em}
  select,input[type=range]{width:100%;background:var(--panel2);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:9px 10px;font-size:14px}
  select:focus,input:focus{outline:none;border-color:var(--accent)}
  .hint{font-size:11.5px;color:var(--muted);margin-top:5px}
  .confrow{display:flex;align-items:center;gap:12px}
  .confrow output{min-width:52px;text-align:right;font-weight:600;color:var(--accent)}
  .drop{border:2px dashed var(--border);border-radius:12px;padding:34px 16px;text-align:center;color:var(--muted);
        cursor:pointer;transition:.18s;margin-top:6px}
  .drop:hover,.drop.over{border-color:var(--accent);color:var(--text);background:rgba(88,166,255,.06)}
  .drop .ico{font-size:30px;display:block;margin-bottom:8px}
  button{width:100%;margin-top:16px;padding:12px;border:0;border-radius:8px;font-size:15px;font-weight:600;cursor:pointer;
         background:var(--green);color:#fff;transition:.15s}
  button:hover{filter:brightness(1.12)}
  button:disabled{opacity:.5;cursor:not-allowed}
  .result h2{margin:0 0 12px;font-size:15px}
  .imgwrap{position:relative;border:1px solid var(--border);border-radius:10px;overflow:hidden;background:#000;min-height:220px;display:flex;align-items:center;justify-content:center}
  .imgwrap img{max-width:100%;max-height:70vh;display:block}
  .spinner{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;background:rgba(13,17,23,.78);color:var(--muted);font-size:13px}
  .ring{width:34px;height:34px;border:3px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  .chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}
  .chip{padding:6px 10px;border-radius:999px;font-size:12.5px;font-weight:600;display:flex;gap:6px;align-items:center}
  .chip.g{background:rgba(46,160,67,.15);color:#56d364;border:1px solid rgba(46,160,67,.4)}
  .chip.o{background:rgba(240,136,62,.15);color:#ffb37a;border:1px solid rgba(240,136,62,.4)}
  .summary{display:flex;gap:18px;margin-top:12px;flex-wrap:wrap}
  .stat{background:var(--panel2);border:1px solid var(--border);border-radius:10px;padding:10px 14px;min-width:120px}
  .stat b{display:block;font-size:19px}
  .stat span{font-size:11px;color:var(--muted)}
  .err{background:rgba(248,81,73,.12);border:1px solid rgba(248,81,73,.4);color:#ff9a94;border-radius:10px;padding:12px 14px;margin-top:12px;font-size:13px}
  .toggle{font-size:12px;color:var(--accent);background:none;border:none;cursor:pointer;padding:0;margin-top:10px;text-decoration:underline}
</style>
</head>
<body>
<header>
  <div class="dot"></div>
  <h1>Hand-Bill Detector</h1>
  <span class="sub">image testing UI · same pipeline as the CCTV/live scripts</span>
</header>
<main>
  <section class="card">
    <label>Model</label>
    <select id="model"></select>
    <div class="hint" id="modelhint"></div>

    <label>Confidence threshold</label>
    <div class="confrow">
      <input type="range" id="conf" min="0.05" max="0.90" step="0.05" value="0.25">
      <output id="confval">0.25</output>
    </div>

    <div class="drop" id="drop">
      <span class="ico">🖼️</span>
      Drop an image here<br>or click to browse
    </div>
    <input type="file" id="file" accept="image/*" hidden>

    <button id="go" disabled>Detect</button>
    <div class="hint" style="text-align:center">Tip: testing with a phone photo or CCTV frame?</div>
  </section>

  <section class="card result">
    <h2>Result</h2>
    <div class="imgwrap">
      <img id="out" alt="annotated result">
      <div class="spinner" id="spinner" hidden><div class="ring"></div>running detection…</div>
    </div>
    <button class="toggle" id="tog" hidden>show original</button>
    <div class="summary" id="summary"></div>
    <div class="chips" id="chips"></div>
    <div class="err" id="err" hidden></div>
  </section>
</main>

<script>
const MODELS = __MODELS__;
const $ = id => document.getElementById(id);
const modelSel = $('model'), confEl = $('conf'), confVal = $('confval');

// ---- model dropdown ----
MODELS.forEach(m => { const o = document.createElement('option'); o.value = m.file; o.textContent = m.label; modelSel.appendChild(o); });
const pickHint = () => { const m = MODELS.find(x => x.file === modelSel.value); $('modelhint').textContent = '↳ ' + (m ? m.note : ''); };
modelSel.addEventListener('change', pickHint); pickHint();

confEl.addEventListener('input', () => confVal.textContent = (+confEl.value).toFixed(2));

// ---- file selection ----
let currentFile = null, showOrig = false, lastAnnotated = null, lastOriginal = null;
const drop = $('drop'), fileEl = $('file');
drop.addEventListener('click', () => fileEl.click());
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', e => { e.preventDefault(); drop.classList.remove('over'); if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]); });
fileEl.addEventListener('change', () => { if (fileEl.files.length) setFile(fileEl.files[0]); });
function setFile(f) {
  currentFile = f; showOrig = false; $('tog').hidden = true;
  drop.innerHTML = '<span class="ico">✅</span>' + f.name + '<br><small>click to change</small>';
  $('go').disabled = false; detect();
}
// re-detect when model/conf changes
modelSel.addEventListener('change', () => currentFile && detect());
confEl.addEventListener('change', () => currentFile && detect());
$('tog').addEventListener('click', () => { showOrig = !showOrig; $('out').src = showOrig ? lastOriginal : lastAnnotated; $('tog').textContent = showOrig ? 'show annotated' : 'show original'; });

// ---- detection ----
async function detect() {
  if (!currentFile) return;
  const spinner = $('spinner'), err = $('err');
  err.hidden = true; spinner.hidden = false; $('summary').innerHTML = ''; $('chips').innerHTML = ''; $('tog').hidden = true;
  try {
    const res = await fetch('/detect', { method: 'POST', headers: { 'X-Model': modelSel.value, 'X-Conf': confEl.value }, body: currentFile });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || 'Detection failed');
    lastAnnotated = 'data:image/jpeg;base64,' + data.image;
    lastOriginal = data.original_b64 ? 'data:image/jpeg;base64,' + data.original_b64 : lastAnnotated;
    $('out').src = lastAnnotated; $('tog').hidden = true;

    const n = data.count, dt = data.detections || [];
    $('summary').innerHTML =
      `<div class="stat"><b>${n}</b><span>detection${n===1?'':'s'}</span></div>` +
      `<div class="stat"><b>${dt.length ? Math.max(...dt.map(d=>d.conf)).toFixed(2) : '—'}</b><span>best confidence</span></div>` +
      `<div class="stat"><b>${data.elapsed}s</b><span>inference</span></div>`;

    $('chips').innerHTML = dt.length ? dt.map(d =>
      `<span class="chip ${d.color==='green'?'g':'o'}">● ${d.conf.toFixed(2)} · [${d.box.join(', ')}]</span>`).join('') :
      '<div class="hint">No bill found at this threshold.</div>';
  } catch (e) {
    err.hidden = false; err.textContent = '⚠ ' + e.message;
  } finally { spinner.hidden = true; }
}
</script>
</body>
</html>"""


class DetectorServer:
    """Holds the loaded models so they persist across requests."""

    ALLOWED = {c[1] for c in MODEL_CHOICES}

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = Path(models_dir)
        self._models: dict[str, object] = {}

    def get_model(self, name: str):
        """Load (and cache) a model by file name; validates against the allowlist."""
        if name not in self.ALLOWED:
            raise ValueError(f"Unknown model: {name!r}")
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
        if self.path.split("?")[0] in ("/", "/index.html"):
            page = PAGE.replace("__MODELS__", json.dumps([
                {"label": l, "file": f, "note": n} for l, f, n in MODEL_CHOICES
            ]))
            self._send(200, page.encode(), "text/html; charset=utf-8")
        else:
            self._json({"ok": False, "error": "Not found"}, 404)

    def do_POST(self):  # noqa: N802
        if self.path.split("?")[0] != "/detect":
            return self._json({"ok": False, "error": "Not found"}, 404)
        try:
            self._detect()
        except Exception as exc:  # surface any error to the UI
            self._json({"ok": False, "error": str(exc)}, 500)

    def _detect(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_UPLOAD_BYTES:
            return self._json({"ok": False, "error": "Image too large (max 25 MB)"}, 413)
        raw = self.rfile.read(length)
        if not raw:
            return self._json({"ok": False, "error": "No image received"}, 400)

        model_name = self.headers.get("X-Model", "hand_bill_detector.pt")
        conf = max(0.0, min(1.0, float(self.headers.get("X-Conf", "0.25"))))

        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return self._json({"ok": False, "error": "Could not decode that file as an image"}, 400)

        model = self.server.detector.get_model(model_name)

        t0 = time.time()
        result = model.predict(img, imgsz=640, conf=conf, verbose=False)[0]
        elapsed = time.time() - t0

        frame = img.copy()
        boxes = result.boxes
        dets = []
        if boxes is not None and len(boxes):
            frame_area = frame.shape[0] * frame.shape[1]
            annotator = Annotator(frame, line_width=3, pil=False)
            for box, c, cls in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()):
                conf_v = float(c)
                label = f"{result.names.get(int(cls), 'hand_bill')} {conf_v:.2f}"
                tb = tighten_box(box, 0.10, frame_area)
                annotator.box_label(tb, label, color=box_color(conf_v), txt_color=label_txt_color(conf_v))
                dets.append({
                    "conf": round(conf_v, 3),
                    "box": [round(float(v)) for v in tb],
                    "color": "green" if conf_v >= 0.6 else "orange",
                })
            if dets:  # banner, matching main.py
                max_conf = max(d["conf"] for d in dets)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), box_color(max_conf), -1)
                cv2.putText(frame, f"BILL DETECTED x{len(dets)}", (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

        _, buf = cv2.imencode(".jpg", frame)
        _, buf2 = cv2.imencode(".jpg", img)
        self._json({
            "ok": True,
            "model": model_name,
            "conf": conf,
            "count": len(dets),
            "detections": dets,
            "elapsed": round(elapsed, 2),
            "image": base64.b64encode(buf).decode(),
            "original_b64": base64.b64encode(buf2).decode(),
        })

def main() -> None:
    parser = argparse.ArgumentParser(description="Temporary image-testing UI for the hand-bill detector")
    parser.add_argument("--port", type=int, default=8000, help="port to listen on (default 8000)")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    args = parser.parse_args()

    # single-threaded: the UI sends one request at a time and ultralytics predict
    # is not guaranteed thread-safe on a shared model instance
    server = HTTPServer((args.host, args.port), Handler)
    server.detector = DetectorServer(MODELS_DIR)
    print(f"Hand-Bill detector UI running at http://{args.host}:{args.port}")
    print(f"Models: {MODELS_DIR}  |  Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

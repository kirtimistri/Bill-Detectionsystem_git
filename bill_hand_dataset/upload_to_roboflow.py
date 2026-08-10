"""Upload ``raw/`` images together with their auto-generated labels to Roboflow.

Pairs each image in ``bill_hand_dataset/raw/`` with its YOLO label file in
``bill_hand_dataset/labels/`` (same stem) and uploads both to your Roboflow
project. Images without a label file are uploaded without annotations so you
can box them manually in the Roboflow UI.

Requires the ``roboflow`` package and a free API key:

    pip install roboflow
    python upload_to_roboflow.py --api-key YOUR_KEY --workspace YOUR_WORKSPACE --project YOUR_PROJECT

Find your key under Roboflow -> Settings -> API Keys (free account is fine).
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
RAW = BASE / "raw"
LABELS = BASE / "labels"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_opt() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", required=True, help="Roboflow API key (Settings -> API Keys)")
    parser.add_argument("--workspace", required=True, help="Roboflow workspace URL slug")
    parser.add_argument("--project", required=True, help="Roboflow project URL slug")
    parser.add_argument(
        "--batch", default="auto_annotated",
        help="Roboflow batch name for this upload (default: auto_annotated)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="only list what would be uploaded, do not call the API",
    )
    return parser.parse_args()


def main(opt: argparse.Namespace) -> int:
    """Upload images + labels to the Roboflow project."""
    images = sorted(p for p in RAW.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"error: no images found in {RAW}")
        return 1

    paired = 0
    unlabeled = 0
    for img in images:
        lab = LABELS / f"{img.stem}.txt"
        if lab.exists() and lab.stat().st_size > 0:
            paired += 1
        else:
            unlabeled += 1

    print(f"images   : {len(images)}")
    print(f"with labels: {paired}   without: {unlabeled}")
    print(f"workspace: {opt.workspace}   project: {opt.project}   batch: {opt.batch}")

    if opt.dry_run:
        print("\ndry-run - no upload performed")
        return 0

    try:
        from roboflow import Roboflow
    except ImportError:
        print("\n'roboflow' package is not installed. Run:  pip install roboflow")
        return 1

    rf = Roboflow(api_key=opt.api_key)
    project = rf.workspace(opt.workspace).project(opt.project)

    # Older roboflow versions pass annotations through project.upload(...);
    # newer ones (>= 1.1.6) use project.single_upload(...). Decide once.
    supports_annotation = "annotation_path" in inspect.signature(project.upload).parameters
    print(f"roboflow upload API: {'single_upload' if not supports_annotation else 'upload(annotation_path=...)'}")

    uploaded = 0
    failed = []
    for img in images:
        lab = LABELS / f"{img.stem}.txt"
        annotation = str(lab) if lab.exists() and lab.stat().st_size > 0 else None
        try:
            if annotation:
                if supports_annotation:
                    project.upload(
                        str(img), batch_name=opt.batch, annotation_path=annotation
                    )
                else:
                    project.single_upload(
                        str(img), batch_name=opt.batch, annotation_path=annotation
                    )
            else:
                project.upload(str(img), batch_name=opt.batch)
            uploaded += 1
            print(f"  ok  {img.name}" + (f"  (+{lab.name})" if annotation else "  (no label)"))
        except Exception as e:  # noqa: BLE001 - report and continue
            failed.append((img.name, str(e)))
            print(f"  FAIL {img.name}: {e}")

    print(f"\nuploaded: {uploaded}/{len(images)}")
    if failed:
        print("failed:")
        for name, err in failed:
            print(f"  - {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(parse_opt()))

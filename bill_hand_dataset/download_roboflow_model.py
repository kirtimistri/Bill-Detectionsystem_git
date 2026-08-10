"""Download trained model weights from a Roboflow project.

Fetches the trained weights for a given project/version and saves the YOLO
``.pt`` file into ``bill_hand_dataset/models/`` (or a custom output dir).

Usage::

    python download_roboflow_model.py --api-key YOUR_KEY \\
        --workspace kirti-mistri --project receipt-in-hand

If ``--version`` is omitted the script lists the project's versions so you can
pick the trained one. Run it again with ``--version N`` to download.

Requires the ``roboflow`` package (pip install roboflow). API key: Roboflow ->
Settings -> API Keys (free account is fine).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
DEFAULT_OUT = BASE / "models"


def parse_opt() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", required=True, help="Roboflow API key (Settings -> API Keys)")
    parser.add_argument("--workspace", required=True, help="Roboflow workspace URL slug")
    parser.add_argument("--project", required=True, help="Roboflow project URL slug")
    parser.add_argument("--version", type=int, default=None, help="version number to download (default: list versions)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory (default: models/)")
    return parser.parse_args()


def main(opt: argparse.Namespace) -> int:
    """List versions and/or download the trained weights."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("\n'roboflow' package is not installed. Run:  pip install roboflow")
        return 1

    opt.out.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=opt.api_key)
    project = rf.workspace(opt.workspace).project(opt.project)

    versions = project.versions()
    print(f"project: {opt.workspace}/{opt.project}")
    print(f"versions ({len(versions)}):")
    for v in versions:
        # roboflow's Version object exposes attributes; be defensive either way
        if isinstance(v, dict):
            vid, vmodel, vcreated = v.get("id"), v.get("model"), v.get("created_at")
        else:
            vid, vmodel, vcreated = getattr(v, "id", "?"), getattr(v, "model", None), getattr(v, "created_at", "?")
        trained = "TRAINED" if vmodel else "untrained"
        print(f"  version {str(vid):<4} {trained:8} created {vcreated}")

    if opt.version is None:
        print("\nNo --version given. Re-run with --version N to download the trained one.")
        return 0

    version = project.version(opt.version)
    model = version.model
    print(f"\ndownloading trained weights for version {opt.version} -> {opt.out} ...")
    model.download("yolov8", location=str(opt.out))

    downloaded = sorted(opt.out.glob("*.pt")) + sorted(opt.out.glob("*.zip"))
    print("\ndownloaded files:")
    for f in downloaded:
        print(f"  {f}  ({f.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(parse_opt()))

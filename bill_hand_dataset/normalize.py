import glob
import hashlib
import os
import shutil
import sys

from PIL import Image

SRC = r"C:\Users\MY PC\Pictures\bill holding hand"
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(DST, exist_ok=True)
    seen = {}
    kept = 0
    skipped = 0
    for path in sorted(glob.glob(os.path.join(SRC, "*"))):
        ext = os.path.splitext(path)[1].lower()
        if ext not in (".jpg", ".jpeg", ".jfif", ".png"):
            print(f"skip (unsupported): {os.path.basename(path)}")
            continue
        digest = md5(path)
        if digest in seen:
            print(f"duplicate, skip: {os.path.basename(path)} == {seen[digest]}")
            skipped += 1
            continue
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                out_name = f"{kept + 1:03d}.jpg"
                out_path = os.path.join(DST, out_name)
                im.save(out_path, "JPEG", quality=95)
            seen[digest] = out_name
            print(f"kept: {os.path.basename(path):45s} -> {out_name}")
            kept += 1
        except Exception as e:
            print(f"error: {os.path.basename(path)}: {e}")
    print(f"\nTotal kept: {kept}, skipped: {skipped}")


if __name__ == "__main__":
    sys.exit(main())

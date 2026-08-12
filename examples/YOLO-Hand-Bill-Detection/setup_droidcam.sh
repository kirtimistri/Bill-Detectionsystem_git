#!/usr/bin/env bash
# One-time DroidCam setup. Run with:  sudo ./setup_droidcam.sh
set -euo pipefail

DROIDCAM_DIR="/home/kirti/Downloads/droidcam"

if [[ "$(id -u)" -ne 0 ]]; then
    echo "Please run with sudo:  sudo $0"
    exit 1
fi

echo "[1/2] Installing DroidCam client..."
"$DROIDCAM_DIR/install-client"

echo "[2/2] Loading v4l2loopback (standard signed module, no build needed)..."
if ! lsmod | grep -q v4l2loopback; then
    modprobe v4l2loopback exclusive_caps=1 card_label="DroidCam"
    echo "module loaded."
else
    echo "v4l2loopback already loaded."
fi

echo ""
echo "Done. New video device(s):"
ls -1 /dev/video* 2>/dev/null || echo "none found"
echo ""
echo "Next: start droidcam-cli (phone app running, phone + PC on same Wi-Fi):"
echo "  droidcam-cli <PHONE-IP> 4747 /dev/video2 &"
echo "Then detect:"
echo "  python examples/YOLO-Hand-Bill-Detection/live_detect.py --url /dev/video2 --weights models/hand_bill_detector.pt"

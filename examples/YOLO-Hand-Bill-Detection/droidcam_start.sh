#!/usr/bin/env bash
# =============================================================================
# droidcam_start.sh — one-shot DroidCam setup + live hand-bill detection
#
# Steps performed:
#   1. Asks for your phone's IP (shown inside the DroidCam app)
#   2. Sets up Python env (creates repo-local .venv + installs deps if needed)
#   3. Loads the v4l2loopback virtual-camera driver (prompts for sudo password)
#   4. Waits for the phone to appear on Wi-Fi (hints to install/start the app)
#   5. Starts droidcam-cli: phone camera -> virtual device /dev/videoN
#   6. Runs live_detect.py with hand_bill_detector_v2.pt
#
# Usage:  ./droidcam_start.sh
#   Overrides:  DROIDCAM_DIR=/path/to/droidcam  WEIGHTS=/path/to/model.pt
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"   # bill-detection-git

DROIDCAM_DIR="${DROIDCAM_DIR:-$HOME/Downloads/droidcam}"
WEIGHTS="${WEIGHTS:-$REPO_ROOT/models/hand_bill_detector_v2.pt}"
LIVE_SCRIPT="$SCRIPT_DIR/live_detect.py"
VENV_DIR="$REPO_ROOT/.venv"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info() { echo -e "${GREEN}[*]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
fail() { echo -e "${RED}[x]${NC} $*"; exit 1; }

# --- 0. sanity checks ----------------------------------------------------------
command -v droidcam-cli >/dev/null 2>&1 || [ -x "$DROIDCAM_DIR/droidcam-cli" ] \
    || fail "droidcam-cli not found (looked in PATH and $DROIDCAM_DIR). Download DroidCam from https://www.dev47apps.com/droidcam/linux/"
[ -f "$WEIGHTS" ]       || fail "Model weights not found: $WEIGHTS"
[ -f "$LIVE_SCRIPT" ]   || fail "live_detect.py not found: $LIVE_SCRIPT"

# --- 1. phone IP ---------------------------------------------------------------
read -r -p "Phone IP (shown in DroidCam app, e.g. 192.168.1.42): " PHONE_IP
[[ "$PHONE_IP" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]] || fail "Invalid IP address: '$PHONE_IP'"

# --- 2. Python environment ------------------------------------------------------
PY="python3"
if ! python3 -c "import cv2, ultralytics" >/dev/null 2>&1; then
    if [ ! -d "$VENV_DIR" ]; then
        info "Creating virtualenv at $VENV_DIR ..."
        python3 -m venv "$VENV_DIR" || fail \
            "Could not create venv (install with:  sudo apt install python3-venv)."
    fi
    PY="$VENV_DIR/bin/python"
    if ! "$PY" -c "import cv2, ultralytics" >/dev/null 2>&1; then
        info "Installing requirements (cv2 + ultralytics) into venv — this can take a few minutes ..."
        "$VENV_DIR/bin/pip" install --upgrade pip -q
        "$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"
    fi
fi
info "Python env ready: $PY"

# --- 3. v4l2loopback driver (needs sudo) ---------------------------------------
mapfile -t DEVICES_BEFORE < <(ls /dev/video* 2>/dev/null || true)

load_driver() {
    if lsmod | grep -q '^v4l2loopback'; then
        info "v4l2loopback already loaded."
    else
        info "Loading v4l2loopback driver (sudo will ask for your password) ..."
        if ! sudo modprobe v4l2loopback exclusive_caps=1 card_label="DroidCam"; then
            warn "modprobe failed — trying DroidCam's DKMS installer ..."
            sudo "$DROIDCAM_DIR/install-video" || fail \
                "Driver install failed. Run manually:  sudo $DROIDCAM_DIR/install-video"
            sudo modprobe v4l2loopback exclusive_caps=1 card_label="DroidCam" || fail \
                "Driver still not loading after install-video."
        fi
    fi
    sleep 1
}

pick_device() {
    local d new=""
    for d in /dev/video*; do
        [ -e "$d" ] || continue
        if ! printf '%s\n' "${DEVICES_BEFORE[@]}" | grep -qx "$d"; then
            new="$d"; break
        fi
    done
    [ -n "$new" ] || new="/dev/video2"
    [ -e "$new" ] || fail "Virtual camera $new not present. Check driver load."
    echo "$new"
}

# --- 4. wait for phone -----------------------------------------------------------
phone_ok() {
    # droidcam-cli talks to the DroidCam app on TCP 4747; probe both ping
    # and the port (some networks block ICMP while TCP works).
    ping -c 1 -W 2 "$PHONE_IP" >/dev/null 2>&1 \
        || timeout 2 bash -c "</dev/tcp/$PHONE_IP/4747" 2>/dev/null
}

wait_for_phone() {
    info "Waiting for phone at $PHONE_IP ..."
    echo -e "${YELLOW}  If you haven't yet: install \"DroidCam\" from the Play Store, open the app,"
    echo -e "  and make sure the phone is on the SAME Wi-Fi network as this PC.${NC}"
    local tries=0
    while ! phone_ok; do
        tries=$((tries + 1))
        if [ "$tries" -ge 30 ]; then
            warn "Phone at $PHONE_IP not reachable after ~60 s."
            read -r -p "Retry? [y/N] " ans || fail "Aborted."
            [[ "$ans" =~ ^[Yy]$ ]] || fail "Aborted. Check Wi-Fi and the DroidCam app."
            tries=0
        fi
        sleep 2
    done
    info "Phone reachable at $PHONE_IP."
}

check_device_writable() {
    if [ ! -w "$DEVICE" ]; then
        warn "$DEVICE is not writable by user '$USER' (v4l2loopback nodes are usually root:video)."
        if groups "$USER" | grep -qw video; then
            fail "You are in the 'video' group but the node is still not writable. Try:  sudo chmod 666 $DEVICE"
        fi
        fail "Add yourself to the video group and re-login, then re-run:  sudo usermod -aG video \"$USER\""
    fi
}

verify_frames() {
    info "Checking that frames actually arrive from the phone ..."
    if ! timeout 15 "$PY" - "$DEVICE" <<'PYEOF'
import sys, time, cv2
cap = cv2.VideoCapture(sys.argv[1])
deadline = time.time() + 12
ok = False
while time.time() < deadline:
    ok, frame = cap.read()
    if ok and frame is not None:
        ok = True
        break
    time.sleep(0.3)
cap.release()
sys.exit(0 if ok else 1)
PYEOF
    then
        fail "No frames from the phone. Is the DroidCam app open (green screen) and is $PHONE_IP correct? Check with:  v4l2-ctl --list-devices"
    fi
    info "Frame stream OK."
}

# --- 5. start droidcam-cli --------------------------------------------------------
start_droidcam() {
    local cli
    command -v droidcam-cli >/dev/null 2>&1 && cli="$(command -v droidcam-cli)" || cli="$DROIDCAM_DIR/droidcam-cli"
    info "Starting $cli $PHONE_IP 4747 $DEVICE ..."
    "$cli" "$PHONE_IP" 4747 "$DEVICE" &
    DC_PID=$!
    sleep 3
    kill -0 "$DC_PID" 2>/dev/null || fail "droidcam-cli exited. Is the app running on the phone?"
    info "Virtual camera up at $DEVICE (pid $DC_PID)."
    trap 'kill "$DC_PID" 2>/dev/null; info "droidcam-cli stopped."' EXIT
}

# --- run --------------------------------------------------------------------------
load_driver
DEVICE="$(pick_device)"
info "Virtual camera device: $DEVICE"
check_device_writable
wait_for_phone
start_droidcam
verify_frames
info "Starting live detection — press 'q' in the window to quit, 's' to save a snapshot."
"$PY" "$LIVE_SCRIPT" --url "$DEVICE" --weights "$WEIGHTS"

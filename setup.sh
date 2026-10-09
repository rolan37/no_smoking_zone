#!/usr/bin/env bash
# One-time setup for Raspberry Pi OS (64-bit, Bookworm). Run from this folder: ./setup.sh
set -e
cd "$(dirname "$0")"

sudo apt update
sudo apt install -y python3-venv python3-picamera2 libgl1 libglib2.0-0 curl

# --system-site-packages lets the venv see picamera2 installed by apt
python3 -m venv --system-site-packages .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

mkdir -p models
[ -f models/face_landmarker.task ] || curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
[ -f models/hand_landmarker.task ] || curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

[ -f .env ] || { cp .env.example .env; chmod 600 .env; echo "Created .env - edit it and add your Telegram bot token and chat ID."; }

echo
echo "Setup done."
echo "1) Edit .env (nano .env)"
echo "2) Test with a monitor attached:  ./.venv/bin/python smokewatch.py --show"
echo "3) Install the boot service:      ./install-service.sh"

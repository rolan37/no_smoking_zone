#!/usr/bin/env bash
# Installs SmokeWatch as a systemd service that starts at boot. Run from this folder.
set -e
cd "$(dirname "$0")"
DIR="$(pwd)"
USER_NAME="$(whoami)"

sudo tee /etc/systemd/system/smokewatch.service >/dev/null <<EOF
[Unit]
Description=SmokeWatch no-smoking zone monitor
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${USER_NAME}
WorkingDirectory=${DIR}
ExecStart=${DIR}/.venv/bin/python ${DIR}/smokewatch.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo usermod -aG video "${USER_NAME}" || true
sudo systemctl daemon-reload
sudo systemctl enable --now smokewatch
echo "Service installed. Logs: journalctl -u smokewatch -f"

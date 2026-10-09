# SmokeWatch – Raspberry Pi edition (Telegram alerts)

Headless computer-vision monitor for no-smoking zones. It detects a repeated hand-to-mouth
gesture pattern (MediaPipe Face + Hand Landmarker), then sends the admin a **Telegram message with a
photo** of the person (face boxed, time and zone stamped) and writes a line to `events.csv`.
Photos are not kept on the Pi unless you turn `save_snapshots` on.

## Hardware
- Raspberry Pi 4 (4 GB+) or Pi 5, 64-bit Raspberry Pi OS (Bookworm), with internet access
- USB webcam (or Pi Camera Module: set `"camera": "picamera2"` in `config.json`)

## 1. Create the Telegram bot
1. In Telegram, open **@BotFather**, send `/newbot`, follow the prompts, and copy the **token**.
2. Open your new bot and press **Start** (send it any message). Do this from every admin account.
3. Find your chat ID: message **@userinfobot**, or open
   `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser and read `"chat":{"id": ...}`.
4. Put both in `.env` (see below). Several admins: separate IDs with commas.

## 2. Install on the Pi
```bash
cd smokewatch
chmod +x setup.sh install-service.sh
./setup.sh                                  # packages, venv, models, creates .env
nano .env                                   # paste token and chat ID
./.venv/bin/python smokewatch.py --show     # test with a monitor attached (q to quit)
./install-service.sh                        # start automatically at every boot
```

## Admin commands (in the Telegram chat)
`/status` – state, alert count, last alert · `/pause` – stop monitoring (e.g. during events) · `/resume`
Only chat IDs listed in `.env` are obeyed.

## Day to day
- Logs: `journalctl -u smokewatch -f`
- Events: `events.csv` (timestamp, zone, event)
- After editing `config.json` or `.env`: `sudo systemctl restart smokewatch`
- If the Wi-Fi drops, alerts are retried for a short while; very long outages lose the message (the CSV log still has it)

## Tuning (`config.json`)
| Setting | Meaning |
|---|---|
| `closeness` | Fingertip-to-mouth distance as a fraction of face width. Raise if it misses, lower if it over-triggers |
| `hold_seconds` | How long the hand must stay at the mouth to count as one gesture |
| `gestures_needed` / `window_seconds` | Gestures within the window that raise an alert |
| `cooldown_seconds` | Minimum gap between alerts |
| `send_photo` / `mark_face` | Attach a photo / draw a box around the person |
| `quiet_hours` | Disable alerts outside school hours |
| `process_fps` | Frames analysed per second. Lower it if the Pi runs hot |

## Placement tips
Mount the camera at head height, 2–4 m from where people stand, facing them, face at least ~60 px wide.
Avoid strong backlight.

## Privacy and responsibility
- The photo shows a real person, who may be a student. Use it only for the school's discipline or safeguarding process,
  limit the Telegram chat to the named admins, and don't forward it.
- Get written approval from school management, display a camera notice, and inform students and parents.
- Follow local law and school policy on image use. Keep `save_snapshots` off unless your policy allows storage and sets a deletion period.
- The system detects *behaviour*, not the cigarette. Eating, drinking, coughing or phone calls can cause false alarms,
  which is why each alert says "verify before acting". Never treat an alert as proof.
- Keep `.env` private (`chmod 600`). If the bot token leaks, revoke it in @BotFather with `/revoke`.

## Next upgrade
Train a YOLO cigarette/vape detector (e.g. YOLOv8n exported to NCNN/TFLite) and require both signals
(object near the mouth + gesture) before alerting. That cuts false alarms sharply.

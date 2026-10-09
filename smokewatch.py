#!/usr/bin/env python3
"""SmokeWatch (Raspberry Pi edition, Telegram alerts)

Headless computer-vision monitor for no-smoking zones.
Detects a repeated hand-to-mouth gesture pattern using MediaPipe Face + Hand
Landmarker, then sends the admin a Telegram message with a photo and logs the
event to CSV.

Run:   python3 smokewatch.py            (headless, for deployment)
Debug: python3 smokewatch.py --show     (live preview window, press q to quit)

Secrets live in .env (never in config.json):
    TELEGRAM_BOT_TOKEN=123456:ABC...
    TELEGRAM_CHAT_ID=111111111,222222222     (one or more admin chat IDs)
"""
import argparse
import csv
import json
import math
import os
import queue
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import mediapipe as mp
import requests
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

HERE = os.path.dirname(os.path.abspath(__file__))
FINGERTIPS = (4, 8, 12, 16, 20)

DEFAULTS = {
    "zone_name": "Main gate - No Smoking Zone",
    "camera": "usb",              # "usb" (OpenCV) or "picamera2"
    "camera_index": 0,
    "frame_width": 640,
    "frame_height": 480,
    "process_fps": 8,             # max frames analysed per second (saves CPU)
    "closeness": 0.28,            # fingertip-to-mouth distance / face width
    "hold_seconds": 1.0,          # hand must stay at mouth this long = 1 gesture
    "gestures_needed": 3,         # gestures inside the window to raise an alert
    "window_seconds": 90,
    "cooldown_seconds": 30,       # minimum gap between alerts
    "send_photo": True,           # attach the frame to the Telegram alert
    "mark_face": True,            # draw a box around the person in the photo
    "quiet_hours": {"enabled": False, "start": "18:00", "end": "07:00"},
    "log_file": "events.csv",
    "save_snapshots": False,      # also keep photos on the Pi (off for privacy)
    "snapshot_dir": "snapshots",
    "face_model": "models/face_landmarker.task",
    "hand_model": "models/hand_landmarker.task",
}


def load_config(path):
    cfg = dict(DEFAULTS)
    if os.path.exists(path):
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg


def load_env(path):
    """Tiny .env loader (KEY=VALUE per line)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def p(path):
    return path if os.path.isabs(path) else os.path.join(HERE, path)


# ---------------------------------------------------------------- telegram
class Telegram:
    """Sends alerts from a background queue (retries if the network drops)
    and listens for /status /pause /resume from authorised admin chats."""

    def __init__(self, token, chat_ids, on_command):
        self.token = token
        self.chat_ids = [c.strip() for c in chat_ids if c.strip()]
        self.on_command = on_command
        self.q = queue.Queue(maxsize=20)
        self.stop = threading.Event()
        threading.Thread(target=self._sender, daemon=True).start()
        threading.Thread(target=self._poller, daemon=True).start()

    @property
    def ok(self):
        return bool(self.token and self.chat_ids)

    def _api(self, method, data=None, files=None, timeout=20):
        r = requests.post(f"https://api.telegram.org/bot{self.token}/{method}",
                          data=data, files=files, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def send_text(self, text):
        self._enqueue(("text", None, text))

    def send_photo(self, jpeg_bytes, caption):
        self._enqueue(("photo", jpeg_bytes, caption))

    def _enqueue(self, item):
        try:
            self.q.put_nowait(item)
        except queue.Full:
            print("[telegram] queue full, dropping message", flush=True)

    def _sender(self):
        while not self.stop.is_set():
            try:
                kind, blob, text = self.q.get(timeout=1)
            except queue.Empty:
                continue
            for chat in self.chat_ids:
                for attempt in range(5):
                    try:
                        if kind == "photo":
                            self._api("sendPhoto", {"chat_id": chat, "caption": text},
                                      {"photo": ("alert.jpg", blob, "image/jpeg")})
                        else:
                            self._api("sendMessage", {"chat_id": chat, "text": text})
                        break
                    except Exception as e:
                        # never print the exception text: it contains the bot token in the URL
                        print(f"[telegram] send failed ({type(e).__name__}), retry {attempt + 1}", flush=True)
                        time.sleep(5 * (attempt + 1))

    def _poller(self):
        offset = None
        while not self.stop.is_set():
            try:
                res = self._api("getUpdates", {"timeout": 25, "offset": offset or 0}, timeout=35)
                for u in res.get("result", []):
                    offset = u["update_id"] + 1
                    msg = u.get("message") or {}
                    chat = str(msg.get("chat", {}).get("id", ""))
                    text = (msg.get("text") or "").strip().lower().split("@")[0]
                    if chat in self.chat_ids and text.startswith("/"):
                        reply = self.on_command(text)
                        if reply:
                            self._api("sendMessage", {"chat_id": chat, "text": reply})
            except Exception as e:
                print(f"[telegram] poll error ({type(e).__name__})", flush=True)
                time.sleep(10)


# ---------------------------------------------------------------- camera
class Camera:
    def __init__(self, cfg):
        self.kind = cfg["camera"]
        w, h = cfg["frame_width"], cfg["frame_height"]
        if self.kind == "picamera2":
            from picamera2 import Picamera2
            self.cam = Picamera2()
            self.cam.configure(self.cam.create_video_configuration(
                main={"size": (w, h), "format": "RGB888"}))
            self.cam.start()
        else:
            self.cam = cv2.VideoCapture(cfg["camera_index"])
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.cam.isOpened():
                raise RuntimeError("Cannot open camera index %s" % cfg["camera_index"])

    def read(self):
        """Return (rgb, bgr) numpy frames, or (None, None)."""
        if self.kind == "picamera2":
            bgr = self.cam.capture_array()      # "RGB888" is stored B,G,R in memory
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), bgr
        ok, bgr = self.cam.read()
        if not ok:
            return None, None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), bgr

    def close(self):
        if self.kind == "picamera2":
            self.cam.stop()
        else:
            self.cam.release()


# ---------------------------------------------------------------- detector
class GestureDetector:
    def __init__(self, cfg):
        face_opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=p(cfg["face_model"])),
            running_mode=vision.RunningMode.VIDEO, num_faces=3)
        hand_opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=p(cfg["hand_model"])),
            running_mode=vision.RunningMode.VIDEO, num_hands=4)
        self.face = vision.FaceLandmarker.create_from_options(face_opts)
        self.hand = vision.HandLandmarker.create_from_options(hand_opts)
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.near_since = None
        self.counted = False
        self.gestures = deque()
        self.last_bbox = None

    def step(self, rgb, ts_ms, now):
        """Process one frame. Returns (alert_bool, info_dict)."""
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        faces = self.face.detect_for_video(img, ts_ms).face_landmarks
        hands = self.hand.detect_for_video(img, ts_ms).hand_landmarks

        best, best_face = None, None   # smallest normalised fingertip-to-mouth distance
        for f in faces:
            mouth = ((f[13].x + f[14].x) / 2, (f[13].y + f[14].y) / 2)
            face_w = math.hypot(f[234].x - f[454].x, f[234].y - f[454].y) or 1.0
            for h in hands:
                for i in FINGERTIPS:
                    d = math.hypot(h[i].x - mouth[0], h[i].y - mouth[1]) / face_w
                    if best is None or d < best:
                        best, best_face = d, f

        is_near = best is not None and best < self.cfg["closeness"]
        if is_near:
            xs = [pt.x for pt in best_face]
            ys = [pt.y for pt in best_face]
            self.last_bbox = (min(xs), min(ys), max(xs), max(ys))  # normalised 0..1
            if self.near_since is None:
                self.near_since, self.counted = now, False
            elif not self.counted and now - self.near_since >= self.cfg["hold_seconds"]:
                self.counted = True
                self.gestures.append(now)
        else:
            self.near_since, self.counted = None, False

        while self.gestures and now - self.gestures[0] > self.cfg["window_seconds"]:
            self.gestures.popleft()

        alert = False
        if len(self.gestures) >= self.cfg["gestures_needed"]:
            alert = True
            self.gestures.clear()
        return alert, {"faces": len(faces), "hands": len(hands), "dist": best,
                       "near": is_near, "gestures": len(self.gestures),
                       "bbox": self.last_bbox}


# ---------------------------------------------------------------- helpers
def in_quiet_hours(q):
    if not q.get("enabled"):
        return False
    t = datetime.now().strftime("%H:%M")
    s, e = q["start"], q["end"]
    return (s <= t < e) if s <= e else (t >= s or t < e)


def log_event(cfg):
    path = p(cfg["log_file"])
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "zone", "event"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), cfg["zone_name"],
                    "possible_smoking_gesture_pattern"])


def make_photo(bgr, bbox, cfg):
    """Return JPEG bytes: frame with optional face box and timestamp banner."""
    img = bgr.copy()
    h, w = img.shape[:2]
    if cfg["mark_face"] and bbox:
        x0, y0, x1, y1 = int(bbox[0] * w), int(bbox[1] * h), int(bbox[2] * w), int(bbox[3] * h)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 3)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S  ") + cfg["zone_name"]
    cv2.rectangle(img, (0, h - 24), (w, h), (0, 0, 0), -1)
    cv2.putText(img, stamp, (6, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--show", action="store_true", help="show a live preview window (debug)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    load_env(os.path.join(HERE, ".env"))

    for key in ("face_model", "hand_model"):
        if not os.path.exists(p(cfg[key])):
            sys.exit(f"Missing model {cfg[key]}. Run ./setup.sh first.")

    stop = threading.Event()
    paused = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    state = {"alerts": 0, "last_alert": None, "started": datetime.now()}

    def on_command(text):
        if text == "/pause":
            paused.set(); return "⏸ Monitoring paused. Send /resume to restart."
        if text == "/resume":
            paused.clear(); return "▶ Monitoring resumed."
        if text in ("/status", "/start"):
            la = state["last_alert"].strftime("%H:%M:%S") if state["last_alert"] else "none"
            return (f"Zone: {cfg['zone_name']}\n"
                    f"State: {'PAUSED' if paused.is_set() else 'monitoring'}\n"
                    f"Alerts since start: {state['alerts']}\nLast alert: {la}\n"
                    f"Up since: {state['started']:%Y-%m-%d %H:%M}\n"
                    "Commands: /status /pause /resume")
        return "Commands: /status /pause /resume"

    tg = Telegram(os.environ.get("TELEGRAM_BOT_TOKEN", ""),
                  os.environ.get("TELEGRAM_CHAT_ID", "").split(","), on_command)
    if tg.ok:
        tg.send_text(f"✅ SmokeWatch online\nZone: {cfg['zone_name']}")
    else:
        print("[telegram] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set; alerts will be log-only", flush=True)

    cam = Camera(cfg)
    det = GestureDetector(cfg)
    min_dt = 1.0 / max(1, cfg["process_fps"])
    last_proc = last_alert = 0.0
    t0 = time.monotonic()
    print(f"[smokewatch] running in '{cfg['zone_name']}'", flush=True)

    try:
        while not stop.is_set():
            now = time.monotonic()
            rgb, bgr = cam.read()
            if rgb is None:
                time.sleep(0.2)
                continue
            if now - last_proc < min_dt:
                continue
            last_proc = now
            if paused.is_set():
                det.reset()
                time.sleep(0.2)
                continue
            ts_ms = int((now - t0) * 1000)

            alert, info = det.step(rgb, ts_ms, now)

            if alert and now - last_alert > cfg["cooldown_seconds"] \
                    and not in_quiet_hours(cfg["quiet_hours"]):
                last_alert = now
                state["alerts"] += 1
                state["last_alert"] = datetime.now()
                log_event(cfg)
                photo = make_photo(bgr, info["bbox"], cfg) if (cfg["send_photo"] or cfg["save_snapshots"]) else None
                if tg.ok:
                    caption = (f"⚠️ Possible smoking/vaping\nZone: {cfg['zone_name']}\n"
                               f"Time: {datetime.now():%Y-%m-%d %H:%M:%S}\nAlert #{state['alerts']}\n"
                               "Automated detection. Please verify before acting.")
                    if cfg["send_photo"] and photo:
                        tg.send_photo(photo, caption)
                    else:
                        tg.send_text(caption)
                if cfg["save_snapshots"] and photo:
                    os.makedirs(p(cfg["snapshot_dir"]), exist_ok=True)
                    name = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
                    with open(os.path.join(p(cfg["snapshot_dir"]), name), "wb") as f:
                        f.write(photo)
                print(f"[ALERT #{state['alerts']}] {datetime.now():%H:%M:%S} {cfg['zone_name']}", flush=True)

            if args.show:
                view = bgr.copy()
                d = "-" if info["dist"] is None else f"{info['dist']:.2f}"
                txt = f"faces {info['faces']} hands {info['hands']} dist {d} gestures {info['gestures']} alerts {state['alerts']}"
                cv2.putText(view, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 0, 255) if info["near"] else (0, 255, 0), 2)
                cv2.imshow("SmokeWatch", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if tg.ok:
            tg.send_text("🔴 SmokeWatch stopped")
            time.sleep(2)
        tg.stop.set()
        cam.close()
        if args.show:
            cv2.destroyAllWindows()
        print("[smokewatch] stopped", flush=True)


if __name__ == "__main__":
    main()

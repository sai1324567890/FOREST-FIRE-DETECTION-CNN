#!/usr/bin/env python3
import argparse
import logging
import os
import smtplib
import time
from email.message import EmailMessage
from pathlib import Path

import cv2
from tensorflow import keras

IMG = 128


def send_email(subject, body, attachment):
    host, user, pwd, to = (os.getenv(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASS", "ALERT_TO"))
    if not all([host, user, pwd, to]):
        return False
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, user, to
    msg.set_content(body)
    if attachment and Path(attachment).exists():
        msg.add_attachment(Path(attachment).read_bytes(), maintype="image", subtype="jpeg",
                           filename=Path(attachment).name)
    with smtplib.SMTP_SSL(host, int(os.getenv("SMTP_PORT", 465))) as s:
        s.login(user, pwd)
        s.send_message(msg)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0 for webcam, or video path / RTSP URL")
    ap.add_argument("--model", default="outputs/fire_cnn.keras")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--consecutive", type=int, default=5, help="positive frames needed to alert")
    ap.add_argument("--cooldown", type=int, default=60, help="seconds between alerts")
    ap.add_argument("--no-display", action="store_true")
    a = ap.parse_args()

    Path("alerts").mkdir(exist_ok=True)
    logging.basicConfig(filename="alerts/alerts.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    model = keras.models.load_model(a.model)
    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source: {a.source}")

    streak, last_alert = 0, 0.0
    print("Monitoring... press 'q' in the video window (or Ctrl+C) to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(cv2.resize(frame, (IMG, IMG)), cv2.COLOR_BGR2RGB).astype("float32")
            prob = float(model.predict(rgb[None], verbose=0)[0][0])
            streak = streak + 1 if prob >= a.threshold else 0
            if streak >= a.consecutive and time.time() - last_alert > a.cooldown:
                last_alert = time.time()
                snap = f"alerts/fire_{int(last_alert)}.jpg"
                cv2.imwrite(snap, frame)
                msg = f"FIRE ALERT! P(fire)={prob:.2f}  snapshot={snap}"
                print("\a" + msg)
                logging.warning(msg)
                try:
                    if send_email("Forest fire alert", msg, snap):
                        logging.info("Alert e-mail sent")
                except Exception as e:
                    logging.error("E-mail failed: %s", e)
            if not a.no_display:
                color = (0, 0, 255) if streak else (0, 200, 0)
                cv2.putText(frame, f"P(fire)={prob:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.imshow("Fire Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# Detection of Fire Combustion in Forest Using Deep Learning

A CNN (TensorFlow/Keras) that classifies surveillance frames as **fire / no_fire**, plus a
real-time monitor that raises alerts (console, log, snapshot, optional e-mail).
False positives are reduced with data augmentation, a probability threshold and a
**consecutive-frame rule** (alert only after N positive frames).

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Get data
**Quick test (synthetic, no download):**
```bash
python make_demo_dataset.py --n 200
```
**Real results:** download a public dataset (e.g. Kaggle "Fire Dataset", "Forest Fire Dataset", or FLAME)
and arrange it as:
```
data/fire/*.jpg
data/no_fire/*.jpg
```

## 2. Train
```bash
python train.py --data data --epochs 15
```
Outputs `outputs/fire_cnn.keras`, `outputs/metrics.json` (accuracy, precision, recall, false positives)
and `outputs/training_curves.png`.

## 3. Predict
```bash
python predict.py path/to/image.jpg
python predict.py path/to/folder
```

## 4. Real-time alerts
```bash
python realtime_alert.py --source 0                       # webcam
python realtime_alert.py --source forest.mp4 --consecutive 5 --threshold 0.8
```
Optional e-mail: set `SMTP_HOST`, `SMTP_USER`, `SMTP_PASS`, `ALERT_TO` (and `SMTP_PORT`).

> The synthetic dataset only verifies that the pipeline works. Report metrics only from a real dataset.

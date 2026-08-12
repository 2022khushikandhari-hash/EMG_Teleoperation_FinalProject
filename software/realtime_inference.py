
"""
REAL-TIME INFERENCE — paste this into a separate file and run it
while Arduino is sending serial data at 115200 baud.
"""
import serial
import pickle
import numpy as np
from collections import deque

MODEL_PATH = "emg_model.pkl"
SERIAL_PORT = "COM3"     # change to your port (Linux: /dev/ttyUSB0)
BAUD = 115200
WINDOW_SIZE = 20

model = pickle.load(open(MODEL_PATH, "rb"))
CLASSES = {0:"REST", 1:"ELBOW_FLEX", 2:"ELBOW_EXTEND",
           3:"WRIST_PRONATE", 4:"GRIP_CLOSE", 5:"GRIP_OPEN"}

b_buf = deque(maxlen=WINDOW_SIZE)
t_buf = deque(maxlen=WINDOW_SIZE)

ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
print("Listening... Ctrl+C to stop")

while True:
    line = ser.readline().decode("utf-8", errors="ignore").strip()
    # Expected format: "Bicep_AD8232:XX,Tricep_EXG:YY"
    try:
        parts = dict(p.split(":") for p in line.split(","))
        b = float(parts["Bicep_AD8232"])
        t = float(parts["Tricep_EXG"])
        b_buf.append(b)
        t_buf.append(t)

        if len(b_buf) == WINDOW_SIZE:
            bw = np.array(b_buf)
            tw = np.array(t_buf)

            feats = [
                np.mean(bw), np.max(bw), np.std(bw), np.sqrt(np.mean(bw**2)), np.ptp(bw),
                np.mean(tw), np.max(tw), np.std(tw), np.sqrt(np.mean(tw**2)), np.ptp(tw),
                np.mean(bw) / (np.mean(tw) + 1e-6),
                np.mean(bw) + np.mean(tw),
                np.mean(bw) - np.mean(tw),
                min(np.mean(bw), np.mean(tw)),
            ]

            pred = model.predict([feats])[0]
            prob = model.predict_proba([feats])[0].max()

            if prob > 0.65:   # confidence threshold
                print(f"  Gesture: {CLASSES[pred]:<18} confidence: {prob:.2f}")

    except Exception:
        pass

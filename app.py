from flask import Flask, render_template, Response
import cv2
import os
import numpy as np
from datetime import datetime
import csv

app = Flask(__name__)

# Paths
DATASET_DIR = "dataset"
TRAINER_PATH = "trainer/trainer.yml"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
ATTENDANCE_FILE = "attendance.csv"

# Load recognizer and face cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(TRAINER_PATH)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
names = sorted(os.listdir(DATASET_DIR))

if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Employee ID", "Date", "Time"])

recorded_today = set()
today = datetime.now().strftime("%d-%m-%Y")
with open(ATTENDANCE_FILE, "r") as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
        if len(row) >= 3 and row[2] == today:
            recorded_today.add(row[0])

camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)

            for (x, y, w, h) in faces:
                id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                if confidence < 70:
                    name_folder = names[id_]
                    try:
                        first, last, emp_id = name_folder.split("_")
                        full_name = f"{first} {last}"
                    except:
                        full_name = name_folder
                        emp_id = "Unknown"

                    if full_name not in recorded_today:
                        now = datetime.now()
                        with open(ATTENDANCE_FILE, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([full_name, emp_id, now.strftime("%d-%m-%Y"), now.strftime("%H:%M:%S")])
                        recorded_today.add(full_name)
                        print(f"✅ Attendance recorded for {full_name}")

                    label = f"{full_name} ({int(confidence)}%)"
                    color = (0, 255, 0)
                else:
                    label = "Unknown"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Encode frame to JPEG for web display
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

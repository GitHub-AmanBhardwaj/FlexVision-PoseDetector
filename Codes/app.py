from flask import Flask, render_template, request
import cv2
import mediapipe as mp
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

POSE_DATA_FILE = "pose_data.pkl"

def load_pose_data():
    try:
        with open(POSE_DATA_FILE, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {"data": [], "labels": []}

def save_pose_data(data):
    with open(POSE_DATA_FILE, "wb") as f:
        pickle.dump(data, f)

def capture_pose_landmarks(image):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        landmarks = [
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in results.pose_landmarks.landmark
        ]
        return np.array(landmarks).flatten()
    return None

def add_pose(pose_name):
    data = load_pose_data()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Add Pose", frame)
        cv2.setWindowProperty("Add Pose", cv2.WND_PROP_TOPMOST, 1)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            landmarks = capture_pose_landmarks(frame)
            if landmarks is not None:
                data["data"].append(landmarks)
                data["labels"].append(pose_name)
                print('Captured')
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    save_pose_data(data)


def detect_pose():
    data = load_pose_data()
    if not data["data"]:
        print('NO PICKLE DATA')
        return

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(data["data"], data["labels"])

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = capture_pose_landmarks(frame)
            if landmarks is not None:
                landmarks = landmarks.reshape(1, -1)
                prediction = knn.predict(landmarks)[0]
                confidence = knn.predict_proba(landmarks).max()

                cv2.putText(frame, f"~Pose: {prediction} ({confidence:.2f})", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (249, 245, 152), 1)

        cv2.imshow("Detect Pose", frame)
        cv2.setWindowProperty("Detect Pose", cv2.WND_PROP_TOPMOST, 1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/live', methods=['GET','POST'])
def live():
    if request.method=='POST':
        detect_pose()
    return render_template('live.html')

@app.route('/add', methods=['GET','POST'])
def add():
    if request.method == 'POST':
        pose_name = request.form['poseName']
        add_pose(pose_name)
    return render_template('add.html')

@app.route('/links')
def links():
    return render_template('links.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)
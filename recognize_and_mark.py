import face_recognition
import cv2
import pickle
import csv
from datetime import datetime
from collections import defaultdict, deque

from head_pose import get_attention_score
from face_align import align_face

# Load encodings
with open("encodings.pickle", "rb") as f:
    known_faces = pickle.load(f)

known_names = list(known_faces.keys())
known_encodings = list(known_faces.values())

present_students = set()
attention_windows = defaultdict(lambda: deque(maxlen=30))
smoothed_scores = defaultdict(float)

cap = cv2.VideoCapture(0)
print("[INFO] Attendance system started. Press 'q' to quit.")

def smooth_average(prev_avg, new_val, alpha=0.6):
    return alpha * new_val + (1 - alpha) * prev_avg

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding, box in zip(encodings, boxes):
        distances = face_recognition.face_distance(known_encodings, encoding)
        min_dist = distances.min()
        best_index = distances.argmin()

        if min_dist < 0.45:
            name = known_names[best_index]

            top, right, bottom, left = box

            # Add margin and check size
            margin = 10
            top = max(0, top - margin)
            bottom = min(frame.shape[0], bottom + margin)
            left = max(0, left - margin)
            right = min(frame.shape[1], right + margin)

            if (bottom - top) < 50 or (right - left) < 50:
                continue

            face_region = frame[top:bottom, left:right]
            if face_region.size == 0:
                continue

            aligned_face = align_face(face_region)
            attention = get_attention_score(aligned_face)

            attention_windows[name].append(attention)
            smoothed_scores[name] = smooth_average(smoothed_scores[name], attention)

            buffer_len = len(attention_windows[name])
            avg_attention = sum(attention_windows[name]) / buffer_len if buffer_len > 0 else 0

            if buffer_len == 30 and avg_attention >= 0.7 and name not in present_students:
                present_students.add(name)
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open("attendance.csv", "a", newline="") as f:
                    csv.writer(f).writerow([name, now, avg_attention])
                print(f"[INFO] {name} marked present at {now}")

            # Draw visuals
            color = (0, 255, 0) if avg_attention >= 0.7 else (0, 255, 255) if avg_attention >= 0.4 else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            label = f"{name} [{int(smoothed_scores[name] * 100)}%]"
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Draw buffer bar
            bar_x, bar_y = left, bottom + 10
            bar_width = right - left
            fill_width = int((buffer_len / 30) * bar_width)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 10), (200, 200, 200), 1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + 10), color, -1)

    cv2.imshow("Attention-Aware Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

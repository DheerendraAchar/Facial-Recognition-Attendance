# capture_faces.py

import cv2
import os

# Ask the user for student info
student_id = input("Enter Student ID (e.g., 2023001): ")
student_name = input("Enter Student Name (e.g., Anil): ")

# Create a folder name like: dataset/2023001_John_Doe
folder_name = f"dataset/{student_id}_{student_name.replace(' ', '_')}"
os.makedirs(folder_name, exist_ok=True)
# Initialize webcam
cap = cv2.VideoCapture(0)
# Load OpenCV's built-in Haar cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 0
max_images = 50

print("[INFO] Starting webcam to capture face images...")
print("Press 'q' to quit early.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Could not access webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert frame to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        count += 1
        # Crop and save the face
        face_img = frame[y:y+h, x:x+w]
        file_path = os.path.join(folder_name, f"{count}.jpg")
        cv2.imwrite(file_path, face_img)

        # Draw a rectangle around the face for feedback
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Images: {count}/{max_images}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Show the live video
    cv2.imshow("Face Capture", frame)

    # Break if user presses 'q' or we've captured enough images
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= max_images:
        break

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Captured {count} images for {student_name} at {folder_name}")

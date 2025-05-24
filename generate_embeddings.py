# generate_embeddings.py

import os
import pickle
import numpy as np
import face_recognition
from mtcnn import MTCNN
from face_align import align_face
import cv2

dataset_path = "dataset"
encodings_dict = {}
detector = MTCNN()

print("[INFO] Generating aligned face embeddings...")

for student_folder in os.listdir(dataset_path):
    student_path = os.path.join(dataset_path, student_folder)
    if not os.path.isdir(student_path):
        continue

    embeddings = []

    for img_name in os.listdir(student_path):
        img_path = os.path.join(student_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        aligned = align_face(img)
        if aligned is None:
            continue

        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model='hog')
        encs = face_recognition.face_encodings(rgb, boxes)

        if encs:
            embeddings.append(encs[0])

    if embeddings:
        encodings_dict[student_folder] = np.mean(embeddings, axis=0)
        print(f"[INFO] Encoded {student_folder} ({len(embeddings)} images)")

with open("encodings.pickle", "wb") as f:
    pickle.dump(encodings_dict, f)

print("[INFO] Saved averaged encodings to encodings.pickle")

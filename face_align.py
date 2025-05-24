import cv2
import numpy as np
import face_recognition

def align_face(image):
    face_landmarks_list = face_recognition.face_landmarks(image)
    if not face_landmarks_list:
        return image  # Return original if no face

    landmarks = face_landmarks_list[0]
    if 'left_eye' not in landmarks or 'right_eye' not in landmarks:
        return image

    left_eye = np.mean(landmarks['left_eye'], axis=0)
    right_eye = np.mean(landmarks['right_eye'], axis=0)

    # Calculate angle
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Rotate image around center
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned_image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return aligned_image

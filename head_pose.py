import cv2
import numpy as np
import math
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def get_attention_score(image):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return 0.0  # No face detected

        landmarks = results.multi_face_landmarks[0].landmark

        # Define key points for head pose
        nose_tip = landmarks[1]
        chin = landmarks[152]
        left_eye = landmarks[263]
        right_eye = landmarks[33]
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]

        # Convert normalized coordinates to pixel coordinates
        h, w = image.shape[:2]
        points_2d = np.array([
            [nose_tip.x * w, nose_tip.y * h],
            [chin.x * w, chin.y * h],
            [left_eye.x * w, left_eye.y * h],
            [right_eye.x * w, right_eye.y * h],
            [left_mouth.x * w, left_mouth.y * h],
            [right_mouth.x * w, right_mouth.y * h]
        ], dtype='double')

        points_3d = np.array([
            [0.0, 0.0, 0.0],
            [0.0, -63.6, -12.5],
            [-43.3, 32.7, -26.0],
            [43.3, 32.7, -26.0],
            [-28.9, -28.9, -24.1],
            [28.9, -28.9, -24.1]
        ])

        focal_length = w
        cam_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype='double')

        dist_coeffs = np.zeros((4, 1))
        success, rotation_vec, _ = cv2.solvePnP(points_3d, points_2d, cam_matrix, dist_coeffs)

        if not success:
            return 0.0

        # Convert rotation vector to Euler angles
        rot_mat, _ = cv2.Rodrigues(rotation_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)

        pitch = angles[0]  # Head up/down
        yaw = angles[1]    # Head left/right

        # Normalize to attention score
        pitch_thresh = 10
        yaw_thresh = 20

        attention = 1.0
        if abs(pitch) > pitch_thresh or abs(yaw) > yaw_thresh:
            attention = 0.0
        elif abs(pitch) > pitch_thresh / 2 or abs(yaw) > yaw_thresh / 2:
            attention = 0.5

        return attention

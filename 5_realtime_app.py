import cv2 as opencv
import time
import pandas
import mediapipe
from pathlib import Path
from mediapipe.framework.formats import landmark_pb2

# Initialize MediaPipe Holistic model
mediapipe_holistic = mediapipe.solutions.holistic
mediapipe_drawing = mediapipe.solutions.drawing_utils
mediapipe_drawing_styles = mediapipe.solutions.drawing_styles

# Set up MediaPipe Holistic model with specified parameters
holistic = mediapipe_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def process_frame_with_holistic(original_frame):
    rgb_frame = opencv.cvtColor(original_frame, opencv.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    mp_detected_frame = holistic.process(rgb_frame)
    rgb_frame.flags.writeable = True
    frame_output = opencv.cvtColor(rgb_frame, opencv.COLOR_RGB2BGR)

    return mp_detected_frame, frame_output


def euclidean_distance(pointA, pointB):
    euclidean_output = ((pointA.x - pointB.x) ** 2 + (pointA.y - pointB.y) ** 2) ** 0.5
    return euclidean_output


def normalize_landmarks(landmarks, shoulder_center_point, shoulder_width):
    zero_landmark_output = [(0, 0)] * len(landmarks)
    normalized_landmark_output = [
        (
            (landmark_point.x - shoulder_center_point[0]) / shoulder_width,
            (landmark_point.y - shoulder_center_point[1]) / shoulder_width
        )
        for landmark_point in landmarks
    ]

    if not landmarks or shoulder_width == 0:
        return zero_landmark_output
    else:
        return normalized_landmark_output
    

def get_body_reference_points(mp_detected_frame):
    if not mp_detected_frame.pose_landmarks:
        return None, None, None
    
    body_landmark_0_to_16 = mp_detected_frame.pose_landmarks.landmark[1:17]


def get_normalized_holistic_landmarks(mp_detected_frame,
                                      body_landmark_0_to_16,
                                      shoulder_center_point,
                                      shoulder_width):
    
    normalized_0_to_16_body_landmarks = normalize_landmarks(
        body_landmark_0_to_16,
        shoulder_center_point,
        shoulder_width
    )

    normalized_right_hand_landmarks = normalize_landmarks(
        mp_detected_frame.right_hand_landmarks.landmark,
        shoulder_center_point,
        shoulder_width
    ) if mp_detected_frame.right_hand_landmarks else [(0, 0)] * 21

    normalized_left_hand_landmarks = normalize_landmarks(
        mp_detected_frame.left_hand_landmarks.landmark,
        shoulder_center_point,
        shoulder_width
    ) if mp_detected_frame.left_hand_landmarks else [(0, 0)] * 21

    return normalized_0_to_16_body_landmarks, normalized_right_hand_landmarks, normalized_left_hand_landmarks


def flatten_normalized_landmarks(normalized_0_to_16_body_landmarks,
                                 normalized_right_hand_landmarks,
                                 normalized_left_hand_landmarks):
    flatten_landmarks = []

    for landmark_x, landmark_y in (normalized_0_to_16_body_landmarks +
                                   normalized_right_hand_landmarks +
                                   normalized_left_hand_landmarks):
        flatten_landmarks.extend([landmark_x, landmark_y])

    return flatten_landmarks


def main():
    capture = opencv.VideoCapture(1)
    with holistic :
        while capture.isOpened():
            success, original_frame = capture.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            mp_detected_frame, original_frame = process_frame_with_holistic(original_frame)

            mediapipe_drawing.draw_landmarks(
                original_frame,
                mp_detected_frame.pose_landmarks,
                mediapipe_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mediapipe_drawing_styles.get_default_pose_landmarks_style()
            )

            opencv.imshow('MediaPipe Holistic', opencv.flip(original_frame, 1))
            if opencv.waitKey(5) & 0xFF == 27:
                break
    capture.release()

if __name__ == "__main__":
    main()
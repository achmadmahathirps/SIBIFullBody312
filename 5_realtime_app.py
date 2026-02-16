import time
import pandas
import cv2 as opencv
import mediapipe
from pathlib import Path
from mediapipe.framework.formats import landmark_pb2

# Initialize MediaPipe Holistic model
mediapipe_holistic = mediapipe.solutions.holistic
mediapipe_drawing = mediapipe.solutions.drawing_utils
mediapipe_drawing_styles = mediapipe.solutions.drawing_styles


# Distance and landmark normalization
def euclidean_distance(pointA, pointB):
    return ((pointA.x - pointB.x) ** 2 + (pointA.y - pointB.y) ** 2) ** 0.5


# Normalize landmarks based on the distance between the shoulders
def normalize_landmarks(landmarks, shoulder_center_point, shoulder_width):
    if not landmarks or shoulder_width == 0:
        return [(0, 0)] * len(landmarks)

    return [
        (
            (landmark_point.x - shoulder_center_point[0]) / shoulder_width,
            (landmark_point.y - shoulder_center_point[1]) / shoulder_width
        )
        for landmark_point in landmarks
    ]
    

# Normalize pose and hands using should-based coordinates
def get_normalized_pose_and_hands(detection_results,
                                  pose_landmark_1_to_16,
                                  shoulder_center_point,
                                  shoulder_width):
    normalized_custom_pose = normalize_landmarks(
        pose_landmark_1_to_16,
        shoulder_center_point,
        shoulder_width
    )

    normalized_right_hand = normalize_landmarks(
        detection_results.right_hand_landmarks.landmark,
        shoulder_center_point,
        shoulder_width
    ) if detection_results.right_hand_landmarks else [(0, 0)] * 21

    normalized_left_hand = normalize_landmarks(
        detection_results.left_hand_landmarks.landmark,
        shoulder_center_point,
        shoulder_width
    ) if detection_results.left_hand_landmarks else [(0, 0)] * 21

    return normalized_custom_pose, normalized_right_hand, normalized_left_hand


# Flatten normalized landmarks into a single list
def flatten_normalized_landmarks(normalized_custom_pose,
                                 normalized_right_hand,
                                 normalized_left_hand):
    extracted_frame_landmarks = []

    for landmark_x, landmark_y in (
        normalized_custom_pose + normalized_right_hand + normalized_left_hand
    ):
        extracted_frame_landmarks.extend([landmark_x, landmark_y])

    return extracted_frame_landmarks


def main():
    capture = opencv.VideoCapture(0)
    with mediapipe_holistic.Holistic(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while capture.isOpened():
            success, frame = capture.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            frame.flags.writeable = False
            frame = opencv.cvtColor(frame, opencv.COLOR_BGR2RGB)
            results = holistic.process(frame)

            frame.flags.writeable = True
            frame = opencv.cvtColor(frame, opencv.COLOR_RGB2BGR)

            mediapipe_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mediapipe_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mediapipe_drawing_styles.get_default_pose_landmarks_style()
            )
            
            opencv.imshow('MediaPipe Holistic', opencv.flip(frame, 1))
            if opencv.waitKey(5) & 0xFF == 27:
                break
    capture.release()
    opencv.destroyAllWindows()

if __name__ == "__main__":
    main()

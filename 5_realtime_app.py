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
    if not landmarks or shoulder_width == 0:
        return [(0.0, 0.0)] * (len(landmarks) if landmarks else 0)

    return [
        (
            (lm.x - shoulder_center_point[0]) / shoulder_width,
            (lm.y - shoulder_center_point[1]) / shoulder_width
        )
        for lm in landmarks
    ]
 

def get_body_reference_points(mp_detected_frame):
    if not mp_detected_frame.pose_landmarks:
        return None, None, None
    
    body_landmark_0_to_16 = mp_detected_frame.pose_landmarks.landmark[1:17]

    left_shoulder_point = mp_detected_frame.pose_landmarks.landmark[
        mediapipe_holistic.PoseLandmark.LEFT_SHOULDER
    ]

    right_shoulder_point = mp_detected_frame.pose_landmarks.landmark[
        mediapipe_holistic.PoseLandmark.RIGHT_SHOULDER
    ]

    shoulder_center_point = (
        (left_shoulder_point.x + right_shoulder_point.x) / 2,
        (left_shoulder_point.y + right_shoulder_point.y) / 2
    )

    shoulder_width = euclidean_distance(left_shoulder_point, right_shoulder_point)

    return body_landmark_0_to_16, shoulder_center_point, shoulder_width


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


def draw_custom_landmarks(original_frame, mp_detected_frame):
    if mp_detected_frame.pose_landmarks:
        filtered_pose = landmark_pb2.NormalizedLandmarkList(
            landmark=[
                mp_detected_frame.pose_landmarks.landmark[i]
                for i in range(0, 17)
            ]
        )

        filtered_connections = [
            connection for connection in mediapipe_holistic.POSE_CONNECTIONS
            if connection[0] < 17 and connection[1] < 17
        ]

        mediapipe_drawing.draw_landmarks(
            original_frame,
            filtered_pose,
            filtered_connections,
            landmark_drawing_spec=mediapipe_drawing_styles.get_default_pose_landmarks_style()
        )

    if mp_detected_frame.right_hand_landmarks:
        mediapipe_drawing.draw_landmarks(
            original_frame,
            mp_detected_frame.right_hand_landmarks,
            mediapipe_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mediapipe_drawing_styles.get_default_hand_landmarks_style()
        )

    if mp_detected_frame.left_hand_landmarks:
        mediapipe_drawing.draw_landmarks(
            original_frame,
            mp_detected_frame.left_hand_landmarks,
            mediapipe_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mediapipe_drawing_styles.get_default_hand_landmarks_style()
        )
    
    return original_frame


def extract_landmarks_from_original_frame(original_frame):
    mp_detected_frame, original_frame = process_frame_with_holistic(original_frame)

    body_landmark_0_to_16, shoulder_center_point, shoulder_width = get_body_reference_points(mp_detected_frame)

    if body_landmark_0_to_16 is None:
        return None, original_frame
    
    normalized_0_to_16_body_landmarks, normalized_right_hand_landmarks, normalized_left_hand_landmarks = get_normalized_holistic_landmarks(
        mp_detected_frame,
        body_landmark_0_to_16,
        shoulder_center_point,
        shoulder_width
    )

    flatten_landmarks = flatten_normalized_landmarks(
        normalized_0_to_16_body_landmarks,
        normalized_right_hand_landmarks,
        normalized_left_hand_landmarks
    )

    original_frame = draw_custom_landmarks(original_frame, mp_detected_frame)

    return flatten_landmarks, original_frame



def main():
    capture = opencv.VideoCapture(0)
    with holistic :
        while capture.isOpened():
            success, original_frame = capture.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            flatten_landmarks, original_frame = extract_landmarks_from_original_frame(original_frame)

            print(flatten_landmarks)

            opencv.imshow('MediaPipe Holistic', opencv.flip(original_frame, 1))
            if opencv.waitKey(5) & 0xFF == 27:
                break
    capture.release()
    opencv.destroyAllWindows()

if __name__ == "__main__":
    main()
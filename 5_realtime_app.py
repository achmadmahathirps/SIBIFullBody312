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
    model_complexity=0,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def main():
    capture = opencv.VideoCapture(1)
    with holistic :
        while capture.isOpened():
            success, original_frame = capture.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
        
            frame.flags.writeable = False
            frame = opencv.cvtColor(frame, opencv.COLOR_BGR2RGB)
            results = holistic.process(frame)

            frame.flags.writeable = True
            frame = opencv.cvtColor(frame, opencv.COLOR_RGB2BGR)

            # mediapipe_drawing.draw_landmarks(
            #     frame,
            #     results.pose_landmarks,
            #     mediapipe_holistic.POSE_CONNECTIONS,
            #     landmark_drawing_spec=mediapipe_drawing_styles.get_default_pose_landmarks_style()
            # )

            opencv.imshow('MediaPipe Holistic', opencv.flip(frame, 1))
            if opencv.waitKey(5) & 0xFF == 27:
                break
    capture.release()
    opencv.destroyAllWindows()

if __name__ == "__main__":
    main()
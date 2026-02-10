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
    
    body_landmark_0_to_16 = mp_detected_frame.pose_landmarks.landmark[:17]

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


def draw_normalized_stickman(normalized_custom_pose,
                             normalized_right_hand,
                             normalized_left_hand,
                             canvas_size=(500, 500),
                             scale=100,

                             # axes / grid / ticks
                             show_center_cross=True,
                             show_grid=True,
                             grid_interval=0.5,     # in normalized units
                             show_ticks=True,
                             tick_interval=0.5,     # in normalized units
                             tick_size_pixels=5,

                             # pose colors
                             pose_dot_color=(255, 255, 255),
                             pose_line_color=(255, 255, 255),
                             pose_text_color=(255, 255, 255),

                             # right hand colors (dots/lines)
                             right_dot_color=(200, 200, 200),
                             right_line_color=(200, 200, 200),

                             # left hand colors (dots/lines)
                             left_dot_color=(200, 200, 200),
                             left_line_color=(200, 200, 200),

                             # hand label colors (near dots + bottom lists)
                             right_label_color=(0, 255, 0),      # green
                             left_label_color=(0, 165, 255),     # orange-ish

                             # overlay info
                             sequence_count=None,
                             frame_index=None,
                             label=None,
                             fps=None,
                             overlay_color=(0, 255, 0),
                             overlay_font_scale=0.6,

                             # pose text controls
                             show_pose_coords=True,
                             show_pose_numbers=True,

                             # hand list controls (bottom)
                             show_hand_lists=True,
                             hand_font_scale=0.25,

                             # hand labels near dots
                             show_hand_labels=True,
                             hand_label_font_scale=0.25):
    """
    Stickman visualization with:
      - Pose dots + P{i} labels + coords near pose points
      - Hand dots + RH_i/LH_i labels near hand points (separate colors)
      - Bottom-left RH_i (x,y), bottom-right (x,y) LH_i (skip zero points)
      - Overlay: label, sequence, frame, FPS
      - Axes: X red horizontal, Y green vertical
      - Tick marks and optional faint grid in normalized space
    """

    h, w = canvas_size
    stickman_frame = opencv.UMat(h, w, opencv.CV_8UC3).get()
    stickman_frame[:] = 0

    center_x, center_y = w // 2, h // 2

    # ---- overlay info block (top-left) ----
    overlay_lines = []
    if label is not None:
        overlay_lines.append(f"Label: {label}")
    if sequence_count is not None and frame_index is not None:
        overlay_lines.append(f"Seq: {sequence_count:03} | Frame: {frame_index:03}")
    elif sequence_count is not None:
        overlay_lines.append(f"Seq: {sequence_count:03}")
    elif frame_index is not None:
        overlay_lines.append(f"Frame: {frame_index:03}")
    if fps is not None:
        overlay_lines.append(f"FPS: {fps:.1f}")

    for i, text in enumerate(overlay_lines):
        opencv.putText(
            stickman_frame,
            text,
            (10, 25 + i * 22),
            opencv.FONT_HERSHEY_SIMPLEX,
            overlay_font_scale,
            overlay_color,
            2,
            opencv.LINE_AA
        )

    # ---- axes / grid / ticks at (0,0) ----
    if show_center_cross:
        x_color = (0, 0, 255)     # red (horizontal axis)
        y_color = (0, 255, 0)     # green (vertical axis)
        grid_color = (60, 60, 60) # faint dark gray grid

        # convert normalized interval -> pixels, clamp to >= 1px
        grid_step_px = max(1, int(scale * grid_interval)) if grid_interval else None
        tick_step_px = max(1, int(scale * tick_interval)) if tick_interval else None

        # ---- faint grid (optional) ----
        if show_grid and grid_step_px is not None:
            # vertical lines
            x = center_x
            while x < w:
                opencv.line(stickman_frame, (x, 0), (x, h), grid_color, 1)
                x += grid_step_px
            x = center_x
            while x > 0:
                opencv.line(stickman_frame, (x, 0), (x, h), grid_color, 1)
                x -= grid_step_px

            # horizontal lines
            y = center_y
            while y < h:
                opencv.line(stickman_frame, (0, y), (w, y), grid_color, 1)
                y += grid_step_px
            y = center_y
            while y > 0:
                opencv.line(stickman_frame, (0, y), (w, y), grid_color, 1)
                y -= grid_step_px

        # ---- main axes ----
        opencv.line(stickman_frame, (0, center_y), (w, center_y), x_color, 1)  # X axis
        opencv.line(stickman_frame, (center_x, 0), (center_x, h), y_color, 1)  # Y axis

        # ---- tick marks (optional) ----
        if show_ticks and tick_step_px is not None:
            t = tick_size_pixels

            # X-axis ticks
            x = center_x
            while x < w:
                opencv.line(stickman_frame, (x, center_y - t), (x, center_y + t), x_color, 2)
                x += tick_step_px
            x = center_x
            while x > 0:
                opencv.line(stickman_frame, (x, center_y - t), (x, center_y + t), x_color, 2)
                x -= tick_step_px

            # Y-axis ticks
            y = center_y
            while y < h:
                opencv.line(stickman_frame, (center_x - t, y), (center_x + t, y), y_color, 2)
                y += tick_step_px
            y = center_y
            while y > 0:
                opencv.line(stickman_frame, (center_x - t, y), (center_x + t, y), y_color, 2)
                y -= tick_step_px

        # ---- axis labels ----
        axis_font_scale = 0.4
        axis_thickness = 1

        # X labels
        opencv.putText(stickman_frame, "-X", (10, center_y - 5),
                       opencv.FONT_HERSHEY_SIMPLEX, axis_font_scale, x_color, axis_thickness, opencv.LINE_AA)
        opencv.putText(stickman_frame, "+X", (w - 40, center_y - 5),
                       opencv.FONT_HERSHEY_SIMPLEX, axis_font_scale, x_color, axis_thickness, opencv.LINE_AA)

        # Y labels (up is -Y, down is +Y)
        opencv.putText(stickman_frame, "-Y", (center_x + 5, 20),
                       opencv.FONT_HERSHEY_SIMPLEX, axis_font_scale, y_color, axis_thickness, opencv.LINE_AA)
        opencv.putText(stickman_frame, "+Y", (center_x + 5, h - 10),
                       opencv.FONT_HERSHEY_SIMPLEX, axis_font_scale, y_color, axis_thickness, opencv.LINE_AA)

        # center label
        opencv.putText(stickman_frame, "(0,0)",
                       (center_x + 6, center_y - 6),
                       opencv.FONT_HERSHEY_SIMPLEX,
                       0.45, (255, 255, 255), 1, opencv.LINE_AA)

    # mapping normalized -> pixel
    def to_pixel(pt):
        x, y = pt
        return int(center_x + x * scale), int(center_y + y * scale)

    # ---- pose text near dots ----
    def draw_pose_text(px, py, idx, x_norm, y_norm):
        lines = []
        if show_pose_numbers:
            lines.append(f"P{idx}")
        if show_pose_coords:
            lines.append(f"({x_norm:.2f},{y_norm:.2f})")

        for j, text in enumerate(lines):
            opencv.putText(
                stickman_frame,
                text,
                (px + 4, py + 4 + j * 12),
                opencv.FONT_HERSHEY_SIMPLEX,
                0.35,
                pose_text_color,
                1,
                opencv.LINE_AA
            )

    # ---- hand label near dots ----
    def draw_hand_label(px, py, text, color):
        opencv.putText(
            stickman_frame,
            text,
            (px + 3, py + 3),
            opencv.FONT_HERSHEY_SIMPLEX,
            hand_label_font_scale,
            color,
            1,
            opencv.LINE_AA
        )

    # ---------------- POSE ----------------
    pose_pixels = [to_pixel(pt) for pt in normalized_custom_pose]

    for i, (px, py) in enumerate(pose_pixels):
        x_norm, y_norm = normalized_custom_pose[i]
        opencv.circle(stickman_frame, (px, py), 3, pose_dot_color, -1)
        draw_pose_text(px, py, i, x_norm, y_norm)

    for a, b in mediapipe_holistic.POSE_CONNECTIONS:
        if a < 17 and b < 17:
            opencv.line(stickman_frame, pose_pixels[a], pose_pixels[b], pose_line_color, 2)

    # ---------------- RIGHT HAND (dots + labels) ----------------
    right_pixels = [to_pixel(pt) for pt in normalized_right_hand]

    for i, (px, py) in enumerate(right_pixels):
        opencv.circle(stickman_frame, (px, py), 2, right_dot_color, -1)
        if show_hand_labels:
            draw_hand_label(px, py, f"RH_{i}", right_label_color)

    for a, b in mediapipe_holistic.HAND_CONNECTIONS:
        opencv.line(stickman_frame, right_pixels[a], right_pixels[b], right_line_color, 1)

    # ---------------- LEFT HAND (dots + labels) ----------------
    left_pixels = [to_pixel(pt) for pt in normalized_left_hand]

    for i, (px, py) in enumerate(left_pixels):
        opencv.circle(stickman_frame, (px, py), 2, left_dot_color, -1)
        if show_hand_labels:
            draw_hand_label(px, py, f"LH_{i}", left_label_color)

    for a, b in mediapipe_holistic.HAND_CONNECTIONS:
        opencv.line(stickman_frame, left_pixels[a], left_pixels[b], left_line_color, 1)

    # ---------------- BOTTOM HAND LISTS (skip (0,0)) ----------------
    if show_hand_lists:
        epsilon = 1e-6
        bottom_margin = 10
        line_height = 12
        start_y = h - bottom_margin - (21 * line_height)

        # Right-hand list bottom-left
        x_left = 10
        right_line_index = 0
        for i in range(21):
            x_norm, y_norm = normalized_right_hand[i]
            if abs(x_norm) < epsilon and abs(y_norm) < epsilon:
                continue

            text = f"RH_{i} ({x_norm:.2f},{y_norm:.2f})"
            y_pos = start_y + right_line_index * line_height
            right_line_index += 1

            opencv.putText(
                stickman_frame,
                text,
                (x_left, y_pos),
                opencv.FONT_HERSHEY_SIMPLEX,
                hand_font_scale,
                right_label_color,
                1,
                opencv.LINE_AA
            )

        # Left-hand list bottom-right (right aligned)
        x_right = w - 10
        left_line_index = 0
        for i in range(21):
            x_norm, y_norm = normalized_left_hand[i]
            if abs(x_norm) < epsilon and abs(y_norm) < epsilon:
                continue

            text = f"({x_norm:.2f},{y_norm:.2f}) LH_{i}"
            y_pos = start_y + left_line_index * line_height
            left_line_index += 1

            (text_w, _), _ = opencv.getTextSize(
                text, opencv.FONT_HERSHEY_SIMPLEX, hand_font_scale, 1
            )
            opencv.putText(
                stickman_frame,
                text,
                (x_right - text_w, y_pos),
                opencv.FONT_HERSHEY_SIMPLEX,
                hand_font_scale,
                left_label_color,
                1,
                opencv.LINE_AA
            )

    return stickman_frame


def extract_landmarks_from_original_frame(original_frame, fps=None):
    mp_detected_frame, original_frame = process_frame_with_holistic(original_frame)

    body_landmark_0_to_16, shoulder_center_point, shoulder_width = get_body_reference_points(mp_detected_frame)

    if body_landmark_0_to_16 is None:
        return None, original_frame, None
    
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

    stickman_frame = draw_normalized_stickman(
        normalized_custom_pose=normalized_0_to_16_body_landmarks,
        normalized_right_hand=normalized_right_hand_landmarks,
        normalized_left_hand=normalized_left_hand_landmarks,
        fps=fps
    )

    return flatten_landmarks, original_frame, stickman_frame



def main():
    capture = opencv.VideoCapture(0)

    previous_time = time.time()

    with holistic :
        while capture.isOpened():
            success, original_frame = capture.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            now = time.time()
            dt = now - previous_time
            previous_time = now
            fps = (1.0 / dt) if dt > 0 else None

            flatten_landmarks, original_frame, stickman_frame = extract_landmarks_from_original_frame(original_frame, fps=fps)

            opencv.imshow('MediaPipe Holistic', opencv.flip(original_frame, 1))

            if stickman_frame is not None:
                opencv.imshow('Stickman Frame', stickman_frame)

            if flatten_landmarks is not None:
                print(flatten_landmarks)

            print(flatten_landmarks)

            
            opencv.imshow('Stickman Frame', stickman_frame)

            if opencv.waitKey(5) & 0xFF == 27:
                break
    capture.release()
    opencv.destroyAllWindows()

if __name__ == "__main__":
    main()
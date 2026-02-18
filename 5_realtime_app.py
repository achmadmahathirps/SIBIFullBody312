import time
import pandas
import numpy as np
import cv2 as opencv
from collections import deque
from pathlib import Path

import mediapipe
from mediapipe.framework.formats import landmark_pb2

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, X, lengths):
        out, _ = self.lstm(X)  # out: [B, T, 2*hidden]
        B, T, D = out.shape

        # mask padding positions so mean pooling uses only valid frames
        mask = torch.arange(T, device=out.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_f = mask.float().unsqueeze(-1)

        summed = (out * mask_f).sum(dim=1)
        denom = lengths.clamp(min=1).unsqueeze(1).float()
        feat = summed / denom  # mean over valid timesteps

        logits = self.classifier(feat)
        return logits
    
    
class LSTMRealtimeInferencer:
    """
    Maintains a rolling buffer of per-frame features and runs BiLSTM inference.
    """
    def __init__(self, ckpt_path: str, device: str | None = None, min_frames: int = 10):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.pad_len = int(ckpt["pad_len"])
        self.n_features = int(ckpt["n_features"])
        self.id_to_label = ckpt["id_to_label"]

        hidden_size = int(ckpt["hidden_size"])
        num_layers = int(ckpt["num_layers"])
        dropout = float(ckpt["dropout"])
        num_classes = len(self.id_to_label)

        self.model = BiLSTMClassifier(
            input_size=self.n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        ).to(self.device)

        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.buffer = deque(maxlen=self.pad_len)
        self.min_frames = min_frames

        # optional smoothing (EMA over probabilities)
        self.ema_probs = None
        self.ema_alpha = 0.25  # 0..1 higher = less smoothing

    def push_frame(self, feature_118: list[float] | np.ndarray):
        arr = np.asarray(feature_118, dtype=np.float32)
        if arr.shape[0] != self.n_features:
            raise ValueError(f"Expected feature dim {self.n_features}, got {arr.shape[0]}")
        self.buffer.append(arr)

    @torch.no_grad()
    def predict(self):
        """
        Returns (label, confidence, probs) or (None, 0.0, None) if not enough frames yet.
        """
        length = len(self.buffer)
        if length < self.min_frames:
            return None, 0.0, None

        # build [1, pad_len, n_features]
        X = np.zeros((self.pad_len, self.n_features), dtype=np.float32)
        X[:length, :] = np.stack(list(self.buffer), axis=0)

        X_t = torch.from_numpy(X).unsqueeze(0).to(self.device)          # [1, 83, 118]
        lengths_t = torch.tensor([length], dtype=torch.long).to(self.device)

        logits = self.model(X_t, lengths_t)                             # [1, num_classes]
        probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

        # EMA smoothing to reduce flicker
        if self.ema_probs is None:
            self.ema_probs = probs
        else:
            self.ema_probs = self.ema_alpha * probs + (1 - self.ema_alpha) * self.ema_probs

        pred_id = int(np.argmax(self.ema_probs))
        conf = float(self.ema_probs[pred_id])
        label = self.id_to_label[pred_id]
        return label, conf, self.ema_probs


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
    

# Normalize pose and hands using shoulder-based coordinates
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


# Get pose landmarks and shoulder reference points
def get_pose_reference_points(detection_results):
    if not detection_results.pose_landmarks:
        return None, None, None

    pose_landmark_1_to_16 = detection_results.pose_landmarks.landmark[:17]

    left_shoulder = detection_results.pose_landmarks.landmark[
        mediapipe_holistic.PoseLandmark.LEFT_SHOULDER
    ]
    right_shoulder = detection_results.pose_landmarks.landmark[
        mediapipe_holistic.PoseLandmark.RIGHT_SHOULDER
    ]

    shoulder_center_point = (
        (left_shoulder.x + right_shoulder.x) / 2,
        (left_shoulder.y + right_shoulder.y) / 2
    )

    shoulder_width = euclidean_distance(left_shoulder, right_shoulder)

    return pose_landmark_1_to_16, shoulder_center_point, shoulder_width


def draw_pose_and_hands_on_frame(output_frame, detection_results):

    # Filter pose landmarks to 0–16 (upper body)
    filtered_pose = landmark_pb2.NormalizedLandmarkList(
        landmark=[detection_results.pose_landmarks.landmark[i] for i in range(17)]
    )

    filtered_connections = [
        connection
        for connection in mediapipe_holistic.POSE_CONNECTIONS
        if connection[0] < 17 and connection[1] < 17
    ]

    # Draw pose
    mediapipe_drawing.draw_landmarks(
        output_frame,
        filtered_pose,
        filtered_connections,
        landmark_drawing_spec=mediapipe_drawing_styles.get_default_pose_landmarks_style()
    )

    # Draw right hand
    if detection_results.right_hand_landmarks:
        mediapipe_drawing.draw_landmarks(
            output_frame,
            detection_results.right_hand_landmarks,
            mediapipe_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mediapipe_drawing_styles.get_default_hand_landmarks_style()
        )

    # Draw left hand
    if detection_results.left_hand_landmarks:
        mediapipe_drawing.draw_landmarks(
            output_frame,
            detection_results.left_hand_landmarks,
            mediapipe_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mediapipe_drawing_styles.get_default_hand_landmarks_style()
        )

    return output_frame


def draw_normalized_stickman(normalized_custom_pose,
                             normalized_right_hand,
                             normalized_left_hand):
    """
    Stickman visualization with:
      - Pose dots + P{i} labels + coords near pose points
      - Hand dots + RH_i/LH_i labels near hand points (separate colors)
      - Bottom-left RH_i (x,y), bottom-right (x,y) LH_i (skip zero points)
      - Overlay: label, sequence, frame, FPS
      - Axes: X red horizontal, Y green vertical
      - Tick marks and optional faint grid in normalized space
    """
    
    canvas_size=(550, 550)
    scale=100

    # axes / grid / ticks
    show_center_cross=True
    show_grid=True
    grid_interval=0.5     # in normalized units
    show_ticks=True
    tick_interval=1.0     # in normalized units
    tick_size_pixels=5

    # pose colors
    pose_dot_color=(255, 255, 255)
    pose_line_color=(255, 255, 255)
    pose_text_color=(255, 255, 255)

    # right hand colors (dots/lines)
    right_dot_color=(200, 200, 200)
    right_line_color=(200, 200, 200)

    # left hand colors (dots/lines)
    left_dot_color=(200, 200, 200)
    left_line_color=(200, 200, 200)

    # hand label colors (near dots + bottom lists)
    right_label_color=(0, 255, 0)      # green
    left_label_color=(0, 165, 255)     # orange-ish

    # pose text controls
    show_pose_coords=True
    show_pose_numbers=True

    # hand list controls (bottom)
    show_hand_lists=True
    hand_font_scale=0.25

    # hand labels near dots
    show_hand_labels=True
    hand_label_font_scale=0.25
    
    h, w = canvas_size
    stickman_frame = opencv.UMat(h, w, opencv.CV_8UC3).get()
    stickman_frame[:] = 0

    center_x, center_y = w // 2, h // 2

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


def main():
    capture = opencv.VideoCapture(1)
    if not capture.isOpened():
        print("[ERROR] Cannot open webcam (VideoCapture(0) failed).")
        return
    
    with mediapipe_holistic.Holistic(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        infer = LSTMRealtimeInferencer("trained_model/best_lstm.pt", min_frames=10)
        
        while capture.isOpened():
            success, frame = capture.read()
            frame = opencv.flip(frame, 1)  # mirror horizontally for natural webcam view
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            # Mediapipe Holistic processes
            frame.flags.writeable = False
            frame = opencv.cvtColor(frame, opencv.COLOR_BGR2RGB)
            results = holistic.process(frame)
            frame.flags.writeable = True
            frame = opencv.cvtColor(frame, opencv.COLOR_RGB2BGR)
            
            # Get pose reference points
            pose_landmark_1_to_16, shoulder_center_point, shoulder_width = get_pose_reference_points(results)
            
            if pose_landmark_1_to_16 is None:
                opencv.imshow('MediaPipe Holistic', opencv.flip(frame, 1))
                if opencv.waitKey(5) & 0xFF == 27:
                    break
                continue
            
            normalized_custom_pose, normalized_right_hand, normalized_left_hand = get_normalized_pose_and_hands(
                results,
                pose_landmark_1_to_16,
                shoulder_center_point,
                shoulder_width
            )
            
            extracted_frame_landmarks = flatten_normalized_landmarks(
                normalized_custom_pose,
                normalized_right_hand,
                normalized_left_hand
            )
            
            # push current frame feature into sequence buffer
            infer.push_frame(extracted_frame_landmarks)

            # run prediction (you can also do this every N frames to reduce CPU load)
            label, conf, _ = infer.predict()

            if label is not None:
                # draw prediction
                opencv.putText(
                    frame,
                    f"{label} ({conf*100:.1f}%)",
                    (10, 40),
                    opencv.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    opencv.LINE_AA
                )
            
            frame = draw_pose_and_hands_on_frame(frame, results)
            
            # Stickman with overlays
            stickman_frame = draw_normalized_stickman(
                normalized_custom_pose,
                normalized_right_hand,
                normalized_left_hand,
            )
            
            opencv.imshow('MediaPipe Holistic', frame)
            opencv.imshow('Stickman Frame', stickman_frame)
            if opencv.waitKey(5) & 0xFF == 27:
                break
            
        capture.release()
        opencv.destroyAllWindows()

if __name__ == "__main__":
    main()

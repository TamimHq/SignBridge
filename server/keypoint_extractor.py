"""
Keypoint extraction for the server - MediaPipe Tasks API version (0.10.20+).

The legacy mp.solutions.holistic API was removed in newer MediaPipe. This uses
the modern Tasks API (PoseLandmarker + HandLandmarker) and combines them into
the SAME 144-dim format the model trained on:
    6 upper-body pose joints (shoulders, elbows, wrists) x 3 = 18
    21 left-hand landmarks  x 3 = 63
    21 right-hand landmarks x 3 = 63
    total = 144

Model bundle files are downloaded automatically on first run.
"""

import os
import urllib.request
import numpy as np

POSE_UPPER = [11, 12, 13, 14, 15, 16]  # shoulders, elbows, wrists
N_HAND = 21
TARGET_FRAMES = 30
FEATURE_DIM = len(POSE_UPPER) * 3 + N_HAND * 3 + N_HAND * 3  # 144

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS_CACHE = os.path.join(_HERE, "mp_models")

_POSE_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
_HAND_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)




def _download_if_needed(url: str, filename: str) -> str:
    os.makedirs(_MODELS_CACHE, exist_ok=True)
    path = os.path.join(_MODELS_CACHE, filename)
    if not os.path.exists(path):
        print(f"[Extractor] Downloading {filename} (first run only)...")
        urllib.request.urlretrieve(url, path)
        print(f"[Extractor] Saved {filename}")
    return path


def _build_landmarkers():
    """Create a fresh pair of landmarkers. VIDEO mode requires timestamps to be
    strictly increasing per instance, so we build new ones for each video."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    pose_path = _download_if_needed(_POSE_TASK_URL, "pose_landmarker_lite.task")
    hand_path = _download_if_needed(_HAND_TASK_URL, "hand_landmarker.task")

    pose_opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=pose_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=hand_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return (
        vision.PoseLandmarker.create_from_options(pose_opts),
        vision.HandLandmarker.create_from_options(hand_opts),
    )



def _frame_vector(pose_result, hand_result) -> np.ndarray:
    """Combine pose + hand results into the 144-dim vector, matching training."""
    vec = []

    if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
        lms = pose_result.pose_landmarks[0]
        for idx in POSE_UPPER:
            lm = lms[idx]
            vec.extend([lm.x, lm.y, lm.visibility])
    else:
        vec.extend([0.0] * (len(POSE_UPPER) * 3))

    left = None
    right = None
    if hand_result.hand_landmarks:
        for i, hand_lms in enumerate(hand_result.hand_landmarks):
            label = hand_result.handedness[i][0].category_name
            if label == "Left" and left is None:
                left = hand_lms
            elif label == "Right" and right is None:
                right = hand_lms

    for hand in (left, right):
        if hand is not None:
            for lm in hand:
                vec.extend([lm.x, lm.y, 1.0])
        else:
            vec.extend([0.0] * (N_HAND * 3))

    return np.array(vec, dtype=np.float32)


def resample_frames(frames, target=TARGET_FRAMES):
    if len(frames) == 0:
        return np.zeros((target, FEATURE_DIM), dtype=np.float32)
    frames = np.array(frames, dtype=np.float32)
    if len(frames) == target:
        return frames
    idx = np.linspace(0, len(frames) - 1, target).astype(int)
    return frames[idx]


def normalize_keypoints(kps):
    """Center on mid-shoulder, scale by shoulder width. Matches training exactly."""
    kps = kps.copy()
    l_sh = kps[:, 0:2]
    r_sh = kps[:, 3:5]
    mid = (l_sh + r_sh) / 2
    width = np.linalg.norm(l_sh - r_sh, axis=1, keepdims=True)
    width = np.where(width < 1e-6, 1.0, width)
    for i in range(FEATURE_DIM // 3):
        xi, yi = i * 3, i * 3 + 1
        kps[:, xi] = (kps[:, xi] - mid[:, 0]) / width[:, 0]
        kps[:, yi] = (kps[:, yi] - mid[:, 1]) / width[:, 0]
    return kps


def extract_from_video(video_path: str) -> np.ndarray:
    """Run MediaPipe Tasks over a video -> normalized (30, 144) array."""
    import cv2
    import mediapipe as mp

    pose_landmarker, hand_landmarker = _build_landmarkers()

    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        frame_idx = 0

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((frame_idx / fps) * 1000)

            pose_result = pose_landmarker.detect_for_video(mp_image, ts_ms)
            hand_result = hand_landmarker.detect_for_video(mp_image, ts_ms)

            frames.append(_frame_vector(pose_result, hand_result))
            frame_idx += 1

        cap.release()
    finally:
        pose_landmarker.close()
        hand_landmarker.close()

    if not frames:
        raise ValueError("No frames could be read from the video")

    return normalize_keypoints(resample_frames(frames))
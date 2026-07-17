"""
Keypoint extraction for the server — MUST match the training pipeline exactly.
Reuses the same 144-dim feature layout, resampling, and normalization used in
the Colab training notebook, so inference keypoints match what the model learned.
"""

import numpy as np

# Lazy MediaPipe import so the server can start even if mediapipe isn't installed
_mp_holistic = None
_holistic = None

POSE_UPPER = [11, 12, 13, 14, 15, 16]  # shoulders, elbows, wrists
N_HAND = 21
TARGET_FRAMES = 30
FEATURE_DIM = len(POSE_UPPER) * 3 + N_HAND * 3 + N_HAND * 3  # 144


def _ensure_mediapipe():
    global _mp_holistic, _holistic
    if _holistic is None:
        import mediapipe as mp
        _mp_holistic = mp.solutions.holistic
        _holistic = _mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,   # complexity 1 for inference accuracy
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _holistic


def extract_frame_keypoints(results):
    """One MediaPipe Holistic result → 144-dim vector."""
    vec = []
    if results.pose_landmarks:
        lms = results.pose_landmarks.landmark
        for idx in POSE_UPPER:
            lm = lms[idx]
            vec.extend([lm.x, lm.y, lm.visibility])
    else:
        vec.extend([0.0] * (len(POSE_UPPER) * 3))

    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            vec.extend([lm.x, lm.y, 1.0])
    else:
        vec.extend([0.0] * (N_HAND * 3))

    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
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
    """Center on mid-shoulder, scale by shoulder width. Matches training."""
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
    """Run MediaPipe over a video file → normalized (30, 144) array."""
    import cv2
    holistic = _ensure_mediapipe()

    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        frames.append(extract_frame_keypoints(results))
    cap.release()

    if not frames:
        raise ValueError("No frames could be read from the video")

    resampled = resample_frames(frames)
    return normalize_keypoints(resampled)
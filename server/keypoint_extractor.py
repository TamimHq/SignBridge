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
    """Create a fresh pair of landmarkers.

    VIDEO running mode requires strictly increasing timestamps per instance,
    so we build new ones for each video rather than reusing globals.
    """
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



def _transcode_to_mp4(src_path: str) -> str:
    """Convert any container (webm from browsers, mov from iOS) to H.264 mp4.

    Uses the ffmpeg binary bundled with imageio-ffmpeg, so nothing needs to be
    installed system-wide. Returns the path of the converted file.
    """
    import subprocess
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    dst_path = src_path + ".converted.mp4"
    subprocess.run(
        [
            ffmpeg, "-y", "-loglevel", "error",
            "-i", src_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
            dst_path,
        ],
        check=True,
    )
    return dst_path


def _read_frames(video_path: str):
    """Read all frames of a video as BGR numpy arrays via OpenCV."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = []
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        out.append(frame)
    cap.release()
    return out, fps

def extract_from_video(video_path: str) -> np.ndarray:
    """Run MediaPipe Tasks over a video -> normalized (30, 144) array.

    Accepts mp4 (Android), mov (iOS) and webm (browsers). If OpenCV cannot
    decode the container, the file is transcoded to mp4 and retried.
    """
    import os
    import cv2
    import mediapipe as mp

    raw_frames, fps = _read_frames(video_path)

    converted_path = None
    if not raw_frames:
        # Likely webm or another container OpenCV can't decode - transcode
        try:
            converted_path = _transcode_to_mp4(video_path)
            raw_frames, fps = _read_frames(converted_path)
        except Exception as e:
            raise ValueError(
                f"Could not decode video, and transcoding failed: {e}"
            )

    if not raw_frames:
        raise ValueError("No frames could be read from the video")

    pose_landmarker, hand_landmarker = _build_landmarkers()
    try:
        frames = []
        for i, frame in enumerate(raw_frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((i / fps) * 1000)
            pose_result = pose_landmarker.detect_for_video(mp_image, ts_ms)
            hand_result = hand_landmarker.detect_for_video(mp_image, ts_ms)
            frames.append(_frame_vector(pose_result, hand_result))
    finally:
        pose_landmarker.close()
        hand_landmarker.close()
        if converted_path and os.path.exists(converted_path):
            os.unlink(converted_path)

    # ── Diagnostics: how much did MediaPipe actually see? ──
    arr = np.array(frames, dtype=np.float32)
    pose_ok = float(np.mean(np.any(arr[:, 0:18] != 0, axis=1)))
    lh_ok = float(np.mean(np.any(arr[:, 18:81] != 0, axis=1)))
    rh_ok = float(np.mean(np.any(arr[:, 81:144] != 0, axis=1)))
    print(
        f"[Extractor] frames={len(frames)} "
        f"pose={pose_ok*100:.0f}% leftHand={lh_ok*100:.0f}% rightHand={rh_ok*100:.0f}%"
    )
    if lh_ok < 0.2 and rh_ok < 0.2:
        print("[Extractor] WARNING: hands barely detected - check framing/lighting")

    return normalize_keypoints(resample_frames(frames))
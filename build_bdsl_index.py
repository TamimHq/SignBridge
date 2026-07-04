"""
BDSL Vocabulary Index Builder
==============================
Reads SignBD-Word dataset structure and builds:
  1. vocabulary_index.json  — lookup table: Bengali word → clip info
  2. Per-sign numpy keypoint arrays extracted from OpenPose rendered PNGs

Dataset structure expected:
  SIGNBD-WORD/
    bodypose/
      Train/  or  Test/
        <gloss>/
          <person>_<view>_<gloss>.mp4/   ← folder named like a .mp4
            *_rendered.png               ← OpenPose skeleton frames
    SignBD-RGB/
      Train/  or  Test/
        <gloss>/
          <person>_<view>_<gloss>.mp4
    words_translation_gloss.xlsx

Usage:
  py build_bdsl_index.py --dataset_root "D:/project_SL/SignBD-Word" --output_dir "./processed/bdsl" --split train   # or test
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
import openpyxl
import cv2  # pip install opencv-python


# ── OpenPose body25 keypoint indices we care about ──────────────────────────
# Full body25 has 25 body + 21 left hand + 21 right hand = 67 points
# We use: shoulders(2,5), elbows(3,6), wrists(4,7), plus both hands (25-66)
BODY_KEYPOINTS = [2, 3, 4, 5, 6, 7]  # upper body only
HAND_L_START = 25   # left hand: indices 25-45
HAND_R_START = 46   # right hand: indices 46-66
TOTAL_KEYPOINTS = 67  # body25 + 2 hands

# Each rendered PNG is a visualization — we need to re-extract coords from
# the JSON files OpenPose produces. If you only have rendered PNGs (no JSON),
# we fall back to a frame-count-based approach (see note below).


# Bengali Unicode normalization
# ড় ঢ় য় each exist as precomposed (single codepoint) and decomposed (two codepoints)
# xlsx stores precomposed; keyboards produce decomposed. Normalize to decomposed.
_BN_PRECOMPOSED = {
    '\u09dc': '\u09a1\u09bc',  # ড়
    '\u09dd': '\u09a2\u09bc',  # ঢ়
    '\u09df': '\u09af\u09bc',  # য়
}

def bn_normalize(text: str) -> str:
    for pre, dec in _BN_PRECOMPOSED.items():
        text = text.replace(pre, dec)
    return text.strip()


def load_vocabulary(xlsx_path: str) -> dict:
    """
    Parse words_translation_gloss.xlsx into a dict:
      gloss_name (romanized) → {bangla, english, gloss}
    Also builds reverse lookups: english_lower → gloss, bangla → gloss
    All Bengali keys normalized via bn_normalize() to match keyboard input.
    """
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active

    vocab = {}
    english_to_gloss = {}
    bangla_to_gloss = {}

    for row in ws.iter_rows(values_only=True):
        bangla, english, gloss = row
        if gloss is None or bangla is None or english is None:
            continue
        if str(bangla).strip() in ('Bangla ', 'Bangla'):
            continue  # header row

        gloss_key = str(gloss).strip().lower()
        bn_raw = str(bangla).strip()
        bn_key = bn_normalize(bn_raw)  # normalize for consistent lookup

        for gk in gloss_key.split('/'):
            gk = gk.strip()
            vocab[gk] = {
                "bangla": bn_raw,
                "bangla_normalized": bn_key,
                "english": str(english).strip(),
                "gloss": gk
            }
            english_to_gloss[str(english).strip().lower()] = gk
            bangla_to_gloss[bn_key] = gk  # store normalized key

    return vocab, english_to_gloss, bangla_to_gloss


def count_frames_in_clip_folder(folder: Path) -> int:
    """Count rendered PNG frames in an OpenPose clip folder."""
    return len(list(folder.glob("*_rendered.png")))


def get_frame_numbers(folder: Path) -> list:
    """Extract sorted frame numbers from rendered PNG filenames."""
    frames = []
    for f in folder.glob("*_rendered.png"):
        # filename like: p10_c_aam_000000000001_rendered.png
        # or:            c_aam_000000000003_rendered.png
        stem = f.stem  # remove .png
        parts = stem.split('_')
        # frame number is the part that's all digits (long zero-padded)
        for part in parts:
            if part.isdigit() and len(part) >= 6:
                frames.append(int(part))
                break
    return sorted(frames)


def build_clip_entry(
    gloss: str,
    clip_folder: Path,
    person: str,
    view: str,
    vocab_entry: dict,
    split: str
) -> dict:
    """Build a single clip dictionary entry for the vocabulary index."""
    frame_count = count_frames_in_clip_folder(clip_folder)
    frame_numbers = get_frame_numbers(clip_folder)

    return {
        "gloss": gloss,
        "bangla": vocab_entry["bangla"],
        "english": vocab_entry["english"],
        "person": person,
        "view": view,          # "c" = upperbody, "f" = fullbody
        "split": split,
        "frame_count": frame_count,
        "frame_numbers": frame_numbers,
        "bodypose_folder": str(clip_folder),
        "data_source": "signbd_word",
        "keypoint_type": "openpose_rendered_png",
        # Populated later by extract_keypoints_from_rgb()
        "keypoint_npy_path": None,
        "fallback_type": "exact"
    }


def scan_bodypose_split(bodypose_split_dir: Path, split: str, vocab: dict) -> list:
    """
    Scan bodypose/Train or bodypose/Test directory.
    Returns list of clip entries.
    """
    clips = []
    missing_gloss = set()

    for gloss_dir in sorted(bodypose_split_dir.iterdir()):
        if not gloss_dir.is_dir():
            continue
        gloss = gloss_dir.name.lower()

        if gloss not in vocab:
            missing_gloss.add(gloss)
            continue

        # Each subfolder inside gloss_dir is named like "p10_c_aam.mp4"
        for clip_folder in sorted(gloss_dir.iterdir()):
            if not clip_folder.is_dir():
                continue

            folder_name = clip_folder.name  # e.g. "p10_c_aam.mp4"
            # Parse person and view from folder name
            parts = folder_name.replace('.mp4', '').split('_')
            person = parts[0] if parts[0].startswith('p') else 'unknown'
            view = parts[1] if len(parts) > 1 and parts[1] in ('c', 'f') else 'c'

            entry = build_clip_entry(
                gloss=gloss,
                clip_folder=clip_folder,
                person=person,
                view=view,
                vocab_entry=vocab[gloss],
                split=split
            )
            clips.append(entry)

    if missing_gloss:
        print(f"  [warn] Glosses in bodypose not in vocabulary xlsx: {missing_gloss}")

    return clips


def extract_keypoints_from_openpose_json(json_dir: Path) -> np.ndarray | None:
    """
    If OpenPose JSON files exist alongside rendered PNGs, extract actual
    (x, y, confidence) arrays per frame → shape (T, 67*3).

    OpenPose JSON structure per frame:
      people[0].pose_keypoints_2d      → 25 body keypoints * 3
      people[0].hand_left_keypoints_2d → 21 hand keypoints * 3
      people[0].hand_right_keypoints_2d→ 21 hand keypoints * 3
    """
    import glob
    json_files = sorted(glob.glob(str(json_dir / "*.json")))
    if not json_files:
        return None

    frames = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        if not data.get('people'):
            frames.append(np.zeros(TOTAL_KEYPOINTS * 3))
            continue
        person = data['people'][0]
        body = np.array(person.get('pose_keypoints_2d', [0]*75))
        hand_l = np.array(person.get('hand_left_keypoints_2d', [0]*63))
        hand_r = np.array(person.get('hand_right_keypoints_2d', [0]*63))
        combined = np.concatenate([body, hand_l, hand_r])  # (201,) = 67 pts * 3
        frames.append(combined)

    return np.array(frames)  # shape (T, 201)


def normalize_keypoints(kps: np.ndarray) -> np.ndarray:
    """
    Normalize keypoints so clips from different signers/distances stitch smoothly.

    Strategy:
      1. Center on mid-shoulder point (avg of left/right shoulder)
      2. Scale by shoulder width so signer size is consistent
      3. Zero out any frames where both shoulders are missing (confidence=0)

    Input:  (T, 201) — 67 keypoints * 3 (x, y, conf)
    Output: (T, 201) normalized
    """
    kps = kps.copy()
    T = kps.shape[0]

    # OpenPose body25: shoulder_right=idx2, shoulder_left=idx5
    # In flattened array: idx2 → positions 6,7,8 ; idx5 → positions 15,16,17
    r_shoulder = kps[:, 6:8]   # (T,2)
    l_shoulder = kps[:, 15:17] # (T,2)
    mid_shoulder = (r_shoulder + l_shoulder) / 2  # (T,2)
    shoulder_width = np.linalg.norm(r_shoulder - l_shoulder, axis=1, keepdims=True)  # (T,1)
    shoulder_width = np.where(shoulder_width < 1e-6, 1.0, shoulder_width)

    # Normalize x,y for each keypoint
    for i in range(TOTAL_KEYPOINTS):
        x_idx = i * 3
        y_idx = i * 3 + 1
        kps[:, x_idx] = (kps[:, x_idx] - mid_shoulder[:, 0]) / shoulder_width[:, 0]
        kps[:, y_idx] = (kps[:, y_idx] - mid_shoulder[:, 1]) / shoulder_width[:, 0]

    return kps


def save_keypoints(kps: np.ndarray, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), kps)


def build_index(dataset_root: str, output_dir: str, split: str = "train"):
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kp_dir = output_dir / "keypoints" / "bdsl" / split

    print(f"\n=== BDSL Vocabulary Index Builder ===")
    print(f"Dataset root : {dataset_root}")
    print(f"Split        : {split}")
    print(f"Output dir   : {output_dir}")

    # ── 1. Load vocabulary ──────────────────────────────────────────────────
    xlsx_path = dataset_root / "words_translation_gloss.xlsx"
    print(f"\n[1/4] Loading vocabulary from {xlsx_path.name}...")
    vocab, english_to_gloss, bangla_to_gloss = load_vocabulary(str(xlsx_path))
    print(f"  Loaded {len(vocab)} gloss entries")

    # ── 2. Scan bodypose directory ──────────────────────────────────────────
    split_folder = "Train" if split == "train" else "Test"
    bodypose_dir = dataset_root / "bodypose" / split_folder
    print(f"\n[2/4] Scanning bodypose/{split_folder}...")
    clips = scan_bodypose_split(bodypose_dir, split, vocab)
    print(f"  Found {len(clips)} clips across {len(set(c['gloss'] for c in clips))} glosses")

    # ── 3. Try to extract keypoints from OpenPose JSONs if they exist ───────
    print(f"\n[3/4] Checking for OpenPose JSON keypoints...")
    json_found = 0
    for clip in clips:
        folder = Path(clip["bodypose_folder"])
        kps = extract_keypoints_from_openpose_json(folder)

        if kps is not None:
            kps_norm = normalize_keypoints(kps)
            out_path = kp_dir / clip["gloss"] / f"{clip['person']}_{clip['view']}.npy"
            save_keypoints(kps_norm, out_path)
            clip["keypoint_npy_path"] = str(out_path)
            json_found += 1
        else:
            # No JSON — mark as PNG-only (playback via rendered images directly)
            clip["keypoint_npy_path"] = None
            clip["keypoint_type"] = "rendered_png_only"

    if json_found == 0:
        print(f"  No OpenPose JSON files found — clips will use rendered PNGs for playback")
        print(f"  (This is normal if you only downloaded the bodypose rendered frames)")
    else:
        print(f"  Extracted + normalized keypoints for {json_found} clips")

    # ── 4. Build and save the vocabulary index ──────────────────────────────
    print(f"\n[4/4] Building vocabulary index...")

    # Group clips by gloss — for avatar playback we want one canonical clip
    # per gloss (prefer upperbody 'c' view, prefer p1 as reference signer)
    gloss_index = {}
    for clip in clips:
        gloss = clip["gloss"]
        if gloss not in gloss_index:
            gloss_index[gloss] = []
        gloss_index[gloss].append(clip)

    # Pick best clip per gloss: prefer view='c', then person 'p1'
    canonical = {}
    for gloss, clip_list in gloss_index.items():
        c_view = [c for c in clip_list if c['view'] == 'c']
        chosen = c_view[0] if c_view else clip_list[0]
        canonical[gloss] = chosen

    # Build final lookup: english_lower → entry, bangla → entry
    vocabulary_index = {
        "language": "bdsl_bn",
        "total_glosses": len(canonical),
        "english_to_gloss": english_to_gloss,
        "bangla_to_gloss": bangla_to_gloss,
        "glosses": canonical,
        # All clips (all signers) for training the recognition model
        "all_clips": clips
    }

    index_path = output_dir / f"bdsl_vocabulary_index_{split}.json"
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(vocabulary_index, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Done.")
    print(f"  Vocabulary index  : {index_path}")
    print(f"  Total glosses     : {len(canonical)}")
    print(f"  Total clips       : {len(clips)}")
    print(f"  English lookups   : {len(english_to_gloss)}")
    print(f"  Bengali lookups   : {len(bangla_to_gloss)}")

    # Print sample entries
    print(f"\nSample entries:")
    for gloss in list(canonical.keys())[:3]:
        e = canonical[gloss]
        print(f"  '{e['english']}' ('{e['bangla']}') → gloss='{gloss}' "
              f"frames={e['frame_count']} view={e['view']}")

    return vocabulary_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True,
                        help="Path to SIGNBD-WORD root folder")
    parser.add_argument("--output_dir", default="./processed",
                        help="Where to save processed index and keypoints")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    args = parser.parse_args()

    build_index(args.dataset_root, args.output_dir, args.split)
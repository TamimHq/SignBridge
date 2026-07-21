"""
Build an ASL training set from How2Sign OpenPose keypoints.

Unlike build_asl_index.py (which kept ONE canonical clip per word for the
avatar), this collects EVERY single-word clip so we have multiple samples per
class to train a classifier.

Output format matches the BdSL pipeline exactly so the same model architecture
and the same server extractor work:
    (30 frames, 144 features)
    144 = 6 upper-body joints x3  +  21 left-hand x3  +  21 right-hand x3

Joint order matches MediaPipe (what the server uses at inference):
    [L_shoulder, R_shoulder, L_elbow, R_elbow, L_wrist, R_wrist]

Usage:
  py build_asl_dataset.py --asl_root "E:/SL/ASL" --out "processed/asl_training" --min_clips 8
"""

import os
import csv
import json
import glob
import argparse
from collections import defaultdict

import numpy as np

# OpenPose BODY_25 indices, ordered to match MediaPipe's [11,12,13,14,15,16]
#   MediaPipe: L_sh, R_sh, L_el, R_el, L_wr, R_wr
#   OpenPose : 5=LSh 2=RSh 6=LEl 3=REl 7=LWr 4=RWr
OP_POSE_ORDER = [5, 2, 6, 3, 7, 4]

N_HAND = 21
TARGET_FRAMES = 30
FEATURE_DIM = len(OP_POSE_ORDER) * 3 + N_HAND * 3 + N_HAND * 3  # 144

SPLITS = ["train", "val", "test"]

# Words that are the SAME physical sign in ASL. Merging them prevents the model
# from being asked to distinguish gestures that are visually identical, and it
# consolidates their clips into one better-supported class.
SYNONYMS = {
    "hi": "hello",
    "hello": "hello",
    "ok": "okay",
    "okay": "okay",
}


# ── Keypoint helpers (must mirror server/keypoint_extractor.py) ──────────────

def resample_frames(frames, target=TARGET_FRAMES):
    if len(frames) == 0:
        return np.zeros((target, FEATURE_DIM), dtype=np.float32)
    frames = np.asarray(frames, dtype=np.float32)
    if len(frames) == target:
        return frames
    idx = np.linspace(0, len(frames) - 1, target).astype(int)
    return frames[idx]


def normalize_keypoints(kps):
    """Center on mid-shoulder, scale by shoulder width."""
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


def frame_vector_from_openpose(person):
    """One OpenPose 'people[0]' dict -> 144-dim vector."""
    vec = []

    pose = person.get("pose_keypoints_2d") or []
    for idx in OP_POSE_ORDER:
        base = idx * 3
        if base + 2 < len(pose):
            x, y, c = pose[base], pose[base + 1], pose[base + 2]
        else:
            x = y = c = 0.0
        vec.extend([x, y, c])

    for key in ("hand_left_keypoints_2d", "hand_right_keypoints_2d"):
        hand = person.get(key) or []
        for i in range(N_HAND):
            base = i * 3
            if base + 2 < len(hand):
                x, y, c = hand[base], hand[base + 1], hand[base + 2]
                # Match the server extractor: present -> conf 1.0, absent -> 0
                conf = 1.0 if (x != 0.0 or y != 0.0) else 0.0
            else:
                x = y = conf = 0.0
            vec.extend([x, y, conf])

    return np.array(vec, dtype=np.float32)


def load_clip_keypoints(folder):
    """Load all per-frame OpenPose JSONs in a folder -> (30, 144) normalized."""
    files = sorted(glob.glob(os.path.join(folder, "*_keypoints.json")))
    if not files:
        files = sorted(glob.glob(os.path.join(folder, "*.json")))
    if not files:
        return None

    frames = []
    for fp in files:
        try:
            with open(fp) as f:
                data = json.load(f)
        except Exception:
            continue
        people = data.get("people") or []
        if not people:
            frames.append(np.zeros(FEATURE_DIM, dtype=np.float32))
            continue
        frames.append(frame_vector_from_openpose(people[0]))

    if not frames:
        return None
    return normalize_keypoints(resample_frames(frames))


# ── CSV parsing ──────────────────────────────────────────────────────────────

def clean_word(sentence):
    return "".join(ch for ch in sentence.lower().strip()
                   if ch.isalnum() or ch == " ").strip()


def load_single_word_clips(asl_root):
    """Return list of dicts for clips whose SENTENCE is exactly one word."""
    csv_dir = os.path.join(asl_root, "how2sign_csvs")
    clips = []

    for split in SPLITS:
        path = os.path.join(csv_dir, f"how2sign_realigned_{split}.csv")
        if not os.path.exists(path):
            print(f"  [warn] missing {path}")
            continue

        with open(path, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                sentence = (row.get("SENTENCE") or "").strip()
                word = clean_word(sentence)
                if not word or " " in word:
                    continue          # keep only single-word clips
                if len(word) < 2:
                    continue          # drop stray single letters
                word = SYNONYMS.get(word, word)   # merge identical signs
                clips.append({
                    "word": word,
                    "sentence_name": (row.get("SENTENCE_NAME") or "").strip(),
                    "split": split,
                })

    return clips


def keypoint_folder(asl_root, split, sentence_name):
    return os.path.join(
        asl_root, "keypoints", f"{split}_2D_keypoints",
        "openpose_output", "json", sentence_name,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main(asl_root, out_dir, min_clips):
    os.makedirs(out_dir, exist_ok=True)

    print("\n=== Building ASL training set ===")
    print(f"ASL root : {asl_root}")
    print(f"Output   : {out_dir}")
    print(f"Min clips per word: {min_clips}\n")

    print("[1/3] Scanning CSVs for single-word clips...")
    clips = load_single_word_clips(asl_root)
    by_word = defaultdict(list)
    for c in clips:
        by_word[c["word"]].append(c)
    print(f"  Single-word clips : {len(clips)}")
    print(f"  Distinct words    : {len(by_word)}")

    eligible = {w: cs for w, cs in by_word.items() if len(cs) >= min_clips}
    print(f"  Words with >= {min_clips} clips: {len(eligible)}")
    if not eligible:
        print("\nNo words meet the threshold. Try a lower --min_clips.")
        return

    top = sorted(eligible.items(), key=lambda kv: -len(kv[1]))
    print("\n  Vocabulary to train:")
    for w, cs in top:
        print(f"    {w:15} {len(cs):4d} clips")

    print("\n[2/3] Extracting keypoints (this reads many JSON files)...")
    X, y, words_used = [], [], []
    word_to_idx = {}
    missing = 0

    for word, cs in top:
        kept = 0
        for c in cs:
            folder = keypoint_folder(asl_root, c["split"], c["sentence_name"])
            if not os.path.isdir(folder):
                missing += 1
                continue
            kps = load_clip_keypoints(folder)
            if kps is None:
                missing += 1
                continue
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
                words_used.append(word)
            X.append(kps)
            y.append(word_to_idx[word])
            kept += 1
        print(f"    {word:15} kept {kept}/{len(cs)}")

    if not X:
        print("\nNo keypoints could be extracted. Check the keypoints folder path.")
        return

    X = np.stack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.int64)

    print(f"\n  Extracted: {X.shape[0]} samples, {len(words_used)} classes")
    print(f"  Missing/unreadable folders: {missing}")

    print("\n[3/3] Saving...")
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)
    with open(os.path.join(out_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump({i: w for i, w in enumerate(words_used)}, f,
                  ensure_ascii=False, indent=2)

    print(f"  X.npy       {X.shape}")
    print(f"  y.npy       {y.shape}")
    print(f"  classes.json {len(words_used)} words")
    print("\nNext: py train_asl.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asl_root", default="E:/SL/ASL")
    ap.add_argument("--out", default="processed/asl_training")
    ap.add_argument("--min_clips", type=int, default=8,
                    help="Skip words with fewer clips than this")
    a = ap.parse_args()
    main(a.asl_root, a.out, a.min_clips)
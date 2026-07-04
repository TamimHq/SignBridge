"""
How2Sign ASL Vocabulary Index Builder  (v2 — matches your actual folder structure)
=====================================================================================

Your folder layout:
  D:/PROJECT_SL/ASL/
    how2sign_csvs/
      how2sign_realigned_train.csv
      how2sign_realigned_val.csv
      how2sign_realigned_test.csv
    keypoints/
      train_2D_keypoints/openpose_output/json/<SENTENCE_NAME>/
          <SENTENCE_NAME>_000000000000_keypoints.json
          <SENTENCE_NAME>_000000000001_keypoints.json
          ...  (one JSON per frame)
      val_2D_keypoints/openpose_output/json/<SENTENCE_NAME>/
      test_2D_keypoints/openpose_output/json/<SENTENCE_NAME>/
    train_rgb_front_clips/  (MP4 clips — not needed for this script)
    val_rgb_front_clips/
    test_rgb_front_clips/

Key facts discovered from your data:
  - 26,592 clip folders matched between keypoints and CSV
  - SENTENCE_NAME in CSV == folder name under openpose/json/
  - Each folder contains N per-frame JSON files (OpenPose body25 + hands format)
  - 589 single-word clips, 925 short-phrase clips in CSVs

Usage:
  python build_asl_index.py --asl_root "D:/PROJECT_SL/ASL" --output_dir "./processed/asl"
"""

import os
import re
import csv
import json
import glob
import string
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


# ── OpenPose keypoint structure ──────────────────────────────────────────────
# Each per-frame JSON has:
#   people[0].pose_keypoints_2d      → 75 floats = 25 body pts × (x,y,conf)
#   people[0].hand_left_keypoints_2d → 63 floats = 21 hand pts × (x,y,conf)
#   people[0].hand_right_keypoints_2d→ 63 floats = 21 hand pts × (x,y,conf)
#
# We keep: upper body joints (shoulders=2,5  elbows=3,6  wrists=4,7)
# + both full hands. That's 6 body pts + 21+21 hand pts = 48 pts × 3 = 144 values.

BODY_UPPER_IDX = [2, 3, 4, 5, 6, 7]   # body25 indices for upper body
N_BODY = len(BODY_UPPER_IDX)           # 6
N_HAND = 21
FEATURE_DIM = N_BODY * 3 + N_HAND * 3 + N_HAND * 3  # 18+63+63 = 144


# ── Gloss reordering ─────────────────────────────────────────────────────────
EN_DROP = {"a", "an", "the", "is", "are", "am", "was", "were"}
TIME_WORDS = {"today","yesterday","tomorrow","now","later","before","after"}
QUESTION_WORDS = {"what","where","when","who","why","how"}
COMMON_PHRASES = {
    "thank you","you're welcome","excuse me","i'm sorry",
    "good morning","good afternoon","good evening","good night",
    "how are you","nice to meet you","see you later","what is",
    "where is","how much","how many","i don't know","i understand",
}


def clean_sentence(text: str) -> str:
    text = re.sub(r'^[A-Z][A-Z\s]+:\s*', '', text.strip())  # strip speaker labels
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.lower().strip()


def reorder_gloss(words: list) -> list:
    if len(words) <= 2:
        return words  # too short to reorder safely
    q, t, rest = [], [], []
    for w in words:
        if w in EN_DROP:
            continue
        elif w in QUESTION_WORDS:
            q.append(w)
        elif w in TIME_WORDS:
            t.append(w)
        else:
            rest.append(w)
    return q + t + rest


# ── CSV loading ───────────────────────────────────────────────────────────────
def load_all_csvs(csv_dir: Path) -> list:
    rows = []
    for split, fname in [
        ("train", "how2sign_realigned_train.csv"),
        ("val",   "how2sign_realigned_val.csv"),
        ("test",  "how2sign_realigned_test.csv"),
    ]:
        fpath = csv_dir / fname
        if not fpath.exists():
            print(f"  [warn] Not found: {fpath}")
            continue

        # Try utf-8 first, fall back to utf-8-sig (handles BOM) then latin-1
        for encoding in ['utf-8', 'utf-8-sig', 'latin-1']:
            try:
                with open(fpath, newline='', encoding=encoding) as f:
                    reader = csv.DictReader(f, delimiter='\t')
                    file_rows = list(reader)

                # Check we got the expected columns
                if not file_rows:
                    print(f"  [warn] {fname} is empty")
                    break

                cols = list(file_rows[0].keys())

                # Strip BOM or whitespace from column names just in case
                file_rows = [{k.strip().lstrip('\ufeff'): v for k, v in r.items()}
                             for r in file_rows]
                cols_clean = list(file_rows[0].keys())

                if 'SENTENCE' not in cols_clean:
                    print(f"  [warn] {fname} columns: {cols_clean} — 'SENTENCE' not found, skipping")
                    break

                count = 0
                for row in file_rows:
                    sentence = row.get('SENTENCE', '') or ''
                    row['split'] = split
                    cleaned = clean_sentence(sentence)
                    words = [w for w in cleaned.split() if w]
                    row['_cleaned'] = cleaned
                    row['_words'] = words
                    row['_word_count'] = len(words)
                    rows.append(row)
                    count += 1

                print(f"  Loaded {fname}: {count} rows (encoding={encoding})")
                break  # success — don't try other encodings

            except UnicodeDecodeError:
                continue  # try next encoding
        else:
            print(f"  [error] Could not decode {fname} with any encoding — skipped")

    return rows


# ── Keypoint folder resolution ────────────────────────────────────────────────
SPLIT_TO_KP_FOLDER = {
    "train": "train_2D_keypoints",
    "val":   "val_2D_keypoints",
    "test":  "test_2D_keypoints",
}

def get_keypoint_folder(asl_root: Path, sentence_name: str, split: str) -> Path | None:
    """
    Resolve the per-clip keypoint folder.
    Path: <asl_root>/keypoints/<split>_2D_keypoints/openpose_output/json/<sentence_name>/
    """
    kp_split_folder = SPLIT_TO_KP_FOLDER.get(split, f"{split}_2D_keypoints")
    candidate = asl_root / "keypoints" / kp_split_folder / "openpose_output" / "json" / sentence_name
    if candidate.exists():
        return candidate
    # Sometimes folder has a leading dash in the name (video IDs starting with -)
    # sentence_name in CSV: "--7E2sU6zP4_10-5-rgb_front"
    # folder might be:      "--7E2sU6zP4_10-5-rgb_front" (same)
    # Already handled above — just return None if not found
    return None


# ── Per-frame OpenPose JSON parsing ───────────────────────────────────────────
def load_openpose_clip(clip_folder: Path) -> np.ndarray | None:
    """
    Load all per-frame keypoint JSONs from a clip folder.
    Returns array of shape (T, 144) or None if folder is empty.

    Each JSON file is one frame: <name>_XXXXXXXXX_keypoints.json
    """
    json_files = sorted(clip_folder.glob("*_keypoints.json"))
    if not json_files:
        return None

    frames = []
    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)
        except Exception:
            continue

        people = data.get('people', [])
        if not people:
            frames.append(np.zeros(FEATURE_DIM, dtype=np.float32))
            continue

        p = people[0]
        body_raw = np.array(p.get('pose_keypoints_2d', [0]*75), dtype=np.float32)
        lhand_raw = np.array(p.get('hand_left_keypoints_2d', [0]*63), dtype=np.float32)
        rhand_raw = np.array(p.get('hand_right_keypoints_2d', [0]*63), dtype=np.float32)

        # Extract upper-body joints only from body25
        body_vec = []
        for idx in BODY_UPPER_IDX:
            base = idx * 3
            body_vec.extend(body_raw[base:base+3])  # x, y, confidence

        frame_vec = np.concatenate([
            np.array(body_vec, dtype=np.float32),  # 18
            lhand_raw,                              # 63
            rhand_raw,                              # 63
        ])  # total: 144
        frames.append(frame_vec)

    if not frames:
        return None
    return np.stack(frames, axis=0)  # (T, 144)


def normalize_keypoints(kps: np.ndarray) -> np.ndarray:
    """
    Normalize so clips from different signers stitch without position jumps.
    Centers on mid-shoulder, scales by shoulder width.

    In our 144-dim vector:
      body upper = first 18 values = 6 joints × (x,y,conf)
      joint order matches BODY_UPPER_IDX = [2,3,4,5,6,7]
        idx 0 = joint2 (R shoulder): body_vec[0]=x, [1]=y, [2]=conf
        idx 1 = joint3 (R elbow):    body_vec[3]=x, [4]=y, [5]=conf
        idx 2 = joint4 (R wrist):    body_vec[6]=x, [7]=y
        idx 3 = joint5 (L shoulder): body_vec[9]=x, [10]=y
        idx 4 = joint6 (L elbow):    body_vec[12]=x, [13]=y
        idx 5 = joint7 (L wrist):    body_vec[15]=x, [16]=y
    """
    kps = kps.copy()

    r_shoulder_xy = kps[:, 0:2]    # joint2 x,y
    l_shoulder_xy = kps[:, 9:11]   # joint5 x,y
    mid_xy = (r_shoulder_xy + l_shoulder_xy) / 2   # (T,2)
    width = np.linalg.norm(r_shoulder_xy - l_shoulder_xy, axis=1, keepdims=True)
    width = np.where(width < 1e-6, 1.0, width)

    # Normalize x,y for each of the 6 body joints (stride 3: x,y,conf)
    for i in range(N_BODY):
        xi = i * 3
        yi = i * 3 + 1
        kps[:, xi] = (kps[:, xi] - mid_xy[:, 0]) / width[:, 0]
        kps[:, yi] = (kps[:, yi] - mid_xy[:, 1]) / width[:, 0]

    # Normalize x,y for each hand keypoint (stride 3: x,y,conf)
    hand_offset = N_BODY * 3  # 18
    for h in range(2):        # left then right hand
        for j in range(N_HAND):
            xi = hand_offset + h * N_HAND * 3 + j * 3
            yi = xi + 1
            kps[:, xi] = (kps[:, xi] - mid_xy[:, 0]) / width[:, 0]
            kps[:, yi] = (kps[:, yi] - mid_xy[:, 1]) / width[:, 0]

    return kps


# ── Main builder ──────────────────────────────────────────────────────────────
def build_asl_index(asl_root: str, output_dir: str, extract_keypoints: bool = True):
    asl_root = Path(asl_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kp_out_dir = output_dir / "keypoints" / "asl"

    print("\n=== How2Sign ASL Index Builder v2 ===")
    print(f"ASL root   : {asl_root}")
    print(f"Output dir : {output_dir}")

    # ── 1. Load CSVs ─────────────────────────────────────────────────────────
    print("\n[1/5] Loading CSVs...")
    csv_dir = asl_root / "how2sign_csvs"
    all_rows = load_all_csvs(csv_dir)
    print(f"  Total clips: {len(all_rows):,}")

    # ── 2. Classify by word count ─────────────────────────────────────────────
    print("\n[2/5] Classifying clips...")
    single_word, short_phrase, long_clips = [], [], []
    for row in all_rows:
        wc = row['_word_count']
        if wc == 1:
            single_word.append(row)
        elif wc <= 3:
            short_phrase.append(row)
        else:
            long_clips.append(row)

    print(f"  Single-word clips : {len(single_word):,}")
    print(f"  Short phrase clips: {len(short_phrase):,}")
    print(f"  Long clips        : {len(long_clips):,} (training only)")

    # ── 3. Build word & phrase vocabulary ────────────────────────────────────
    print("\n[3/5] Building vocabulary...")

    word_clips: dict[str, list] = defaultdict(list)
    for row in single_word:
        word = row['_words'][0] if row['_words'] else None
        if word:
            word_clips[word].append(row)

    phrase_clips: dict[str, list] = defaultdict(list)
    for row in short_phrase:
        phrase = row['_cleaned']
        if phrase in COMMON_PHRASES:
            phrase_clips[phrase].append(row)

    print(f"  Unique word types : {len(word_clips):,}")
    print(f"  Common phrases    : {len(phrase_clips):,}")

    # ── 4. Extract & normalize keypoints ─────────────────────────────────────
    print(f"\n[4/5] {'Extracting keypoints' if extract_keypoints else 'Skipping keypoint extraction'}...")

    def pick_canonical(clip_list: list) -> dict:
        """Pick best clip: prefer train split, then shortest duration."""
        train = [c for c in clip_list if c['split'] == 'train']
        pool = sorted(train or clip_list,
                      key=lambda r: float(r['END_REALIGNED']) - float(r['START_REALIGNED']))
        return pool[0]

    def process_kp(row: dict, category: str, safe_key: str) -> str | None:
        if not extract_keypoints:
            return None
        kp_folder = get_keypoint_folder(asl_root, row['SENTENCE_NAME'], row['split'])
        if kp_folder is None:
            return None
        kps = load_openpose_clip(kp_folder)
        if kps is None:
            return None
        kps_norm = normalize_keypoints(kps)
        out_path = kp_out_dir / category / f"{safe_key}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), kps_norm)
        return str(out_path)

    kp_found, kp_missing = 0, 0

    # Build canonical word entries
    canonical_words = {}
    total_words = len(word_clips)
    for i, (word, clips) in enumerate(word_clips.items()):
        if i % 100 == 0:
            print(f"  Words: {i}/{total_words}", end='\r')
        best = pick_canonical(clips)
        safe_key = re.sub(r'[^a-z0-9]', '_', word)
        npy_path = process_kp(best, "words", safe_key)
        if npy_path:
            kp_found += 1
        else:
            kp_missing += 1

        canonical_words[word] = {
            "type": "word",
            "word": word,
            "sentence_id": best['SENTENCE_ID'],
            "sentence_name": best['SENTENCE_NAME'],
            "video_id": best['VIDEO_ID'],
            "start_s": float(best['START_REALIGNED']),
            "end_s": float(best['END_REALIGNED']),
            "duration_s": round(float(best['END_REALIGNED']) - float(best['START_REALIGNED']), 2),
            "split": best['split'],
            "num_clips_available": len(clips),
            "keypoint_folder": str(get_keypoint_folder(asl_root, best['SENTENCE_NAME'], best['split']) or ''),
            "keypoint_npy_path": npy_path,
            "data_source": "how2sign",
            "fallback_type": "exact",
        }

    print(f"\n  Words done. kp_found={kp_found}, kp_missing={kp_missing}")

    # Build canonical phrase entries
    canonical_phrases = {}
    for phrase, clips in phrase_clips.items():
        best = pick_canonical(clips)
        safe_key = re.sub(r'[^a-z0-9]', '_', phrase)
        npy_path = process_kp(best, "phrases", safe_key)
        canonical_phrases[phrase] = {
            "type": "phrase",
            "phrase": phrase,
            "words": phrase.split(),
            "sentence_id": best['SENTENCE_ID'],
            "sentence_name": best['SENTENCE_NAME'],
            "video_id": best['VIDEO_ID'],
            "start_s": float(best['START_REALIGNED']),
            "end_s": float(best['END_REALIGNED']),
            "duration_s": round(float(best['END_REALIGNED']) - float(best['START_REALIGNED']), 2),
            "split": best['split'],
            "keypoint_folder": str(get_keypoint_folder(asl_root, best['SENTENCE_NAME'], best['split']) or ''),
            "keypoint_npy_path": npy_path,
            "data_source": "how2sign",
            "fallback_type": "exact",
        }

    # Fingerspelling placeholder (populate later with ASL alphabet clips)
    fingerspell_index = {
        c: {"letter": c, "keypoint_npy_path": None, "data_source": "todo"}
        for c in "abcdefghijklmnopqrstuvwxyz"
    }

    # ── 5. Save index ──────────────────────────────────────────────────────────
    print("\n[5/5] Saving index...")

    # Full training manifest (all clips, for recognition model training)
    all_clips_export = [{
        "sentence_id": r['SENTENCE_ID'],
        "sentence_name": r['SENTENCE_NAME'],
        "video_id": r['VIDEO_ID'],
        "start_s": float(r['START_REALIGNED']),
        "end_s": float(r['END_REALIGNED']),
        "duration_s": round(float(r['END_REALIGNED']) - float(r['START_REALIGNED']), 2),
        "sentence": r['SENTENCE'],
        "word_count": r['_word_count'],
        "split": r['split'],
        "keypoint_folder": str(get_keypoint_folder(asl_root, r['SENTENCE_NAME'], r['split']) or ''),
    } for r in all_rows]

    vocabulary_index = {
        "language": "asl_en",
        "total_word_types": len(canonical_words),
        "total_phrase_types": len(canonical_phrases),
        "total_training_clips": len(all_rows),
        "word_vocabulary": canonical_words,
        "phrase_vocabulary": canonical_phrases,
        "fingerspell_index": fingerspell_index,
    }

    index_path = output_dir / "asl_vocabulary_index.json"
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(vocabulary_index, f, indent=2)

    # Per-split training manifests
    for split_name in ["train", "val", "test"]:
        split_rows = [c for c in all_clips_export if c["split"] == split_name]
        mpath = output_dir / f"asl_training_manifest_{split_name}.json"
        with open(mpath, 'w') as f:
            json.dump(split_rows, f, indent=2)
        print(f"  Saved {split_name} manifest: {len(split_rows):,} clips → {mpath.name}")

    print(f"\n✓ Done.")
    print(f"  Vocabulary index  : {index_path}")
    print(f"  Word types        : {len(canonical_words):,}")
    print(f"  Phrase types      : {len(canonical_phrases):,}")
    print(f"  Training clips    : {len(all_rows):,}")

    # Top words
    top = sorted(canonical_words.items(), key=lambda x: x[1]['num_clips_available'], reverse=True)[:10]
    print(f"\n  Top 10 single-word types by occurrence:")
    for w, info in top:
        print(f"    '{w}': {info['num_clips_available']} clips, {info['duration_s']}s canonical")

    return vocabulary_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asl_root", required=True,
        help="Root folder: D:/PROJECT_SL/ASL")
    parser.add_argument("--output_dir", default="./processed/asl",
        help="Where to write index JSON and keypoint .npy files")
    parser.add_argument("--skip_keypoints", action="store_true",
        help="Build index only (no keypoint extraction) — fast, for testing")
    args = parser.parse_args()

    build_asl_index(args.asl_root, args.output_dir,
                    extract_keypoints=not args.skip_keypoints)
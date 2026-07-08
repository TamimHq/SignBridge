"""
Convert ASL .npy keypoint files into JSON for Flutter.

Dart cannot read NumPy .npy binary files. This script converts every
extracted keypoint array into a plain JSON file: {"word": ..., "frames": [[144 floats], ...]}

Reads from:  processed/asl/keypoints/asl/{words,phrases}/*.npy
Writes to:   <flutter_project>/assets/asl_keypoints/{words,phrases}/<safe_name>.json

Usage:
  py convert_npy_to_json.py \
      --processed_dir "processed/asl" \
      --flutter_assets "D:/project_SL/signbridge/assets/asl_keypoints"
"""

import json
import argparse
import numpy as np
from pathlib import Path


def convert_folder(src_dir: Path, dest_dir: Path, category: str) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for npy_path in sorted(src_dir.glob("*.npy")):
        arr = np.load(str(npy_path))          # shape (T, 144)
        frames = arr.tolist()                  # list of list of floats

        # npy filename already equals the safe_key exactly
        # (e.g. "hello.npy", "thank_you.npy") — no suffix to strip.
        # Must match Dart's word.replaceAll(RegExp('[^a-z0-9]'), '_') exactly.
        safe_name = npy_path.stem

        out_path = dest_dir / f"{safe_name}.json"
        with open(out_path, 'w') as f:
            json.dump({"frames": frames}, f)
        count += 1
    return count


def main(processed_dir: str, flutter_assets: str):
    processed_dir = Path(processed_dir)
    flutter_assets = Path(flutter_assets)

    print("\n=== Converting .npy keypoints to JSON for Flutter ===\n")

    words_src = processed_dir / "keypoints" / "asl" / "words"
    phrases_src = processed_dir / "keypoints" / "asl" / "phrases"

    n_words = 0
    n_phrases = 0

    if words_src.exists():
        n_words = convert_folder(words_src, flutter_assets / "words", "word")
        print(f"  Converted {n_words} word keypoint files")
    else:
        print(f"  [warn] {words_src} not found")

    if phrases_src.exists():
        n_phrases = convert_folder(phrases_src, flutter_assets / "phrases", "phrase")
        print(f"  Converted {n_phrases} phrase keypoint files")
    else:
        print(f"  [warn] {phrases_src} not found")

    print(f"\n✓ Done. Total: {n_words + n_phrases} JSON files written to {flutter_assets}")
    print(f"\nNext steps:")
    print(f"  1. Add to pubspec.yaml assets:")
    print(f"       - assets/asl_keypoints/words/")
    print(f"       - assets/asl_keypoints/phrases/")
    print(f"  2. Run: flutter pub get")
    print(f"  3. Re-run build_asl_index.py to regenerate the vocabulary index")
    print(f"     with updated keypoint_npy_path values (see next fix)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="processed/asl")
    parser.add_argument("--flutter_assets", default="signbridge/assets/asl_keypoints")
    args = parser.parse_args()
    main(args.processed_dir, args.flutter_assets)
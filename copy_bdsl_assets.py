"""
Copy SignBD-Word bodypose PNGs into Flutter assets structure.

What it does:
  Reads each gloss from bdsl_vocabulary_index_train.json
  Finds the canonical clip folder (c_ view, first person)
  Copies the 30 rendered PNG frames into:
    <flutter_project>/assets/bdsl_poses/<gloss>/frame_000000.png
    <flutter_project>/assets/bdsl_poses/<gloss>/frame_000001.png
    ...

Usage:
  py copy_bdsl_assets.py \
      --signbd_root "D:/project_SL/SignBD-Word" \
      --index "processed/bdsl/bdsl_vocabulary_index_train.json" \
      --flutter_assets "D:/project_SL/signbridge/assets/bdsl_poses"
"""

import os
import json
import shutil
import argparse
from pathlib import Path


def find_rendered_pngs(clip_folder: Path) -> list[Path]:
    """
    Find all rendered PNG frames inside a clip folder.
    Handles naming like:
      p1_c_maa_000000000001_rendered.png
      c_ma_000000000001_rendered.png
      000000000001_rendered.png
    Returns sorted list of PNG paths.
    """
    pngs = sorted(clip_folder.glob("*_rendered.png"))
    if not pngs:
        # Some folders just have plain PNGs
        pngs = sorted(clip_folder.glob("*.png"))
    return pngs


def copy_bdsl_assets(signbd_root: str, index_path: str, flutter_assets: str):
    signbd_root   = Path(signbd_root)
    flutter_assets = Path(flutter_assets)
    flutter_assets.mkdir(parents=True, exist_ok=True)

    with open(index_path, encoding='utf-8') as f:
        index = json.load(f)

    glosses = index.get('glosses', {})
    all_clips = index.get('all_clips', [])

    print(f"\n=== Copying BdSL assets to Flutter ===")
    print(f"SignBD-Word root : {signbd_root}")
    print(f"Flutter assets   : {flutter_assets}")
    print(f"Total glosses    : {len(glosses)}")
    print()

    ok = 0
    missing = 0
    empty = 0

    for gloss, entry in glosses.items():
        # Destination folder
        dest_dir = flutter_assets / gloss
        dest_dir.mkdir(exist_ok=True)

        # Source: bodypose_folder from index
        # e.g. D:\project_SL\SignBD-Word\bodypose\Train\maa\c_ma.mp4
        src_folder = Path(entry.get('bodypose_folder', ''))

        if not src_folder.exists():
            # Try to reconstruct path from signbd_root
            # Pattern: <signbd_root>/bodypose/Train/<gloss>/<first_subfolder>
            gloss_dir = signbd_root / 'bodypose' / 'Train' / gloss
            if gloss_dir.exists():
                subfolders = [d for d in gloss_dir.iterdir() if d.is_dir()]
                # Prefer 'c' view (upperbody)
                c_folders = [d for d in subfolders if '_c_' in d.name or d.name.startswith('c_')]
                src_folder = c_folders[0] if c_folders else (subfolders[0] if subfolders else None)

        if src_folder is None or not src_folder.exists():
            print(f"  [MISSING] {gloss}: folder not found")
            missing += 1
            continue

        # Find PNG frames
        pngs = find_rendered_pngs(src_folder)

        if not pngs:
            print(f"  [EMPTY]   {gloss}: no PNGs in {src_folder.name}")
            empty += 1
            continue

        # Copy and rename to frame_000000.png, frame_000001.png, ...
        for i, src_png in enumerate(pngs):
            dest_name = f"frame_{i:06d}.png"
            dest_path = dest_dir / dest_name
            if not dest_path.exists():  # skip if already copied
                shutil.copy2(src_png, dest_path)

        ok += 1
        if ok <= 5 or ok % 20 == 0:
            print(f"  [OK] {gloss:20} → {len(pngs)} frames → {dest_dir.name}/")

    print(f"\n✓ Done.")
    print(f"  Copied  : {ok} glosses")
    print(f"  Missing : {missing} glosses (folder not found)")
    print(f"  Empty   : {empty} glosses (no PNGs found)")

    # Also write a manifest file Flutter can use to verify assets
    manifest = {
        gloss: {
            "frame_count": len(list((flutter_assets / gloss).glob("*.png"))),
            "english": glosses[gloss].get('english', ''),
            "bangla": glosses[gloss].get('bangla', ''),
        }
        for gloss in glosses
        if (flutter_assets / gloss).exists()
    }
    manifest_path = flutter_assets.parent / 'indices' / 'bdsl_asset_manifest.json'
    manifest_path.parent.mkdir(exist_ok=True)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"  Manifest : {manifest_path}")

    # Print pubspec.yaml reminder
    print(f"""
Next step — make sure pubspec.yaml has:
  flutter:
    assets:
      - assets/bdsl_poses/
      - assets/indices/

Flutter will automatically pick up all files in those folders.
""")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--signbd_root', default='D:/project_SL/SignBD-Word',
                        help='Root of SignBD-Word dataset')
    parser.add_argument('--index', default='processed/bdsl_vocabulary_index_train.json',
                        help='Path to bdsl_vocabulary_index_train.json')
    parser.add_argument('--flutter_assets', default='signbridge/assets/bdsl_poses',
                        help='Flutter project assets/bdsl_poses folder')
    args = parser.parse_args()

    copy_bdsl_assets(args.signbd_root, args.index, args.flutter_assets)

"""
Flatten BdSL pose PNGs so Flutter Web bundles them.

Problem: pubspec's `- assets/bdsl_poses/` does NOT recurse into subfolders,
so PNGs inside bdsl_poses/<gloss>/ never get included in the web build.

Fix: move every PNG up one level, encoding the gloss into the filename:
    bdsl_poses/shunno/frame_000000.png  →  bdsl_poses/shunno__frame_000000.png
    bdsl_poses/maa/frame_000000.png     →  bdsl_poses/maa__frame_000000.png

After flattening, a single `- assets/bdsl_poses/` picks up everything because
all files sit directly in that one folder.

Usage:
  py flatten_bdsl_assets.py --bdsl_poses "E:/PROJECT_SL/signbridge/assets/bdsl_poses"
"""

import shutil
import argparse
from pathlib import Path


def flatten(bdsl_poses: str, keep_subfolders: bool = False):
    root = Path(bdsl_poses)
    if not root.exists():
        print(f"[error] Folder not found: {root}")
        return

    print(f"\n=== Flattening BdSL pose PNGs ===")
    print(f"Folder: {root}\n")

    moved = 0
    gloss_count = 0

    # Find all gloss subfolders
    subfolders = [d for d in root.iterdir() if d.is_dir()]
    print(f"Found {len(subfolders)} gloss subfolders\n")

    for gloss_dir in sorted(subfolders):
        gloss = gloss_dir.name
        pngs = sorted(gloss_dir.glob("*.png"))
        if not pngs:
            continue

        for png in pngs:
            # New name: <gloss>__<original_name>
            # e.g. shunno__frame_000000.png
            new_name = f"{gloss}__{png.name}"
            dest = root / new_name
            if not dest.exists():
                shutil.copy2(png, dest)
                moved += 1

        gloss_count += 1
        if gloss_count <= 5 or gloss_count % 40 == 0:
            print(f"  [{gloss_count}] {gloss}: {len(pngs)} frames flattened")

    print(f"\n✓ Flattened {moved} PNGs from {gloss_count} glosses")

    # Optionally remove the now-redundant subfolders
    if not keep_subfolders:
        print("\nRemoving original subfolders...")
        for gloss_dir in subfolders:
            shutil.rmtree(gloss_dir)
        print("  Subfolders removed")
    else:
        print("\nKept original subfolders (--keep_subfolders)")

    # Verify
    flat_pngs = list(root.glob("*.png"))
    print(f"\nTotal PNGs now directly in bdsl_poses/: {len(flat_pngs)}")
    print(f"Sample names:")
    for p in sorted(flat_pngs)[:3]:
        print(f"  {p.name}")

    print(f"""
Next steps:
  1. pubspec.yaml stays the same: `- assets/bdsl_poses/`
     (now works because all PNGs are directly in that folder)
  2. Update lookup_engine.dart path pattern to:
       'assets/bdsl_poses/${{glossKey}}__frame_$num.png'
     (double underscore between gloss and frame)
  3. flutter clean && flutter pub get && flutter run -d chrome
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bdsl_poses",
                        default="E:/PROJECT_SL/signbridge/assets/bdsl_poses")
    parser.add_argument("--keep_subfolders", action="store_true",
                        help="Keep original subfolders (default: remove them)")
    args = parser.parse_args()
    flatten(args.bdsl_poses, args.keep_subfolders)
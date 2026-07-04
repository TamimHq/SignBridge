"""
Test real indexes with the lookup engine.
Run from D:/project_SL/:
    py test_real_indexes.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from lookup_engine import SignLookupEngine

# ── Paths — adjust if your output folder is different ────────────────────────
ASL_INDEX  = "processed/asl/asl_vocabulary_index.json"
BDSL_INDEX = "processed/bdsl/bdsl_vocabulary_index_train.json"

# ── Load engine ───────────────────────────────────────────────────────────────
print("\n=== Loading real indexes ===")
engine = SignLookupEngine(
    asl_index_path=ASL_INDEX,
    bdsl_index_path=BDSL_INDEX
)
print()

stats = engine.get_vocabulary_stats()
print(f"ASL  vocabulary : {stats['asl_en']['words']} words, {stats['asl_en']['phrases']} phrases")
print(f"BdSL vocabulary : {stats['bdsl_bn']['glosses']} glosses")

# ── ASL tests ─────────────────────────────────────────────────────────────────
print("\n=== ASL / English tests ===\n")

asl_tests = [
    "Hello yes okay good",
    "Thank you",
    "No why",
    "Beautiful wonderful perfect",
    "I want help please",        # 'i','want','please' → fingerspell; 'help' not in vocab either
    "Hi bye welcome",
]

for text in asl_tests:
    result = engine.lookup(text, language="asl_en")
    exact   = [(c.word, c.clip_type) for c in result.clips if c.clip_type == "exact"]
    fspell  = [c.word for c in result.clips if c.clip_type == "fingerspell"]
    skipped = result.oov_words

    print(f'INPUT   : "{text}"')
    print(f'GLOSS   : {result.gloss_sequence}')
    print(f'EXACT   : {exact}')
    if fspell:
        print(f'FSPELL  : {fspell}')
    if skipped:
        print(f'SKIPPED : {skipped}')
    print(f'EST DUR : {result.estimated_duration_s:.1f}s')
    print()

# ── BdSL tests ────────────────────────────────────────────────────────────────
print("=== BdSL / Bengali tests ===\n")

bdsl_tests = [
    "মা বাড়ি ভাল",          # Mother House Good — all in vocab
    "বাবা ভাই বোন",          # Father Brother Sister
    "ধন্যবাদ দয়া",           # Thanks Please
    "আমি ভাল আছি",           # I Good [are] — আছি will be OOV
    "ডাক্তার হাসপাতাল",      # Doctor Hospital
    "আমার বাড়ি বাংলাদেশ",   # My House Bangladesh
]

for text in bdsl_tests:
    result = engine.lookup(text, language="bdsl_bn")
    exact   = [(c.word, c.clip_type) for c in result.clips if c.clip_type == "exact"]
    skipped = result.oov_words

    print(f'INPUT   : "{text}"')
    print(f'GLOSS   : {result.gloss_sequence}')
    print(f'EXACT   : {exact}')
    if skipped:
        print(f'SKIPPED : {skipped}')
    print(f'EST DUR : {result.estimated_duration_s:.1f}s')
    print()

# ── Folder path check ─────────────────────────────────────────────────────────
print("=== Spot-checking keypoint/bodypose folder paths ===\n")

# ASL: check 3 word entries have folder paths
asl_words = engine.asl_index.get('word_vocabulary', {})
for word in ['hi', 'hello', 'good']:
    entry = asl_words.get(word, {})
    folder = entry.get('keypoint_folder', 'MISSING')
    npy    = entry.get('keypoint_npy_path') or '(run without --skip_keypoints to populate)'
    print(f"ASL '{word}':")
    print(f"  folder : {folder}")
    print(f"  npy    : {npy}")
    print()

# BdSL: check 3 gloss entries have bodypose folders
bdsl_glosses = engine.bdsl_index.get('glosses', {})
for gloss in ['maa', 'baari', 'valo']:
    entry = bdsl_glosses.get(gloss, {})
    folder = entry.get('bodypose_folder', 'MISSING')
    bangla = entry.get('bangla', '?')
    frames = entry.get('frame_count', 0)
    print(f"BdSL '{gloss}' ({bangla}):")
    print(f"  bodypose_folder : {folder}")
    print(f"  frame_count     : {frames}")
    print()

print("=== All checks done ===")
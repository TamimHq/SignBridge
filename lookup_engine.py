"""
Sign Avatar Lookup Engine (Tier 1)
====================================
Runtime module that converts typed text → ordered list of sign clips to play.

Handles:
  - ASL/English  (vocabulary_index from build_asl_index.py)
  - BdSL/Bengali (vocabulary_index from build_bdsl_index.py)
  - Phrase matching before word matching
  - Gloss reordering (drop articles, copulas)
  - OOV fallback: fingerspelling → skip-with-flag
  - Linear interpolation metadata for smooth clip stitching

Usage:
  from lookup_engine import SignLookupEngine

  engine = SignLookupEngine(
      asl_index_path="processed/asl/asl_vocabulary_index.json",
      bdsl_index_path="processed/bdsl/bdsl_vocabulary_index_train.json"
  )

  result = engine.lookup("I want water", language="asl_en")
  # returns list of PlaybackClip objects in order
"""

import json
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─── Data classes ────────────────────────────────────────────────────────────

@dataclass
class PlaybackClip:
    """One unit of avatar playback — corresponds to one sign or one letter."""
    word: str                          # The word/phrase/letter this represents
    clip_type: str                     # "exact", "fingerspell", "skip"
    data_source: str                   # "how2sign", "signbd_word", "none"
    keypoint_npy_path: Optional[str]   # Path to .npy keypoint array, or None
    bodypose_folder: Optional[str]     # For BdSL PNG playback
    frame_count: int = 0
    duration_estimate_s: float = 0.0
    interpolate_before: bool = True    # Add transition frames before this clip
    language: str = "asl_en"
    # For fingerspelling
    letters: list = field(default_factory=list)
    sub_clips: list = field(default_factory=list)  # sub PlaybackClips for letters


@dataclass
class LookupResult:
    """Full result of a text → sign lookup."""
    input_text: str
    language: str
    clips: list           # List of PlaybackClip, in playback order
    oov_words: list       # Words that had no clip and were skipped
    fingerspelled_words: list  # Words that fell back to fingerspelling
    total_clips: int
    estimated_duration_s: float
    gloss_sequence: list  # Reordered word sequence used for lookup


# ─── Gloss reordering rules ───────────────────────────────────────────────────
# Simple rule-based, NOT ML. Keep it small and debuggable.

EN_ARTICLES = {"a", "an", "the"}
EN_COPULAS = {"is", "are", "am", "was", "were", "be", "been", "being"}
EN_FILLER = {"just", "like", "you", "know", "kind", "of", "sort", "um", "uh"}

BN_ARTICLES = set()  # Bengali has no articles
BN_COPULAS = {"আছে", "ছিল", "হয়", "হবে"}  # common Bengali copulas


def reorder_gloss_english(words: list) -> list:
    """
    Simplify English word list to ASL-friendly gloss order.
    Rules (applied in order):
      1. Drop articles (a, an, the)
      2. Drop copulas unless sentence is very short (≤3 words)
      3. Move time/question words to front
      4. Keep everything else in order
    """
    TIME_WORDS = {"today", "yesterday", "tomorrow", "now", "later",
                  "before", "after", "always", "never", "sometimes"}
    QUESTION_WORDS = {"what", "where", "when", "who", "why", "how"}

    # Don't strip copulas from very short sentences — "I am tired" → "I tired" is fine
    # but "I am" alone shouldn't become just "I"
    drop_copulas = len(words) > 3

    time_front = []
    question_front = []
    remaining = []

    for w in words:
        wl = w.lower()
        if wl in EN_ARTICLES:
            continue  # drop
        if drop_copulas and wl in EN_COPULAS:
            continue  # drop
        if wl in TIME_WORDS:
            time_front.append(w)
        elif wl in QUESTION_WORDS:
            question_front.append(w)
        else:
            remaining.append(w)

    return question_front + time_front + remaining


def reorder_gloss_bengali(words: list) -> list:
    """
    Simplify Bengali word list to BdSL-friendly gloss order.
    Bengali word order is already SOV which is closer to sign language order.
    Minimal changes: drop copulas from longer sentences.
    """
    drop_copulas = len(words) > 3
    result = []
    for w in words:
        if drop_copulas and w in BN_COPULAS:
            continue
        result.append(w)
    return result


# ─── Tokenizers ──────────────────────────────────────────────────────────────

def tokenize_english(text: str) -> list:
    """Simple English tokenizer."""
    text = text.lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return [w for w in text.split() if w]


def tokenize_bengali(text: str) -> list:
    """
    Bengali tokenizer. Bengali words are space-separated in standard text,
    but punctuation handling differs. Uses basic split + strip.

    For production, replace with:
      from bnlp import BasicTokenizer
      bt = BasicTokenizer()
      return bt.tokenize(text)
    """
    # Remove Bengali punctuation (।, ?, !, ,)
    text = re.sub(r'[।?!,;:\-"\'()]', ' ', text)
    return [w.strip() for w in text.split() if w.strip()]


# ─── Main engine ─────────────────────────────────────────────────────────────

class SignLookupEngine:
    """
    Converts typed text (English or Bengali) into an ordered list of
    PlaybackClip objects for the Tier 1 skeleton avatar renderer.
    """

    AVG_SECONDS_PER_WORD = 1.2    # rough estimate for timing display
    AVG_SECONDS_PER_LETTER = 0.4  # for fingerspelling

    def __init__(
        self,
        asl_index_path: str | None = None,
        bdsl_index_path: str | None = None,
    ):
        self.asl_index = None
        self.bdsl_index = None

        if asl_index_path and Path(asl_index_path).exists():
            with open(asl_index_path, encoding='utf-8') as f:
                self.asl_index = json.load(f)
            print(f"[Engine] ASL index loaded: "
                  f"{self.asl_index['total_word_types']} words, "
                  f"{self.asl_index['total_phrase_types']} phrases")

        if bdsl_index_path and Path(bdsl_index_path).exists():
            with open(bdsl_index_path, encoding='utf-8') as f:
                self.bdsl_index = json.load(f)
            print(f"[Engine] BdSL index loaded: "
                  f"{self.bdsl_index['total_glosses']} glosses")

            # Bengali normalization: decompose precomposed ড় ঢ় য় characters
            # so xlsx-stored keys match what users type on Bengali keyboards.
            _BN_PRE = {
                "\u09dc": "\u09a1\u09bc",
                "\u09dd": "\u09a2\u09bc",
                "\u09df": "\u09af\u09bc",
            }
            def _bn_norm(t):
                for p, d in _BN_PRE.items(): t = t.replace(p, d)
                return t.strip()
            raw_bn_map = self.bdsl_index.get("bangla_to_gloss", {})
            self._bn_nfkc_map = {_bn_norm(k): v for k, v in raw_bn_map.items()}
            self._bn_norm = _bn_norm

    # ── Public API ────────────────────────────────────────────────────────────

    def lookup(self, text: str, language: str = "asl_en") -> LookupResult:
        """
        Main lookup function.

        Args:
          text     : typed text from hearing user
          language : "asl_en" or "bdsl_bn"

        Returns:
          LookupResult with ordered list of PlaybackClips
        """
        if language == "asl_en":
            return self._lookup_asl(text)
        elif language == "bdsl_bn":
            return self._lookup_bdsl(text)
        else:
            raise ValueError(f"Unknown language: {language}. Use 'asl_en' or 'bdsl_bn'")

    # ── ASL / English ─────────────────────────────────────────────────────────

    def _lookup_asl(self, text: str) -> LookupResult:
        if not self.asl_index:
            raise RuntimeError("ASL index not loaded")

        words = tokenize_english(text)
        gloss_words = reorder_gloss_english(words)

        clips = []
        oov_words = []
        fingerspelled_words = []

        phrase_vocab = self.asl_index.get("phrase_vocabulary", {})
        word_vocab = self.asl_index.get("word_vocabulary", {})
        fs_index = self.asl_index.get("fingerspell_index", {})

        i = 0
        while i < len(gloss_words):
            # Try phrase match first (greedy, longest first up to 3 words)
            matched = False
            for length in [3, 2]:
                if i + length <= len(gloss_words):
                    phrase = " ".join(gloss_words[i:i+length])
                    if phrase in phrase_vocab:
                        entry = phrase_vocab[phrase]
                        clips.append(PlaybackClip(
                            word=phrase,
                            clip_type="exact",
                            data_source=entry["data_source"],
                            keypoint_npy_path=entry.get("keypoint_npy_path"),
                            bodypose_folder=None,
                            frame_count=0,
                            duration_estimate_s=entry.get("duration", self.AVG_SECONDS_PER_WORD * length),
                            interpolate_before=len(clips) > 0,
                            language="asl_en"
                        ))
                        i += length
                        matched = True
                        break

            if matched:
                continue

            # Try single word match
            w = gloss_words[i]
            if w in word_vocab:
                entry = word_vocab[w]
                clips.append(PlaybackClip(
                    word=w,
                    clip_type="exact",
                    data_source=entry["data_source"],
                    keypoint_npy_path=entry.get("keypoint_npy_path"),
                    bodypose_folder=None,
                    frame_count=0,
                    duration_estimate_s=entry.get("duration", self.AVG_SECONDS_PER_WORD),
                    interpolate_before=len(clips) > 0,
                    language="asl_en"
                ))
            else:
                # Fingerspelling fallback
                fs_sub = self._build_fingerspell_clips(w, fs_index, "asl_en")
                if fs_sub:
                    fingerspelled_words.append(w)
                    clips.append(PlaybackClip(
                        word=w,
                        clip_type="fingerspell",
                        data_source="fingerspell",
                        keypoint_npy_path=None,
                        bodypose_folder=None,
                        duration_estimate_s=len(w) * self.AVG_SECONDS_PER_LETTER,
                        interpolate_before=len(clips) > 0,
                        language="asl_en",
                        letters=list(w),
                        sub_clips=fs_sub
                    ))
                else:
                    # Skip — no clip, no fingerspelling data
                    oov_words.append(w)

            i += 1

        total_dur = sum(c.duration_estimate_s for c in clips)
        return LookupResult(
            input_text=text,
            language="asl_en",
            clips=clips,
            oov_words=oov_words,
            fingerspelled_words=fingerspelled_words,
            total_clips=len(clips),
            estimated_duration_s=total_dur,
            gloss_sequence=gloss_words
        )

    # ── BdSL / Bengali ────────────────────────────────────────────────────────

    def _lookup_bdsl(self, text: str) -> LookupResult:
        if not self.bdsl_index:
            raise RuntimeError("BdSL index not loaded")

        words = tokenize_bengali(text)
        gloss_words = reorder_gloss_bengali(words)

        clips = []
        oov_words = []
        fingerspelled_words = []

        bn_norm = getattr(self, "_bn_norm", lambda x: x)
        bn_map  = getattr(self, "_bn_nfkc_map", {})
        english_to_gloss = self.bdsl_index.get("english_to_gloss", {})
        glosses = self.bdsl_index.get("glosses", {})

        for w in gloss_words:
            # Normalize user input the same way we normalized the index keys
            gloss_key = bn_map.get(bn_norm(w))

            # If not found, try English (in case text is English for BdSL mode)
            if not gloss_key:
                gloss_key = english_to_gloss.get(w.lower())

            if gloss_key and gloss_key in glosses:
                entry = glosses[gloss_key]
                clips.append(PlaybackClip(
                    word=w,
                    clip_type="exact",
                    data_source="signbd_word",
                    keypoint_npy_path=entry.get("keypoint_npy_path"),
                    bodypose_folder=entry.get("bodypose_folder"),
                    frame_count=entry.get("frame_count", 30),
                    duration_estimate_s=entry.get("frame_count", 30) / 10.0,
                    interpolate_before=len(clips) > 0,
                    language="bdsl_bn"
                ))
            else:
                # BdSL fingerspelling is less standardized — skip for now
                oov_words.append(w)

        total_dur = sum(c.duration_estimate_s for c in clips)
        return LookupResult(
            input_text=text,
            language="bdsl_bn",
            clips=clips,
            oov_words=oov_words,
            fingerspelled_words=fingerspelled_words,
            total_clips=len(clips),
            estimated_duration_s=total_dur,
            gloss_sequence=gloss_words
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_fingerspell_clips(
        self, word: str, fs_index: dict, language: str
    ) -> list:
        """Build per-letter PlaybackClips for fingerspelling a word."""
        sub_clips = []
        for letter in word.lower():
            if letter in fs_index:
                entry = fs_index[letter]
                sub_clips.append(PlaybackClip(
                    word=letter,
                    clip_type="fingerspell",
                    data_source=entry.get("data_source", "none"),
                    keypoint_npy_path=entry.get("keypoint_npy_path"),
                    bodypose_folder=None,
                    duration_estimate_s=self.AVG_SECONDS_PER_LETTER,
                    interpolate_before=False,
                    language=language
                ))
        return sub_clips

    def get_vocabulary_stats(self) -> dict:
        """Return a summary of loaded vocabulary sizes."""
        stats = {}
        if self.asl_index:
            stats["asl_en"] = {
                "words": self.asl_index["total_word_types"],
                "phrases": self.asl_index["total_phrase_types"],
                "total_clips": self.asl_index.get("total_training_clips", self.asl_index.get("total_clips", 0))
            }
        if self.bdsl_index:
            stats["bdsl_bn"] = {
                "glosses": self.bdsl_index["total_glosses"]
            }
        return stats


# ─── Quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Test with mock index if no real files available
    print("=== Lookup Engine Test ===\n")

    # Create a tiny mock ASL index for testing
    mock_asl = {
        "language": "asl_en",
        "total_word_types": 3,
        "total_phrase_types": 1,
        "total_clips": 4,
        "word_vocabulary": {
            "hello": {"data_source": "how2sign", "keypoint_npy_path": None,
                      "duration": 0.5, "sentence_id": "test_1"},
            "water": {"data_source": "how2sign", "keypoint_npy_path": None,
                      "duration": 0.8, "sentence_id": "test_2"},
            "help": {"data_source": "how2sign", "keypoint_npy_path": None,
                     "duration": 0.6, "sentence_id": "test_3"},
        },
        "phrase_vocabulary": {
            "thank you": {"data_source": "how2sign", "keypoint_npy_path": None,
                          "duration": 0.8, "words": ["thank", "you"]}
        },
        "fingerspell_index": {
            c: {"data_source": "none", "keypoint_npy_path": None}
            for c in "abcdefghijklmnopqrstuvwxyz"
        }
    }

    mock_bdsl = {
        "language": "bdsl_bn",
        "total_glosses": 2,
        "english_to_gloss": {"water": "pani", "mother": "maa"},
        "bangla_to_gloss": {"মা": "maa", "বাড়ি": "baari"},
        "glosses": {
            "maa": {"bangla": "মা", "english": "Mother", "gloss": "maa",
                    "frame_count": 30, "bodypose_folder": "bodypose/Train/maa/p1_c_maa.mp4",
                    "keypoint_npy_path": None},
            "baari": {"bangla": "বাড়ি", "english": "House", "gloss": "baari",
                      "frame_count": 30, "bodypose_folder": "bodypose/Train/baari/p1_c_baari.mp4",
                      "keypoint_npy_path": None},
        }
    }

    import tempfile, os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as af:
        json.dump(mock_asl, af)
        asl_path = af.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as bf:
        json.dump(mock_bdsl, bf, ensure_ascii=False)
        bdsl_path = bf.name

    engine = SignLookupEngine(asl_index_path=asl_path, bdsl_index_path=bdsl_path)
    print()

    tests_asl = [
        "Hello, I want water please",
        "Thank you for the help",
        "My name is xyz",          # 'xyz' → fingerspell
    ]

    for text in tests_asl:
        result = engine.lookup(text, language="asl_en")
        print(f"INPUT  : \"{text}\"")
        print(f"GLOSS  : {result.gloss_sequence}")
        print(f"CLIPS  : {[(c.word, c.clip_type) for c in result.clips]}")
        if result.oov_words:
            print(f"SKIPPED: {result.oov_words}")
        if result.fingerspelled_words:
            print(f"FINGERSPELL: {result.fingerspelled_words}")
        print(f"EST DUR: {result.estimated_duration_s:.1f}s")
        print()

    tests_bdsl = [
        "মা বাড়ি",
        "আমি ভাল আছি",  # 'ভাল', 'আছি' — 'আছি' will be OOV
    ]
    print("--- BdSL ---")
    for text in tests_bdsl:
        result = engine.lookup(text, language="bdsl_bn")
        print(f"INPUT  : \"{text}\"")
        print(f"GLOSS  : {result.gloss_sequence}")
        print(f"CLIPS  : {[(c.word, c.clip_type) for c in result.clips]}")
        if result.oov_words:
            print(f"SKIPPED: {result.oov_words}")
        print()

    os.unlink(asl_path)
    os.unlink(bdsl_path)
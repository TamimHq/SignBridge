import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'models.dart';

// ─────────────────────────────────────────────────────────────────────────────
// Dart port of lookup_engine.py
// Works fully offline — reads bundled JSON index files
// ─────────────────────────────────────────────────────────────────────────────

// ── Bengali Unicode normalization ─────────────────────────────────────────────
// Precomposed → decomposed for ড় ঢ় য়
const _bnPrecomposed = {
  '\u09DC': '\u09A1\u09BC',
  '\u09DD': '\u09A2\u09BC',
  '\u09DF': '\u09AF\u09BC',
};

String bnNormalize(String text) {
  var result = text.trim();
  _bnPrecomposed.forEach((pre, dec) {
    result = result.replaceAll(pre, dec);
  });
  return result;
}

// ── English gloss reordering rules ───────────────────────────────────────────
const _enDrop = {'a', 'an', 'the', 'is', 'are', 'am', 'was', 'were'};
const _timeWords = {
  'today',
  'yesterday',
  'tomorrow',
  'now',
  'later',
  'before',
  'after',
};
const _questionWords = {'what', 'where', 'when', 'who', 'why', 'how'};

List<String> reorderGlossEn(List<String> words) {
  if (words.length <= 2) return words;
  final q = <String>[], t = <String>[], rest = <String>[];
  for (final w in words) {
    final wl = w.toLowerCase();
    if (_enDrop.contains(wl)) continue;
    if (_questionWords.contains(wl)) {
      q.add(w);
    } else if (_timeWords.contains(wl)) {
      t.add(w);
    } else {
      rest.add(w);
    }
  }
  return [...q, ...t, ...rest];
}

List<String> reorderGlossBn(List<String> words) {
  const bnCopulas = {'আছে', 'ছিল', 'হয়', 'হবে'};
  if (words.length <= 2) return words;
  return words.where((w) => !bnCopulas.contains(w)).toList();
}

// ── Common phrases ────────────────────────────────────────────────────────────
const _commonPhrases = {
  'thank you',
  "you're welcome",
  'excuse me',
  "i'm sorry",
  'good morning',
  'good afternoon',
  'good evening',
  'good night',
  'how are you',
  'nice to meet you',
  'see you later',
  'what is',
  'where is',
  'how much',
  'how many',
  "i don't know",
  'i understand',
};

// ── Main engine ───────────────────────────────────────────────────────────────

class SignLookupEngine {
  Map<String, dynamic>? _aslIndex;
  Map<String, dynamic>? _bdslIndex;
  Map<String, dynamic>?
  _bdslManifest; // real per-gloss frame counts from copy script

  // BdSL normalized Bengali → gloss key
  Map<String, String> _bnNormMap = {};

  bool get aslLoaded => _aslIndex != null;
  bool get bdslLoaded => _bdslIndex != null;

  // ── Load indexes from bundled assets ────────────────────────────────────────
  Future<void> loadIndexes() async {
    try {
      final aslJson = await rootBundle.loadString(
        'assets/indices/asl_vocabulary_index.json',
      );
      _aslIndex = jsonDecode(aslJson) as Map<String, dynamic>;
      debugPrint(
        '[Engine] ASL loaded: '
        '${_aslIndex!['total_word_types']} words, '
        '${_aslIndex!['total_phrase_types']} phrases',
      );
    } catch (e) {
      debugPrint('[Engine] ASL index not found: $e');
    }

    try {
      final bdslJson = await rootBundle.loadString(
        'assets/indices/bdsl_vocabulary_index_train.json',
      );
      _bdslIndex = jsonDecode(bdslJson) as Map<String, dynamic>;
      _buildBnNormMap();
      debugPrint(
        '[Engine] BdSL loaded: '
        '${_bdslIndex!['total_glosses']} glosses',
      );
    } catch (e) {
      debugPrint('[Engine] BdSL index not found: $e');
    }

    // Load asset manifest with real frame counts (written by copy_bdsl_assets.py)
    try {
      final manifestJson = await rootBundle.loadString(
        'assets/indices/bdsl_asset_manifest.json',
      );
      _bdslManifest = jsonDecode(manifestJson) as Map<String, dynamic>;
      debugPrint(
        '[Engine] BdSL asset manifest loaded: '
        '${_bdslManifest!.length} glosses',
      );
    } catch (e) {
      debugPrint('[Engine] BdSL manifest not found (using index counts): $e');
    }
  }

  void _buildBnNormMap() {
    final rawMap = (_bdslIndex!['bangla_to_gloss'] as Map<String, dynamic>);
    _bnNormMap = {
      for (final e in rawMap.entries) bnNormalize(e.key): e.value as String,
    };
    // Manual correction: Sister typo in xlsx
    _bnNormMap[bnNormalize('বোন')] = 'bon';
  }

  // ── Public lookup ────────────────────────────────────────────────────────────
  LookupResult lookup(String text, AppLanguage language) {
    return language == AppLanguage.aslEn ? _lookupAsl(text) : _lookupBdsl(text);
  }

  // ── ASL / English ────────────────────────────────────────────────────────────
  LookupResult _lookupAsl(String text) {
    final words = _tokenizeEn(text);
    final gloss = reorderGlossEn(words);

    final clips = <PlaybackClip>[];
    final oov = <String>[];
    final fspell = <String>[];

    final phraseVocab =
        _aslIndex?['phrase_vocabulary'] as Map<String, dynamic>? ?? {};
    final wordVocab =
        _aslIndex?['word_vocabulary'] as Map<String, dynamic>? ?? {};
    final fsIndex =
        _aslIndex?['fingerspell_index'] as Map<String, dynamic>? ?? {};

    int i = 0;
    while (i < gloss.length) {
      // Try phrase match first (longest first)
      bool matched = false;
      for (int len = 3; len >= 2; len--) {
        if (i + len <= gloss.length) {
          final phrase = gloss.sublist(i, i + len).join(' ');
          if (phraseVocab.containsKey(phrase)) {
            final entry = phraseVocab[phrase] as Map<String, dynamic>;
            clips.add(_buildAslClip(phrase, entry, isPhrase: true));
            i += len;
            matched = true;
            break;
          }
        }
      }
      if (matched) continue;

      final w = gloss[i];
      if (wordVocab.containsKey(w)) {
        clips.add(_buildAslClip(w, wordVocab[w] as Map<String, dynamic>));
      } else {
        // Fingerspelling fallback
        final sub = _buildFingerspellClips(w, fsIndex, AppLanguage.aslEn);
        if (sub.isNotEmpty) {
          fspell.add(w);
          clips.add(
            PlaybackClip(
              word: w,
              clipType: ClipType.fingerspell,
              dataSource: 'fingerspell',
              durationEstimateS: w.length * 0.4,
              language: AppLanguage.aslEn,
              letters: w.split(''),
              subClips: sub,
            ),
          );
        } else {
          oov.add(w);
          clips.add(
            PlaybackClip(
              word: w,
              clipType: ClipType.skip,
              dataSource: 'none',
              language: AppLanguage.aslEn,
            ),
          );
        }
      }
      i++;
    }

    return LookupResult(
      inputText: text,
      language: AppLanguage.aslEn,
      clips: clips,
      oovWords: oov,
      fingerspelledWords: fspell,
      glossSequence: gloss,
      estimatedDurationS: clips.fold(0, (s, c) => s + c.durationEstimateS),
    );
  }

  // ── BdSL / Bengali ────────────────────────────────────────────────────────
  LookupResult _lookupBdsl(String text) {
    final words = _tokenizeBn(text);
    final gloss = reorderGlossBn(words);

    final clips = <PlaybackClip>[];
    final oov = <String>[];

    final enMap =
        _bdslIndex?['english_to_gloss'] as Map<String, dynamic>? ?? {};
    final glosses = _bdslIndex?['glosses'] as Map<String, dynamic>? ?? {};

    for (final w in gloss) {
      // Bengali lookup with normalization
      String? glossKey = _bnNormMap[bnNormalize(w)];
      // Fallback: try English
      glossKey ??= enMap[w.toLowerCase()] as String?;

      if (glossKey != null && glosses.containsKey(glossKey)) {
        final entry = glosses[glossKey] as Map<String, dynamic>;
        clips.add(_buildBdslClip(w, glossKey, entry));
      } else {
        oov.add(w);
        clips.add(
          PlaybackClip(
            word: w,
            clipType: ClipType.skip,
            dataSource: 'none',
            language: AppLanguage.bdslBn,
          ),
        );
      }
    }

    return LookupResult(
      inputText: text,
      language: AppLanguage.bdslBn,
      clips: clips,
      oovWords: oov,
      fingerspelledWords: const [],
      glossSequence: gloss,
      estimatedDurationS: clips.fold(0, (s, c) => s + c.durationEstimateS),
    );
  }

  // ── Clip builders ────────────────────────────────────────────────────────────
  PlaybackClip _buildAslClip(
    String word,
    Map<String, dynamic> entry, {
    bool isPhrase = false,
  }) {
    // Must match Python's re.sub(r'[^a-z0-9]', '_', word) exactly so the
    // asset filename lines up with what convert_npy_to_json.py generated.
    final safeKey = word.toLowerCase().replaceAll(RegExp(r'[^a-z0-9]'), '_');
    final subfolder = isPhrase ? 'phrases' : 'words';
    final assetPath = 'assets/asl_keypoints/$subfolder/$safeKey.json';

    return PlaybackClip(
      word: word,
      clipType: ClipType.exact,
      dataSource: entry['data_source'] as String? ?? 'how2sign',
      keypointNpyPath:
          assetPath, // now an asset path, loaded async before playback
      frameCount: 0,
      durationEstimateS: (entry['duration_s'] as num?)?.toDouble() ?? 1.2,
      language: AppLanguage.aslEn,
    );
  }

  PlaybackClip _buildBdslClip(
    String word,
    String glossKey,
    Map<String, dynamic> entry,
  ) {
    final folder = entry['bodypose_folder'] as String? ?? '';

    // Use REAL frame count from asset manifest if available, else index value.
    // This prevents 404s from requesting frames that weren't actually copied.
    int frameCount = (entry['frame_count'] as num?)?.toInt() ?? 30;
    final manifestEntry = _bdslManifest?[glossKey] as Map<String, dynamic>?;
    if (manifestEntry != null && manifestEntry['frame_count'] != null) {
      frameCount = (manifestEntry['frame_count'] as num).toInt();
    }

    // Build PNG frame asset paths.
    // Files are FLATTENED into bdsl_poses/ with gloss in the filename,
    // because Flutter's `- assets/bdsl_poses/` does not recurse into subfolders.
    // Pattern: assets/bdsl_poses/<gloss>__frame_<num>.png
    final pngPaths = List.generate(frameCount, (i) {
      final num = i.toString().padLeft(6, '0');
      return 'assets/bdsl_poses/${glossKey}__frame_$num.png';
    });

    return PlaybackClip(
      word: word,
      clipType: ClipType.exact,
      dataSource: 'signbd_word',
      bodyposeFolderPath: folder,
      pngFramePaths: pngPaths,
      frameCount: frameCount,
      durationEstimateS: frameCount / 10.0,
      language: AppLanguage.bdslBn,
    );
  }

  List<PlaybackClip> _buildFingerspellClips(
    String word,
    Map<String, dynamic> fsIndex,
    AppLanguage lang,
  ) {
    final result = <PlaybackClip>[];
    for (final letter in word.toLowerCase().split('')) {
      if (fsIndex.containsKey(letter)) {
        result.add(
          PlaybackClip(
            word: letter,
            clipType: ClipType.fingerspell,
            dataSource: 'fingerspell',
            keypointNpyPath: fsIndex[letter]?['keypoint_npy_path'] as String?,
            durationEstimateS: 0.4,
            language: lang,
          ),
        );
      }
    }
    return result;
  }

  // ── Tokenizers ───────────────────────────────────────────────────────────────
  List<String> _tokenizeEn(String text) {
    return text
        .toLowerCase()
        .replaceAll(RegExp(r"[^\w\s']"), '')
        .split(RegExp(r'\s+'))
        .where((w) => w.isNotEmpty)
        .toList();
  }

  List<String> _tokenizeBn(String text) {
    // Bengali punctuation + common ASCII punctuation to strip before splitting
    final punctuation = RegExp('[।?!,;:\\-"\'()]');
    return text
        .replaceAll(punctuation, ' ')
        .split(RegExp(r'\s+'))
        .where((w) => w.isNotEmpty)
        .toList();
  }

  // ── Stats ────────────────────────────────────────────────────────────────────
  Map<String, dynamic> get stats => {
    'asl_en': aslLoaded
        ? {
            'words': _aslIndex!['total_word_types'],
            'phrases': _aslIndex!['total_phrase_types'],
          }
        : null,
    'bdsl_bn': bdslLoaded ? {'glosses': _bdslIndex!['total_glosses']} : null,
  };
}

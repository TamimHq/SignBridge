// ─────────────────────────────────────────────────────────────────────────────
// Core data models for SignBridge
// ─────────────────────────────────────────────────────────────────────────────

enum AppLanguage { aslEn, bdslBn }

extension AppLanguageX on AppLanguage {
  String get code => this == AppLanguage.aslEn ? 'asl_en' : 'bdsl_bn';
  String get label => this == AppLanguage.aslEn ? 'ASL / EN' : 'BdSL / বাংলা';
  String get ttsLang => this == AppLanguage.aslEn ? 'en-US' : 'bn-BD';
  bool get isBengali => this == AppLanguage.bdslBn;
}

// ── Playback clip ─────────────────────────────────────────────────────────────

enum ClipType { exact, fingerspell, skip }

class PlaybackClip {
  final String word;
  final ClipType clipType;
  final String dataSource; // 'how2sign' | 'signbd_word' | 'fingerspell'

  /// For ASL: path to .npy keypoint file (served by FastAPI or bundled)
  final String? keypointNpyPath;

  /// For BdSL: path to bodypose PNG folder
  final String? bodyposeFolderPath;

  /// Pre-loaded keypoint frames: list of frames, each frame = List<double> (144)
  /// Populated after fetch from server or asset load
  List<List<double>>? keypointFrames;

  /// For BdSL: list of asset paths to PNG frames
  List<String>? pngFramePaths;

  final int frameCount;
  final double durationEstimateS;
  final AppLanguage language;

  // For fingerspelling
  final List<String> letters;
  final List<PlaybackClip> subClips;

  PlaybackClip({
    required this.word,
    required this.clipType,
    required this.dataSource,
    this.keypointNpyPath,
    this.bodyposeFolderPath,
    this.keypointFrames,
    this.pngFramePaths,
    this.frameCount = 0,
    this.durationEstimateS = 1.2,
    this.language = AppLanguage.aslEn,
    this.letters = const [],
    this.subClips = const [],
  });

  bool get hasKeypoints => keypointFrames != null && keypointFrames!.isNotEmpty;
  bool get hasPngFrames => pngFramePaths != null && pngFramePaths!.isNotEmpty;
  bool get isSkip => clipType == ClipType.skip;
  bool get isFingerspell => clipType == ClipType.fingerspell;
}

// ── Lookup result ─────────────────────────────────────────────────────────────

class LookupResult {
  final String inputText;
  final AppLanguage language;
  final List<PlaybackClip> clips;
  final List<String> oovWords;
  final List<String> fingerspelledWords;
  final List<String> glossSequence;
  final double estimatedDurationS;

  const LookupResult({
    required this.inputText,
    required this.language,
    required this.clips,
    required this.oovWords,
    required this.fingerspelledWords,
    required this.glossSequence,
    required this.estimatedDurationS,
  });

  int get totalClips => clips.length;
  bool get hasOov => oovWords.isNotEmpty;
}

// ── Chat message ──────────────────────────────────────────────────────────────

enum MessageSource { signer, speaker, system }

class ChatMessage {
  final String id;
  final String text;
  final MessageSource source;
  final AppLanguage language;
  final DateTime timestamp;
  final bool isNew;

  ChatMessage({
    required this.text,
    required this.source,
    required this.language,
    DateTime? timestamp,
    this.isNew = false,
  }) : id = DateTime.now().millisecondsSinceEpoch.toString(),
       timestamp = timestamp ?? DateTime.now();

  String get timeLabel {
    final h = timestamp.hour.toString().padLeft(2, '0');
    final m = timestamp.minute.toString().padLeft(2, '0');
    return '$h:$m';
  }

  String get sourceLabel => switch (source) {
    MessageSource.signer => '✋ Signer',
    MessageSource.speaker => 'Speaker 🎙',
    MessageSource.system => 'System',
  };
}

// ── Sign recognition result (from server) ────────────────────────────────────

class SignRecognitionResult {
  final String word;
  final double confidence;
  final AppLanguage language;
  final List<double> glossVector; // optional embedding

  const SignRecognitionResult({
    required this.word,
    required this.confidence,
    required this.language,
    this.glossVector = const [],
  });

  factory SignRecognitionResult.fromJson(Map<String, dynamic> json) =>
      SignRecognitionResult(
        word: json['word'] as String,
        confidence: (json['confidence'] as num).toDouble(),
        language: json['language'] == 'asl_en'
            ? AppLanguage.aslEn
            : AppLanguage.bdslBn,
      );
}

// ── Keypoint frame (48 points × 3 values = 144 doubles) ──────────────────────

class KeypointFrame {
  /// Full 144-dim vector: 6 body pts × 3 + 21 L-hand × 3 + 21 R-hand × 3
  final List<double> values;

  const KeypointFrame(this.values);

  // Upper body joints (indices 0-5, stride 3)
  static const _bodyPts = 6;
  static const _handPts = 21;

  /// Get body joint [idx] as (x, y, confidence)
  (double, double, double) bodyJoint(int idx) {
    final base = idx * 3;
    return (values[base], values[base + 1], values[base + 2]);
  }

  /// Get left hand landmark [idx] as (x, y, confidence)
  (double, double, double) leftHand(int idx) {
    final base = _bodyPts * 3 + idx * 3;
    return (values[base], values[base + 1], values[base + 2]);
  }

  /// Get right hand landmark [idx] as (x, y, confidence)
  (double, double, double) rightHand(int idx) {
    final base = _bodyPts * 3 + _handPts * 3 + idx * 3;
    return (values[base], values[base + 1], values[base + 2]);
  }
}

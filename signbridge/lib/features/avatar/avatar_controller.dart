import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import '../../core/models.dart';

// ─────────────────────────────────────────────────────────────────────────────
// Avatar Playback Controller
// Manages the queue of PlaybackClips and drives frame-by-frame animation
// ─────────────────────────────────────────────────────────────────────────────

const _kFps = 10; // playback frames per second
const _kInterpolFrames = 5; // transition frames between clips

enum AvatarState { idle, playing, transitioning, done }

class AvatarController extends ChangeNotifier {
  // ── State ────────────────────────────────────────────────────────────────
  AvatarState state = AvatarState.idle;

  List<PlaybackClip> queue = [];
  int currentClipIdx = 0;
  int currentFrameIdx = 0;
  double transitionAlpha = 1.0; // 0→1 for fade in/out between clips

  // Current frame data
  KeypointFrame? currentKeypointFrame;
  String? currentPngPath; // for BdSL PNG display
  PlaybackClip? currentClip;

  // Queue display (all clips including skips)
  List<PlaybackClip> allClips = [];

  Timer? _timer;
  int _interpolCounter = 0;
  bool _transitioning = false;

  // ── Public API ────────────────────────────────────────────────────────────

  Future<void> playLookupResult(LookupResult result) async {
    stop();

    allClips = result.clips;
    // Only animate non-skip clips
    queue = result.clips.where((c) => c.clipType != ClipType.skip).toList();

    if (queue.isEmpty) {
      state = AvatarState.idle;
      notifyListeners();
      return;
    }

    // Show queue immediately while assets load in background
    state = AvatarState.playing;
    notifyListeners();

    // Preload keypoint JSON for every clip that needs it (ASL words/phrases
    // and fingerspell sub-clips). BdSL PNG paths need no preloading —
    // Image.asset() loads them lazily and synchronously enough for our fps.
    await _preloadKeypoints(queue);

    currentClipIdx = 0;
    currentFrameIdx = 0;
    notifyListeners();

    _scheduleNextFrame();
  }

  /// Loads the bundled JSON keypoint file for each clip (and each
  /// fingerspell sub-clip) that has a keypointNpyPath but no frames yet.
  Future<void> _preloadKeypoints(List<PlaybackClip> clips) async {
    for (final clip in clips) {
      await _preloadOne(clip);
      for (final sub in clip.subClips) {
        await _preloadOne(sub);
      }
    }
  }

  Future<void> _preloadOne(PlaybackClip clip) async {
    if (clip.keypointFrames != null) return; // already loaded
    final path = clip.keypointNpyPath;
    if (path == null || path.isEmpty) return;

    try {
      final jsonStr = await rootBundle.loadString(path);
      final data = jsonDecode(jsonStr) as Map<String, dynamic>;
      final rawFrames = data['frames'] as List;
      clip.keypointFrames = rawFrames
          .map((f) => (f as List).map((v) => (v as num).toDouble()).toList())
          .toList();
    } catch (e) {
      // Asset not found (word not yet converted to JSON, or path mismatch).
      // Clip will fall through to "no data" handling in _tick() and be skipped.
      debugPrint('[Avatar] Could not load keypoints for "${clip.word}": $e');
    }
  }

  void stop() {
    _timer?.cancel();
    _timer = null;
    state = AvatarState.idle;
    currentClip = null;
    currentKeypointFrame = null;
    currentPngPath = null;
    queue = [];
    allClips = [];
    currentClipIdx = 0;
    currentFrameIdx = 0;
    transitionAlpha = 1.0;
    _transitioning = false;
  }

  void pause() {
    _timer?.cancel();
    _timer = null;
  }

  void resume() {
    if (state == AvatarState.playing) _scheduleNextFrame();
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  // ── Internal playback loop ────────────────────────────────────────────────

  void _scheduleNextFrame() {
    _timer?.cancel();
    _timer = Timer(const Duration(milliseconds: 1000 ~/ _kFps), _tick);
  }

  void _tick() {
    if (state != AvatarState.playing && state != AvatarState.transitioning) {
      return;
    }
    if (currentClipIdx >= queue.length) {
      _onDone();
      return;
    }

    final clip = queue[currentClipIdx];
    currentClip = clip;

    // Handle fingerspell — flatten sub-clips into a virtual clip list
    if (clip.isFingerspell && clip.subClips.isNotEmpty) {
      _playFingerspellClip(clip);
      return;
    }

    // Transition fade-in at start of clip
    if (currentFrameIdx == 0 && currentClipIdx > 0) {
      _transitioning = true;
      state = AvatarState.transitioning;
      _interpolCounter = 0;
    }

    if (_transitioning) {
      _interpolCounter++;
      transitionAlpha = _interpolCounter / _kInterpolFrames;
      if (_interpolCounter >= _kInterpolFrames) {
        _transitioning = false;
        transitionAlpha = 1.0;
        state = AvatarState.playing;
      }
    }

    // Advance frame based on clip type
    if (clip.hasKeypoints) {
      _advanceKeypointFrame(clip);
    } else if (clip.hasPngFrames) {
      _advancePngFrame(clip);
    } else {
      // No data yet — skip this clip
      _nextClip();
      return;
    }

    notifyListeners();
    _scheduleNextFrame();
  }

  void _advanceKeypointFrame(PlaybackClip clip) {
    final frames = clip.keypointFrames!;
    if (currentFrameIdx < frames.length) {
      currentKeypointFrame = KeypointFrame(frames[currentFrameIdx]);
      currentPngPath = null;
      currentFrameIdx++;
    } else {
      _nextClip();
    }
  }

  void _advancePngFrame(PlaybackClip clip) {
    final paths = clip.pngFramePaths!;
    if (currentFrameIdx < paths.length) {
      currentPngPath = paths[currentFrameIdx];
      currentKeypointFrame = null;
      currentFrameIdx++;
    } else {
      _nextClip();
    }
  }

  // Fingerspell: treat each sub-clip letter as sequential frames
  int _fsSubIdx = 0;
  int _fsLetterFrameIdx = 0;
  static const _kFramesPerLetter = 8; // hold each letter pose for N frames

  void _playFingerspellClip(PlaybackClip clip) {
    if (_fsSubIdx >= clip.subClips.length) {
      _fsSubIdx = 0;
      _fsLetterFrameIdx = 0;
      _nextClip();
      return;
    }

    final letter = clip.subClips[_fsSubIdx];

    if (letter.hasKeypoints) {
      final frameIdx = _fsLetterFrameIdx % letter.keypointFrames!.length;
      currentKeypointFrame = KeypointFrame(letter.keypointFrames![frameIdx]);
      currentPngPath = null;
    }

    _fsLetterFrameIdx++;
    if (_fsLetterFrameIdx >= _kFramesPerLetter) {
      _fsLetterFrameIdx = 0;
      _fsSubIdx++;
    }

    notifyListeners();
    _scheduleNextFrame();
  }

  void _nextClip() {
    currentClipIdx++;
    currentFrameIdx = 0;
    _fsSubIdx = 0;
    _fsLetterFrameIdx = 0;

    if (currentClipIdx >= queue.length) {
      _onDone();
    } else {
      _scheduleNextFrame();
    }
  }

  void _onDone() {
    state = AvatarState.done;
    currentClip = null;
    _timer?.cancel();
    notifyListeners();

    // Return to idle after a short pause
    Timer(const Duration(milliseconds: 800), () {
      state = AvatarState.idle;
      currentKeypointFrame = null;
      currentPngPath = null;
      notifyListeners();
    });
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  /// Index of currently playing clip in allClips list
  int get playingAllClipsIdx {
    if (currentClip == null) return -1;
    return allClips.indexOf(currentClip!);
  }

  String get statusText {
    if (state == AvatarState.idle) return 'Ready';
    if (state == AvatarState.done) return 'Done';
    if (currentClip == null) return 'Loading...';
    return 'Signing: "${currentClip!.word}" '
        '(${currentClipIdx + 1}/${queue.length})';
  }
}

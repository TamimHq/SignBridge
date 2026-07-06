import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import '../../core/models.dart';
import '../../core/lookup_engine.dart';
import '../avatar/avatar_controller.dart';

// ─────────────────────────────────────────────────────────────────────────────
// Chat Controller
// Manages messages, TTS, language mode, and coordinates with AvatarController
// ─────────────────────────────────────────────────────────────────────────────

class ChatController extends ChangeNotifier {
  final SignLookupEngine lookupEngine;
  final AvatarController avatarController;
  final FlutterTts _tts = FlutterTts();

  List<ChatMessage> messages = [];
  AppLanguage currentLanguage = AppLanguage.aslEn;
  bool isSpeaking = false;
  bool isMicListening = false;
  String? lastRecognizedSign;
  double lastSignConfidence = 0.0;

  ChatController({required this.lookupEngine, required this.avatarController}) {
    _initTts();
    _addWelcomeMessage();
  }

  // ── TTS setup ────────────────────────────────────────────────────────────
  Future<void> _initTts() async {
    await _tts.setVolume(1.0);
    await _tts.setSpeechRate(0.9);
    await _tts.setPitch(1.0);

    _tts.setCompletionHandler(() {
      isSpeaking = false;
      notifyListeners();
    });
  }

  // ── Welcome ───────────────────────────────────────────────────────────────
  void _addWelcomeMessage() {
    messages.add(
      ChatMessage(
        text: 'SignBridge ready. Select ASL/EN or BdSL/বাংলা to begin.',
        source: MessageSource.system,
        language: AppLanguage.aslEn,
      ),
    );
  }

  // ── Language toggle ───────────────────────────────────────────────────────
  void setLanguage(AppLanguage lang) {
    currentLanguage = lang;
    notifyListeners();
  }

  // ── Send typed/mic message (hearing side → avatar signs it) ───────────────
  Future<void> sendSpeakerMessage(String text) async {
    if (text.trim().isEmpty) return;

    final msg = ChatMessage(
      text: text.trim(),
      source: MessageSource.speaker,
      language: currentLanguage,
      isNew: true,
    );
    messages.add(msg);
    notifyListeners();

    // Trigger avatar playback (async: preloads keypoint JSON before animating)
    final result = lookupEngine.lookup(text, currentLanguage);
    await avatarController.playLookupResult(result);
  }

  // ── Send recognized sign (Deaf side → text appears in chat) ──────────────
  void sendSignMessage() {
    if (lastRecognizedSign == null) return;

    messages.add(
      ChatMessage(
        text: lastRecognizedSign!,
        source: MessageSource.signer,
        language: currentLanguage,
        isNew: true,
      ),
    );
    lastRecognizedSign = null;
    lastSignConfidence = 0.0;
    notifyListeners();
  }

  // ── Update sign recognition (called by camera/server stream) ─────────────
  void updateSignPrediction(String word, double confidence) {
    lastRecognizedSign = word;
    lastSignConfidence = confidence;
    notifyListeners();
  }

  // ── TTS: read a message aloud ─────────────────────────────────────────────
  Future<void> speakMessage(ChatMessage msg) async {
    if (isSpeaking) {
      await _tts.stop();
      isSpeaking = false;
      notifyListeners();
      return;
    }

    final lang = msg.language.ttsLang;
    await _tts.setLanguage(lang);

    isSpeaking = true;
    notifyListeners();

    await _tts.speak(msg.text);
  }

  // ── TTS: stop ────────────────────────────────────────────────────────────
  Future<void> stopSpeaking() async {
    await _tts.stop();
    isSpeaking = false;
    notifyListeners();
  }

  // ── Mic state (actual recording handled by SpeechController) ─────────────
  void setMicListening(bool listening) {
    isMicListening = listening;
    notifyListeners();
  }

  // ── Transcription from Whisper (called after mic recording) ──────────────
  Future<void> onTranscription(String text) async {
    isMicListening = false;
    if (text.isNotEmpty) {
      await sendSpeakerMessage(text);
    }
    notifyListeners();
  }

  @override
  void dispose() {
    _tts.stop();
    super.dispose();
  }
}

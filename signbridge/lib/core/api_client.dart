import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'models.dart';

// ─────────────────────────────────────────────────────────────────────────────
// API Client
// Communicates with FastAPI server for ML inference
// Default: http://localhost:8000 (local development)
// Production: update serverUrl to your cloud endpoint
// ─────────────────────────────────────────────────────────────────────────────

class ApiClient extends ChangeNotifier {
  String _serverUrl = 'http://localhost:8000';
  bool _isConnected = false;
  String? _lastError;

  String get serverUrl => _serverUrl;
  bool get isConnected => _isConnected;
  String? get lastError => _lastError;

  // ── Init ──────────────────────────────────────────────────────────────────
  Future<void> init() async {
    final prefs = await SharedPreferences.getInstance();
    _serverUrl = prefs.getString('server_url') ?? 'http://localhost:8000';
    await checkConnection();
  }

  Future<void> setServerUrl(String url) async {
    _serverUrl = url.trimRight().replaceAll(RegExp(r'/$'), '');
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('server_url', _serverUrl);
    await checkConnection();
    notifyListeners();
  }

  // ── Health check ──────────────────────────────────────────────────────────
  Future<bool> checkConnection() async {
    try {
      final res = await http
          .get(Uri.parse('$_serverUrl/health'))
          .timeout(const Duration(seconds: 3));
      _isConnected = res.statusCode == 200;
      _lastError = _isConnected ? null : 'Server returned ${res.statusCode}';
    } catch (e) {
      _isConnected = false;
      _lastError = 'Cannot reach $_serverUrl';
    }
    notifyListeners();
    return _isConnected;
  }

  // ── Sign recognition ──────────────────────────────────────────────────────
  /// Send keypoint frame data to server for sign recognition
  /// Returns the recognized sign and confidence
  Future<SignRecognitionResult> recognizeSign({
    required List<List<double>> keypointBuffer, // last N frames
    required AppLanguage language,
  }) async {
    final res = await http
        .post(
          Uri.parse('$_serverUrl/api/recognize'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'keypoints': keypointBuffer,
            'language': language.code,
          }),
        )
        .timeout(const Duration(seconds: 2));

    if (res.statusCode != 200) {
      throw Exception('Recognize failed: ${res.statusCode}');
    }

    return SignRecognitionResult.fromJson(
      jsonDecode(res.body) as Map<String, dynamic>,
    );
  }

  // ── Whisper ASR ───────────────────────────────────────────────────────────
  /// Send audio file to Whisper endpoint for transcription
  Future<String> transcribeAudio(String filePath, String lang) async {
    final request = http.MultipartRequest(
      'POST',
      Uri.parse('$_serverUrl/api/transcribe'),
    );
    request.files.add(await http.MultipartFile.fromPath('audio', filePath));
    request.fields['language'] = lang; // 'en' or 'bn'

    final streamed = await request.send().timeout(const Duration(seconds: 30));
    final res = await http.Response.fromStream(streamed);

    if (res.statusCode != 200) {
      throw Exception('Transcribe failed: ${res.statusCode}');
    }

    final data = jsonDecode(res.body) as Map<String, dynamic>;
    return data['text'] as String? ?? '';
  }

  // ── Fetch keypoints for a word ────────────────────────────────────────────
  /// Fetch pre-extracted keypoint array for avatar playback
  /// Falls back to bundled assets if server unavailable
  Future<List<List<double>>?> fetchWordKeypoints(String word) async {
    try {
      final res = await http
          .get(Uri.parse('$_serverUrl/api/keypoints/$word?language=asl_en'))
          .timeout(const Duration(seconds: 5));

      if (res.statusCode != 200) return null;

      final data = jsonDecode(res.body) as Map<String, dynamic>;
      final frames = data['frames'] as List;
      return frames
          .map((f) => (f as List).map((v) => (v as num).toDouble()).toList())
          .toList();
    } catch (_) {
      return null;
    }
  }
}

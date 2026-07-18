import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:camera/camera.dart';
import '../../core/api_client.dart';
import '../../shared/theme/app_theme.dart';
import '../features/chat/chat_controller.dart';

// ─────────────────────────────────────────────────────────────────────────────
// Signer Panel
// Records a short clip of one sign, uploads it to the server for recognition,
// and shows the predicted word so it can be sent to chat.
// ─────────────────────────────────────────────────────────────────────────────

const _kClipSeconds = 3; // auto-stop after this many seconds

class SignerPanel extends StatefulWidget {
  const SignerPanel({super.key});

  @override
  State<SignerPanel> createState() => _SignerPanelState();
}

class _SignerPanelState extends State<SignerPanel> {
  CameraController? _cam;
  bool _ready = false;
  String? _error;

  bool _recording = false;
  bool _processing = false;
  int _countdown = _kClipSeconds;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() => _error = 'No camera found');
        return;
      }
      final cam = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );
      _cam = CameraController(cam, ResolutionPreset.medium, enableAudio: false);
      await _cam!.initialize();
      if (mounted) setState(() => _ready = true);
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    }
  }

  @override
  void dispose() {
    _cam?.dispose();
    super.dispose();
  }

  // ── Record → upload → predict ──────────────────────────────────────────────
  Future<void> _recordAndRecognize() async {
    if (!_ready || _recording || _processing) return;

    final chat = context.read<ChatController>();
    final api = context.read<ApiClient>();

    try {
      await _cam!.startVideoRecording();
      setState(() {
        _recording = true;
        _countdown = _kClipSeconds;
      });

      // Simple countdown while recording
      for (int s = _kClipSeconds; s > 0; s--) {
        await Future.delayed(const Duration(seconds: 1));
        if (!mounted) return;
        setState(() => _countdown = s - 1);
      }

      final file = await _cam!.stopVideoRecording();
      if (!mounted) return;
      setState(() {
        _recording = false;
        _processing = true;
      });

      // readAsBytes() works on every platform, unlike file paths on web
      final bytes = await file.readAsBytes();

      final result = await api.recognizeVideo(
        videoBytes: bytes,
        language: chat.currentLanguage,
        filename: file.name.isNotEmpty ? file.name : 'clip.mp4',
      );

      if (!mounted) return;
      chat.updateSignPrediction(result.word, result.confidence);
      setState(() => _processing = false);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _recording = false;
        _processing = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Recognition failed: $e'),
          backgroundColor: AppColors.surface2,
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final chat = context.watch<ChatController>();
    final hasPrediction = chat.lastRecognizedSign != null;

    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.all(10),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              'Signer Camera',
              style: Theme.of(context).textTheme.labelSmall,
            ),
            const SizedBox(height: 8),

            // Camera preview
            Expanded(
              flex: 3,
              child: ClipRRect(
                borderRadius: BorderRadius.circular(10),
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    _buildPreview(),
                    if (_recording)
                      Positioned(
                        top: 8,
                        left: 8,
                        child: _Badge(
                          color: AppColors.red,
                          label: 'REC  $_countdown',
                        ),
                      ),
                    if (_processing)
                      Container(
                        color: Colors.black54,
                        child: const Center(
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              CircularProgressIndicator(
                                color: AppColors.accent,
                                strokeWidth: 2,
                              ),
                              SizedBox(height: 10),
                              Text(
                                'Recognizing…',
                                style: TextStyle(
                                  color: Colors.white,
                                  fontSize: 12,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 10),

            // Prediction card
            _PredictionCard(
              word: chat.lastRecognizedSign,
              confidence: chat.lastSignConfidence,
            ),

            const SizedBox(height: 8),
            ClipRRect(
              borderRadius: BorderRadius.circular(2),
              child: LinearProgressIndicator(
                value: hasPrediction ? chat.lastSignConfidence : 0,
                backgroundColor: AppColors.border,
                valueColor: const AlwaysStoppedAnimation<Color>(
                  AppColors.accent,
                ),
                minHeight: 3,
              ),
            ),
            const SizedBox(height: 10),

            // Record button
            ElevatedButton.icon(
              onPressed: (_ready && !_recording && !_processing)
                  ? _recordAndRecognize
                  : null,
              icon: Icon(
                _recording ? Icons.fiber_manual_record : Icons.videocam,
                size: 18,
              ),
              label: Text(
                _recording
                    ? 'Recording… $_countdown'
                    : _processing
                    ? 'Recognizing…'
                    : 'Record Sign ($_kClipSeconds s)',
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: _recording
                    ? AppColors.red
                    : AppColors.accentPurple,
                disabledBackgroundColor: AppColors.border,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 12),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
            ),

            const SizedBox(height: 8),

            // Send to chat
            ElevatedButton(
              onPressed: hasPrediction ? chat.sendSignMessage : null,
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.accent,
                disabledBackgroundColor: AppColors.border,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 11),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
              child: const Text(
                'Send to Chat',
                style: TextStyle(fontSize: 13, fontWeight: FontWeight.w500),
              ),
            ),

            const SizedBox(height: 6),
            Text(
              'Sign one word, then tap Record.\nServer: ${context.watch<ApiClient>().serverUrl}',
              style: const TextStyle(
                fontSize: 10,
                color: AppColors.textThird,
                height: 1.4,
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPreview() {
    if (_error != null) {
      return Container(
        color: AppColors.signerBg,
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(
                  Icons.videocam_off_rounded,
                  color: AppColors.textThird,
                  size: 36,
                ),
                const SizedBox(height: 8),
                Text(
                  _error!,
                  style: const TextStyle(
                    color: AppColors.textThird,
                    fontSize: 10,
                  ),
                  textAlign: TextAlign.center,
                ),
              ],
            ),
          ),
        ),
      );
    }
    if (!_ready) {
      return Container(
        color: AppColors.signerBg,
        child: const Center(
          child: CircularProgressIndicator(
            color: AppColors.accent,
            strokeWidth: 1.5,
          ),
        ),
      );
    }
    return CameraPreview(_cam!);
  }
}

class _Badge extends StatelessWidget {
  final Color color;
  final String label;
  const _Badge({required this.color, required this.label});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.9),
        borderRadius: BorderRadius.circular(5),
      ),
      child: Text(
        label,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 10,
          fontWeight: FontWeight.w600,
          letterSpacing: 0.5,
        ),
      ),
    );
  }
}

class _PredictionCard extends StatelessWidget {
  final String? word;
  final double confidence;
  const _PredictionCard({this.word, required this.confidence});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: AppColors.surface2,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: AppColors.border, width: 0.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            'Recognized Sign',
            style: Theme.of(context).textTheme.labelSmall,
          ),
          const SizedBox(height: 4),
          Text(
            word ?? '—',
            style: TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.w600,
              color: word != null ? AppColors.accent : AppColors.textThird,
            ),
          ),
          if (word != null)
            Text(
              'Confidence: ${(confidence * 100).toStringAsFixed(0)}%',
              style: const TextStyle(fontSize: 10, color: AppColors.textThird),
            ),
        ],
      ),
    );
  }
}

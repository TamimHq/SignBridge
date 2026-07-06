import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:camera/camera.dart';
import '../../shared/theme/app_theme.dart';
import '../features/chat/chat_controller.dart';
// ─────────────────────────────────────────────────────────────────────────────
// Signer Panel
// Shows live camera feed + skeleton overlay + current sign prediction
// ─────────────────────────────────────────────────────────────────────────────

class SignerPanel extends StatefulWidget {
  const SignerPanel({super.key});

  @override
  State<SignerPanel> createState() => _SignerPanelState();
}

class _SignerPanelState extends State<SignerPanel> {
  CameraController? _cameraCtrl;
  bool _cameraReady = false;
  String? _cameraError;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() => _cameraError = 'No camera found');
        return;
      }

      // Prefer front camera
      final cam = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );

      _cameraCtrl = CameraController(
        cam,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );

      await _cameraCtrl!.initialize();

      // TODO: Start image stream and send frames to FastAPI /recognize endpoint
      // _cameraCtrl!.startImageStream((image) => _onCameraFrame(image));

      if (mounted) setState(() => _cameraReady = true);
    } catch (e) {
      if (mounted) setState(() => _cameraError = e.toString());
    }
  }

  // TODO: Send frames to server for sign recognition
  // void _onCameraFrame(CameraImage image) {
  //   // Throttle to ~10fps
  //   // Convert to JPEG, POST to /api/recognize
  //   // On response, call chat.updateSignPrediction(word, confidence)
  // }

  @override
  void dispose() {
    _cameraCtrl?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final chat = context.watch<ChatController>();
    final hasPrediction = chat.lastRecognizedSign != null;

    return Padding(
      padding: const EdgeInsets.all(10),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Section label
          Text('Signer Camera', style: Theme.of(context).textTheme.labelSmall),
          const SizedBox(height: 8),

          // Camera feed
          Expanded(
            flex: 3,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(10),
              child: Stack(
                fit: StackFit.expand,
                children: [
                  // Camera or placeholder
                  _buildCameraView(),

                  // LIVE badge
                  Positioned(
                    top: 8,
                    left: 8,
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 7,
                        vertical: 3,
                      ),
                      decoration: BoxDecoration(
                        color: AppColors.red.withOpacity(0.85),
                        borderRadius: BorderRadius.circular(5),
                      ),
                      child: const Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          _PulsingDot(color: Colors.white),
                          SizedBox(width: 4),
                          Text(
                            'LIVE',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 10,
                              fontWeight: FontWeight.w600,
                              letterSpacing: 0.5,
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

          // Confidence bar
          ClipRRect(
            borderRadius: BorderRadius.circular(2),
            child: LinearProgressIndicator(
              value: hasPrediction ? chat.lastSignConfidence : 0,
              backgroundColor: AppColors.border,
              valueColor: const AlwaysStoppedAnimation<Color>(AppColors.accent),
              minHeight: 3,
            ),
          ),

          const SizedBox(height: 10),

          // Send to chat button
          ElevatedButton(
            onPressed: hasPrediction ? chat.sendSignMessage : null,
            style: ElevatedButton.styleFrom(
              backgroundColor: AppColors.accent,
              disabledBackgroundColor: AppColors.border,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(10),
              ),
              padding: const EdgeInsets.symmetric(vertical: 11),
            ),
            child: const Text(
              'Send to Chat',
              style: TextStyle(
                color: Colors.white,
                fontSize: 13,
                fontWeight: FontWeight.w500,
              ),
            ),
          ),

          const SizedBox(height: 6),

          // Server connection note
          const Text(
            'Sign recognition requires FastAPI server\n'
            'connection (local or cloud)',
            style: TextStyle(
              fontSize: 10,
              color: AppColors.textThird,
              height: 1.4,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  Widget _buildCameraView() {
    if (_cameraError != null) {
      return Container(
        color: AppColors.signerBg,
        child: Center(
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
                'Camera unavailable',
                style: const TextStyle(
                  color: AppColors.textThird,
                  fontSize: 12,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                _cameraError!,
                style: const TextStyle(
                  color: AppColors.textThird,
                  fontSize: 10,
                ),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      );
    }

    if (!_cameraReady) {
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

    return CameraPreview(_cameraCtrl!);
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

class _PulsingDot extends StatefulWidget {
  final Color color;
  const _PulsingDot({required this.color});

  @override
  State<_PulsingDot> createState() => _PulsingDotState();
}

class _PulsingDotState extends State<_PulsingDot>
    with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;
  late Animation<double> _anim;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1000),
    )..repeat(reverse: true);
    _anim = CurvedAnimation(parent: _ctrl, curve: Curves.easeInOut);
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _anim,
      builder: (_, __) => Opacity(
        opacity: 0.4 + _anim.value * 0.6,
        child: Container(
          width: 5,
          height: 5,
          decoration: BoxDecoration(
            color: widget.color,
            shape: BoxShape.circle,
          ),
        ),
      ),
    );
  }
}

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';
import '../../shared/theme/app_theme.dart';
import '../../core/api_client.dart';
import 'chat_controller.dart';
import '../../core/models.dart';

class InputBar extends StatefulWidget {
  const InputBar({super.key});

  @override
  State<InputBar> createState() => _InputBarState();
}

class _InputBarState extends State<InputBar> {
  final _controller = TextEditingController();
  final _focusNode = FocusNode();
  final _recorder = AudioRecorder();
  bool _isRecording = false;

  @override
  void dispose() {
    _controller.dispose();
    _focusNode.dispose();
    _recorder.dispose();
    super.dispose();
  }

  Future<void> _send() async {
    final text = _controller.text.trim();
    if (text.isEmpty) return;
    _controller.clear();
    await context.read<ChatController>().sendSpeakerMessage(text);
  }

  Future<void> _toggleMic() async {
    final chat = context.read<ChatController>();

    if (_isRecording) {
      // Stop recording → send to Whisper
      final path = await _recorder.stop();
      setState(() => _isRecording = false);
      chat.setMicListening(false);

      if (path != null) {
        try {
          final api = context.read<ApiClient>();
          final lang = chat.currentLanguage.isBengali ? 'bn' : 'en';
          final transcript = await api.transcribeAudio(path, lang);
          await chat.onTranscription(transcript);
        } catch (e) {
          // If server unavailable, show the path so user knows it was recorded
          chat.onTranscription('');
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(
                content: Text('Whisper server unavailable — type your message'),
                backgroundColor: AppColors.surface2,
              ),
            );
          }
        }
      }
    } else {
      // Start recording
      final hasPermission = await _recorder.hasPermission();
      if (!hasPermission) return;

      final dir = await getTemporaryDirectory();
      final path = '${dir.path}/mic_input.wav';

      await _recorder.start(
        const RecordConfig(encoder: AudioEncoder.wav, sampleRate: 16000),
        path: path,
      );
      setState(() => _isRecording = true);
      chat.setMicListening(true);
    }
  }

  @override
  Widget build(BuildContext context) {
    final chat = context.watch<ChatController>();
    final isBn = chat.currentLanguage.isBengali;

    return Container(
      color: AppColors.surface,
      padding: const EdgeInsets.fromLTRB(14, 10, 14, 12),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              // Text field
              Expanded(
                child: TextField(
                  controller: _controller,
                  focusNode: _focusNode,
                  maxLines: null,
                  minLines: 1,
                  keyboardType: TextInputType.multiline,
                  textInputAction: TextInputAction.newline,
                  style: const TextStyle(
                    fontSize: 13,
                    color: AppColors.textPrimary,
                  ),
                  decoration: InputDecoration(
                    hintText: isBn
                        ? 'বাংলায় লিখুন... (অবতার সাইন করবে)'
                        : 'Type in English... (avatar will sign it)',
                  ),
                  onSubmitted: (_) => _send(),
                ),
              ),
              const SizedBox(width: 8),

              // Mic button
              _IconBtn(
                icon: _isRecording ? Icons.stop : Icons.mic,
                color: _isRecording ? AppColors.red : AppColors.textSecond,
                active: _isRecording,
                onTap: _toggleMic,
                tooltip: _isRecording ? 'Stop recording' : 'Record voice',
              ),
              const SizedBox(width: 8),

              // Send button
              GestureDetector(
                onTap: _send,
                child: Container(
                  height: 42,
                  padding: const EdgeInsets.symmetric(horizontal: 18),
                  decoration: BoxDecoration(
                    color: AppColors.accent,
                    borderRadius: BorderRadius.circular(10),
                  ),
                  alignment: Alignment.center,
                  child: const Text(
                    'Send',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 13,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ),
              ),
            ],
          ),

          // Hint
          const SizedBox(height: 5),
          const Text(
            'Tap 🔊 on any message to hear it read aloud',
            style: TextStyle(fontSize: 10, color: AppColors.textThird),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
}

class _IconBtn extends StatelessWidget {
  final IconData icon;
  final Color color;
  final bool active;
  final VoidCallback onTap;
  final String tooltip;

  const _IconBtn({
    required this.icon,
    required this.color,
    required this.active,
    required this.onTap,
    required this.tooltip,
  });

  @override
  Widget build(BuildContext context) {
    return Tooltip(
      message: tooltip,
      child: GestureDetector(
        onTap: onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          width: 42,
          height: 42,
          decoration: BoxDecoration(
            color: active
                ? AppColors.red.withOpacity(0.15)
                : AppColors.surface2,
            borderRadius: BorderRadius.circular(10),
            border: Border.all(
              color: active ? AppColors.red : AppColors.border,
              width: active ? 1.5 : 0.5,
            ),
          ),
          child: Icon(icon, color: color, size: 20),
        ),
      ),
    );
  }
}

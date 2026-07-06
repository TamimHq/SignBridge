import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../core/models.dart';
import '../../shared/theme/app_theme.dart';
import 'chat_controller.dart';

class ChatPanel extends StatefulWidget {
  const ChatPanel({super.key});

  @override
  State<ChatPanel> createState() => _ChatPanelState();
}

class _ChatPanelState extends State<ChatPanel> {
  final _scrollCtrl = ScrollController();

  @override
  void dispose() {
    _scrollCtrl.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollCtrl.hasClients) {
        _scrollCtrl.animateTo(
          _scrollCtrl.position.maxScrollExtent,
          duration: const Duration(milliseconds: 250),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final chat = context.watch<ChatController>();
    _scrollToBottom();

    if (chat.messages.isEmpty) {
      return const Center(
        child: Text(
          'No messages yet',
          style: TextStyle(color: AppColors.textThird, fontSize: 13),
        ),
      );
    }

    return ListView.builder(
      controller: _scrollCtrl,
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      itemCount: chat.messages.length,
      itemBuilder: (context, i) {
        final msg = chat.messages[i];
        return Padding(
          padding: const EdgeInsets.only(bottom: 8),
          child: _ChatBubble(
            message: msg,
            onSpeak: () => chat.speakMessage(msg),
          ),
        );
      },
    );
  }
}

class _ChatBubble extends StatelessWidget {
  final ChatMessage message;
  final VoidCallback onSpeak;

  const _ChatBubble({required this.message, required this.onSpeak});

  @override
  Widget build(BuildContext context) {
    final isSigner = message.source == MessageSource.signer;
    final isSystem = message.source == MessageSource.system;
    final isSpeaker = message.source == MessageSource.speaker;

    if (isSystem) {
      return Center(
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
          decoration: BoxDecoration(
            color: AppColors.surface2,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: AppColors.border),
          ),
          child: Text(
            message.text,
            style: const TextStyle(fontSize: 11, color: AppColors.textSecond),
          ),
        ),
      );
    }

    return Align(
      alignment: isSigner ? Alignment.centerLeft : Alignment.centerRight,
      child: ConstrainedBox(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.78,
        ),
        child: Column(
          crossAxisAlignment: isSigner
              ? CrossAxisAlignment.start
              : CrossAxisAlignment.end,
          children: [
            // Source label
            Padding(
              padding: const EdgeInsets.only(bottom: 3, left: 2, right: 2),
              child: Text(
                message.sourceLabel,
                style: const TextStyle(
                  fontSize: 10,
                  color: AppColors.textThird,
                ),
              ),
            ),

            // Bubble
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 13, vertical: 9),
              decoration: BoxDecoration(
                color: isSigner ? AppColors.msgSign : AppColors.msgVoice,
                borderRadius: BorderRadius.only(
                  topLeft: const Radius.circular(10),
                  topRight: const Radius.circular(10),
                  bottomLeft: Radius.circular(isSigner ? 3 : 10),
                  bottomRight: Radius.circular(isSigner ? 10 : 3),
                ),
                border: Border.all(
                  color: isSigner
                      ? const Color(0xFF1E3A6E)
                      : const Color(0xFF1E3250),
                  width: 0.5,
                ),
              ),
              child: Text(
                message.text,
                style: const TextStyle(
                  fontSize: 13,
                  color: AppColors.textPrimary,
                  height: 1.5,
                ),
              ),
            ),

            // Time + TTS button
            Padding(
              padding: const EdgeInsets.only(top: 3, left: 2, right: 2),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (isSpeaker) ...[
                    _TtsButton(onTap: onSpeak),
                    const SizedBox(width: 4),
                  ],
                  Text(
                    '${message.timeLabel} · ${message.language.label}',
                    style: const TextStyle(
                      fontSize: 10,
                      color: AppColors.textThird,
                    ),
                  ),
                  if (isSigner) ...[
                    const SizedBox(width: 4),
                    _TtsButton(onTap: onSpeak),
                  ],
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _TtsButton extends StatelessWidget {
  final VoidCallback onTap;
  const _TtsButton({required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: const Icon(
        Icons.volume_up_rounded,
        size: 14,
        color: AppColors.textThird,
      ),
    );
  }
}

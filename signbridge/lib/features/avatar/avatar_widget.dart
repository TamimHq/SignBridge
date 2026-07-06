import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../core/models.dart';
import '../../shared/theme/app_theme.dart';
import 'avatar_controller.dart';
import 'skeleton_painter.dart';

// ─────────────────────────────────────────────────────────────────────────────
// Avatar Widget
// Renders skeleton keypoints (ASL) or PNG frames (BdSL) based on current clip
// ─────────────────────────────────────────────────────────────────────────────

class AvatarWidget extends StatefulWidget {
  const AvatarWidget({super.key});

  @override
  State<AvatarWidget> createState() => _AvatarWidgetState();
}

class _AvatarWidgetState extends State<AvatarWidget>
    with SingleTickerProviderStateMixin {
  late AnimationController _idlePulse;
  late Animation<double> _pulseAnim;

  @override
  void initState() {
    super.initState();
    _idlePulse = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1800),
    )..repeat(reverse: true);
    _pulseAnim = CurvedAnimation(parent: _idlePulse, curve: Curves.easeInOut);
  }

  @override
  void dispose() {
    _idlePulse.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<AvatarController>(
      builder: (context, ctrl, _) {
        return Column(
          children: [
            // ── Header bar ───────────────────────────────────────────────────
            Padding(
              padding: const EdgeInsets.fromLTRB(14, 8, 14, 0),
              child: Row(
                children: [
                  Text(
                    'Sign Avatar',
                    style: Theme.of(context).textTheme.labelSmall,
                  ),
                  const Spacer(),
                  _StatusChip(state: ctrl.state, text: ctrl.statusText),
                ],
              ),
            ),

            // ── Main canvas + queue panel ─────────────────────────────────
            Expanded(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(14, 6, 14, 8),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    // Avatar canvas
                    AspectRatio(
                      aspectRatio: 1,
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(10),
                        child: _buildCanvas(ctrl),
                      ),
                    ),
                    const SizedBox(width: 12),
                    // Queue panel
                    Expanded(
                      child: _QueuePanel(
                        allClips: ctrl.allClips,
                        playingIdx: ctrl.playingAllClipsIdx,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        );
      },
    );
  }

  Widget _buildCanvas(AvatarController ctrl) {
    // BdSL: show PNG frame
    if (ctrl.currentPngPath != null) {
      return _PngFrameDisplay(assetPath: ctrl.currentPngPath!);
    }

    // ASL: draw skeleton from keypoints
    if (ctrl.currentKeypointFrame != null) {
      return AnimatedBuilder(
        animation: _pulseAnim,
        builder: (_, __) => CustomPaint(
          painter: SkeletonPainter(
            frame: ctrl.currentKeypointFrame!,
            alpha: ctrl.transitionAlpha,
          ),
        ),
      );
    }

    // Idle state
    return AnimatedBuilder(
      animation: _pulseAnim,
      builder: (_, __) => CustomPaint(
        painter: SkeletonIdlePainter(pulseValue: _pulseAnim.value),
      ),
    );
  }
}

// ── PNG Frame Display ─────────────────────────────────────────────────────────
class _PngFrameDisplay extends StatelessWidget {
  final String assetPath;
  const _PngFrameDisplay({required this.assetPath});

  @override
  Widget build(BuildContext context) {
    return Container(
      color: const Color(0xFF0A0C14),
      child: Image.asset(
        assetPath,
        fit: BoxFit.contain,
        errorBuilder: (_, __, ___) => const Center(
          child: Icon(
            Icons.person_outline,
            color: AppColors.textThird,
            size: 40,
          ),
        ),
      ),
    );
  }
}

// ── Status chip ───────────────────────────────────────────────────────────────
class _StatusChip extends StatelessWidget {
  final AvatarState state;
  final String text;
  const _StatusChip({required this.state, required this.text});

  @override
  Widget build(BuildContext context) {
    final color = switch (state) {
      AvatarState.playing => AppColors.accent,
      AvatarState.transitioning => AppColors.accentPurple,
      AvatarState.done => AppColors.green,
      AvatarState.idle => AppColors.textThird,
    };
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: color.withOpacity(0.12),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Text(
        text.length > 28 ? '${text.substring(0, 28)}…' : text,
        style: TextStyle(fontSize: 10, color: color),
      ),
    );
  }
}

// ── Queue Panel ───────────────────────────────────────────────────────────────
class _QueuePanel extends StatelessWidget {
  final List<PlaybackClip> allClips;
  final int playingIdx;

  const _QueuePanel({required this.allClips, required this.playingIdx});

  @override
  Widget build(BuildContext context) {
    if (allClips.isEmpty) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Sign queue', style: Theme.of(context).textTheme.labelSmall),
          const SizedBox(height: 8),
          Text(
            'Type a message to see signs',
            style: Theme.of(context).textTheme.bodySmall,
          ),
        ],
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Sign queue', style: Theme.of(context).textTheme.labelSmall),
        const SizedBox(height: 6),
        Expanded(
          child: ListView.separated(
            itemCount: allClips.length > 8 ? 8 : allClips.length,
            separatorBuilder: (_, __) => const SizedBox(height: 4),
            itemBuilder: (context, i) {
              final clip = allClips[i];
              final isPlaying = i == playingIdx;
              final isDone = i < playingIdx && clip.clipType != ClipType.skip;
              return _QueueItem(
                clip: clip,
                isPlaying: isPlaying,
                isDone: isDone,
              );
            },
          ),
        ),
        if (allClips.length > 8)
          Padding(
            padding: const EdgeInsets.only(top: 4),
            child: Text(
              '+${allClips.length - 8} more',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ),
      ],
    );
  }
}

class _QueueItem extends StatelessWidget {
  final PlaybackClip clip;
  final bool isPlaying;
  final bool isDone;

  const _QueueItem({
    required this.clip,
    required this.isPlaying,
    required this.isDone,
  });

  @override
  Widget build(BuildContext context) {
    final isSkip = clip.clipType == ClipType.skip;
    final dotColor = isSkip
        ? AppColors.red.withOpacity(0.4)
        : isPlaying
        ? AppColors.accent
        : isDone
        ? AppColors.green
        : AppColors.border;

    final bgColor = isPlaying
        ? AppColors.accent.withOpacity(0.08)
        : AppColors.surface2;

    final borderColor = isPlaying ? AppColors.accent : AppColors.border;

    return AnimatedContainer(
      duration: const Duration(milliseconds: 200),
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: borderColor, width: 0.5),
      ),
      child: Row(
        children: [
          // Status dot
          AnimatedContainer(
            duration: const Duration(milliseconds: 200),
            width: 5,
            height: 5,
            decoration: BoxDecoration(
              color: dotColor,
              shape: BoxShape.circle,
              boxShadow: isPlaying
                  ? [
                      BoxShadow(
                        color: AppColors.accent.withOpacity(0.6),
                        blurRadius: 4,
                      ),
                    ]
                  : null,
            ),
          ),
          const SizedBox(width: 7),
          // Word
          Expanded(
            child: Text(
              clip.word,
              style: TextStyle(
                fontSize: 11,
                color: isSkip
                    ? AppColors.red.withOpacity(0.5)
                    : isDone
                    ? AppColors.textThird
                    : AppColors.textPrimary,
                decoration: isSkip ? TextDecoration.lineThrough : null,
              ),
              overflow: TextOverflow.ellipsis,
            ),
          ),
          // Type badge
          if (clip.clipType == ClipType.fingerspell)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 1),
              decoration: BoxDecoration(
                color: AppColors.accentPurple.withOpacity(0.15),
                borderRadius: BorderRadius.circular(4),
              ),
              child: const Text(
                'spell',
                style: TextStyle(fontSize: 9, color: AppColors.accentPurple),
              ),
            ),
          if (isSkip)
            const Text(
              'skip',
              style: TextStyle(fontSize: 9, color: AppColors.red),
            ),
        ],
      ),
    );
  }
}

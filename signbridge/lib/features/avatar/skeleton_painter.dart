import 'package:flutter/material.dart';
import '../../core/models.dart';

// ─────────────────────────────────────────────────────────────────────────────
// Skeleton Painter — draws a single keypoint frame as a humanoid skeleton
// Used for ASL keypoint-based animation
// ─────────────────────────────────────────────────────────────────────────────

// OpenPose upper-body joint connections (indices into our 6-joint body subset)
// Body order: [R_shoulder, R_elbow, R_wrist, L_shoulder, L_elbow, L_wrist]
const _bodyConnections = [
  [0, 3], // R_shoulder ↔ L_shoulder (neck crossbar)
  [0, 1], // R_shoulder → R_elbow
  [1, 2], // R_elbow    → R_wrist
  [3, 4], // L_shoulder → L_elbow
  [4, 5], // L_elbow    → L_wrist
];

// Hand connections (21 landmarks, MediaPipe hand topology)
const _handConnections = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Palm
  [5, 9], [9, 13], [13, 17],
];

class SkeletonPainter extends CustomPainter {
  final KeypointFrame frame;
  final double alpha; // for fade transitions

  const SkeletonPainter({required this.frame, this.alpha = 1.0});

  @override
  void paint(Canvas canvas, Size size) {
    final W = size.width, H = size.height;

    // Background
    canvas.drawRect(
      Rect.fromLTWH(0, 0, W, H),
      Paint()..color = const Color(0xFF0A0C14),
    );

    // Subtle grid
    final gridPaint = Paint()
      ..color = const Color(0xFF4F6EF7).withOpacity(0.05)
      ..strokeWidth = 1;
    for (double x = 0; x < W; x += 14) {
      canvas.drawLine(Offset(x, 0), Offset(x, H), gridPaint);
    }
    for (double y = 0; y < H; y += 14) {
      canvas.drawLine(Offset(0, y), Offset(W, y), gridPaint);
    }

    // Convert normalized coords to screen coords
    // Our keypoints are normalized relative to shoulder midpoint
    // We map them into the canvas with appropriate scale/offset
    Offset toScreen(double nx, double ny) {
      // Center at 50% horizontal, 40% vertical
      // Scale so shoulder width ≈ 40% of canvas width
      return Offset(W * 0.5 + nx * W * 0.4, H * 0.4 + ny * H * 0.4);
    }

    // ── Body joints ─────────────────────────────────────────────────────────
    final bodyPts = List.generate(6, (i) {
      final (x, y, conf) = frame.bodyJoint(i);
      return (toScreen(x, y), conf);
    });

    final bodyLinePaint = Paint()
      ..color = const Color(0xFF4F6EF7).withOpacity(0.85 * alpha)
      ..strokeWidth = 2.5
      ..strokeCap = StrokeCap.round;

    for (final conn in _bodyConnections) {
      final (ptA, confA) = bodyPts[conn[0]];
      final (ptB, confB) = bodyPts[conn[1]];
      if (confA > 0.1 && confB > 0.1) {
        canvas.drawLine(ptA, ptB, bodyLinePaint);
      }
    }

    // Body joint dots
    for (int i = 0; i < bodyPts.length; i++) {
      final (pt, conf) = bodyPts[i];
      if (conf > 0.1) {
        canvas.drawCircle(
          pt,
          i == 0 ? 4.0 : 3.0,
          Paint()
            ..color = const Color(0xFF4F6EF7).withOpacity(alpha)
            ..style = PaintingStyle.fill,
        );
      }
    }

    // ── Hands ───────────────────────────────────────────────────────────────
    _drawHand(canvas, size, frame, toScreen, isLeft: true, alpha: alpha);
    _drawHand(canvas, size, frame, toScreen, isLeft: false, alpha: alpha);

    // ── Head circle ─────────────────────────────────────────────────────────
    // Estimate head position above R/L shoulder midpoint
    final (rsX, rsY, rsC) = frame.bodyJoint(0); // R shoulder
    final (lsX, lsY, lsC) = frame.bodyJoint(3); // L shoulder
    if (rsC > 0.1 && lsC > 0.1) {
      final midX = (rsX + lsX) / 2;
      final midY = (rsY + lsY) / 2;
      final headPt = toScreen(midX, midY - 0.25); // above shoulders
      canvas.drawCircle(
        headPt,
        10,
        Paint()
          ..color = const Color(0xFF4F6EF7).withOpacity(0.5 * alpha)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.5,
      );
      // Eye dots
      canvas.drawCircle(
        headPt + const Offset(-3.5, -1),
        1.5,
        Paint()..color = const Color(0xFF8B8FA8).withOpacity(alpha),
      );
      canvas.drawCircle(
        headPt + const Offset(3.5, -1),
        1.5,
        Paint()..color = const Color(0xFF8B8FA8).withOpacity(alpha),
      );
    }
  }

  void _drawHand(
    Canvas canvas,
    Size size,
    KeypointFrame frame,
    Offset Function(double, double) toScreen, {
    required bool isLeft,
    required double alpha,
  }) {
    final handLinePaint = Paint()
      ..color = const Color(0xFF7C3AED).withOpacity(0.85 * alpha)
      ..strokeWidth = 1.5
      ..strokeCap = StrokeCap.round;
    final dotPaint = Paint()
      ..color = const Color(0xFFA78BFA).withOpacity(0.8 * alpha)
      ..style = PaintingStyle.fill;

    final pts = List.generate(21, (i) {
      final (x, y, conf) = isLeft ? frame.leftHand(i) : frame.rightHand(i);
      return (toScreen(x, y), conf);
    });

    for (final conn in _handConnections) {
      final (ptA, confA) = pts[conn[0]];
      final (ptB, confB) = pts[conn[1]];
      if (confA > 0.05 && confB > 0.05) {
        canvas.drawLine(ptA, ptB, handLinePaint);
      }
    }

    for (final (pt, conf) in pts) {
      if (conf > 0.05) {
        canvas.drawCircle(pt, 2.0, dotPaint);
      }
    }
  }

  @override
  bool shouldRepaint(SkeletonPainter old) =>
      old.frame != frame || old.alpha != alpha;
}

// ─────────────────────────────────────────────────────────────────────────────
// Idle painter — shown when no animation is playing
// ─────────────────────────────────────────────────────────────────────────────
class SkeletonIdlePainter extends CustomPainter {
  final double pulseValue; // 0.0–1.0 from animation controller

  const SkeletonIdlePainter({required this.pulseValue});

  @override
  void paint(Canvas canvas, Size size) {
    final W = size.width, H = size.height;

    canvas.drawRect(
      Rect.fromLTWH(0, 0, W, H),
      Paint()..color = const Color(0xFF0A0C14),
    );

    // Pulsing circle
    final radius = 18.0 + pulseValue * 6;
    canvas.drawCircle(
      Offset(W / 2, H / 2),
      radius,
      Paint()
        ..color = const Color(0xFF4F6EF7).withOpacity(0.12 + pulseValue * 0.08)
        ..style = PaintingStyle.fill,
    );
    canvas.drawCircle(
      Offset(W / 2, H / 2),
      radius,
      Paint()
        ..color = const Color(0xFF4F6EF7).withOpacity(0.25)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.5,
    );

    // Text
    final tp = TextPainter(
      text: const TextSpan(
        text: 'Avatar idle',
        style: TextStyle(color: Color(0xFF5A5E7A), fontSize: 11),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    tp.paint(canvas, Offset(W / 2 - tp.width / 2, H / 2 + 28));
  }

  @override
  bool shouldRepaint(SkeletonIdlePainter old) => old.pulseValue != pulseValue;
}

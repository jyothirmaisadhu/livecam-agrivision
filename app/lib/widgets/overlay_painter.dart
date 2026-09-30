import 'package:flutter/material.dart';

import '../logic/engine.dart';
import '../ml/models.dart';

/// Draws the lesion mask (red) and pest boxes over an image/preview of the same aspect.
class OverlayPainter extends CustomPainter {
  OverlayPainter({this.mask, this.pests = const []});

  final SegMask? mask;
  final List<PestDetection> pests;

  @override
  void paint(Canvas canvas, Size size) {
    final m = mask;
    if (m != null) {
      const step = 4;
      final cellW = size.width / m.size * step, cellH = size.height / m.size * step;
      final lesion = Paint()..color = Colors.red.withOpacity(0.45);
      for (var y = 0; y < m.size; y += step) {
        for (var x = 0; x < m.size; x += step) {
          if (m.values[y * m.size + x] == 2) {
            canvas.drawRect(Rect.fromLTWH(x / step * cellW, y / step * cellH, cellW, cellH), lesion);
          }
        }
      }
    }
    final box = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.5
      ..color = Colors.orangeAccent;
    for (final p in pests) {
      final r = Rect.fromLTRB(
          p.left * size.width, p.top * size.height, p.right * size.width, p.bottom * size.height);
      canvas.drawRect(r, box);
      final tp = TextPainter(
        text: TextSpan(
          text: ' ${p.label} ${(p.confidence * 100).round()}% ',
          style: const TextStyle(color: Colors.black, fontSize: 12, backgroundColor: Colors.orangeAccent),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, Offset(r.left, (r.top - tp.height).clamp(0, size.height)));
    }
  }

  @override
  bool shouldRepaint(OverlayPainter oldDelegate) => oldDelegate.mask != mask || oldDelegate.pests != pests;
}

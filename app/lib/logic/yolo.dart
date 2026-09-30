import 'dart:math' as math;
import 'dart:typed_data';

import 'engine.dart';

/// Decodes an Ultralytics YOLO TFLite output of shape [1, 4 + numClasses, numAnchors]
/// (rows: cx, cy, w, h, class scores) into boxes normalised to 0..1, then applies per-class NMS.
List<PestDetection> decodeYolo(
  Float32List out,
  List<String> labels, {
  required int numAnchors,
  required int inputSize,
  double confidence = 0.35,
  double iou = 0.45,
  int maxDetections = 50,
}) {
  final numClasses = labels.length;
  assert(out.length == (4 + numClasses) * numAnchors, 'unexpected YOLO output size');
  var maxCoord = 0.0;
  for (var i = 0; i < 4 * numAnchors; i++) {
    maxCoord = math.max(maxCoord, out[i]);
  }
  final scale = maxCoord > 2.0 ? 1.0 / inputSize : 1.0; // some exports emit pixels, others 0..1

  final candidates = <PestDetection>[];
  for (var a = 0; a < numAnchors; a++) {
    var best = -1;
    var bestScore = confidence;
    for (var c = 0; c < numClasses; c++) {
      final s = out[(4 + c) * numAnchors + a];
      if (s >= bestScore) {
        bestScore = s;
        best = c;
      }
    }
    if (best < 0) continue;
    final cx = out[a] * scale, cy = out[numAnchors + a] * scale;
    final w = out[2 * numAnchors + a] * scale, h = out[3 * numAnchors + a] * scale;
    candidates.add(PestDetection(
      labels[best],
      bestScore,
      (cx - w / 2).clamp(0.0, 1.0),
      (cy - h / 2).clamp(0.0, 1.0),
      (cx + w / 2).clamp(0.0, 1.0),
      (cy + h / 2).clamp(0.0, 1.0),
    ));
  }
  return nms(candidates, iou: iou, maxDetections: maxDetections);
}

double boxIou(PestDetection a, PestDetection b) {
  final iw = math.max(0.0, math.min(a.right, b.right) - math.max(a.left, b.left));
  final ih = math.max(0.0, math.min(a.bottom, b.bottom) - math.max(a.top, b.top));
  final inter = iw * ih;
  final union = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - inter;
  return union <= 0 ? 0 : inter / union;
}

List<PestDetection> nms(List<PestDetection> boxes, {double iou = 0.45, int maxDetections = 50}) {
  final sorted = [...boxes]..sort((a, b) => b.confidence.compareTo(a.confidence));
  final kept = <PestDetection>[];
  for (final box in sorted) {
    if (kept.length >= maxDetections) break;
    if (kept.every((k) => k.label != box.label || boxIou(k, box) < iou)) kept.add(box);
  }
  return kept;
}

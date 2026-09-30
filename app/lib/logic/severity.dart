import 'dart:typed_data';

/// Mask values produced by the U-Net.
const int maskBackground = 0;
const int maskLeaf = 1;
const int maskLesion = 2;

class SeverityLevels {
  const SeverityLevels({this.healthy = 1.0, this.mild = 10.0, this.moderate = 25.0});

  final double healthy;
  final double mild;
  final double moderate;

  String levelOf(double percent) {
    if (percent < healthy) return 'healthy';
    if (percent <= mild) return 'mild';
    if (percent <= moderate) return 'moderate';
    return 'severe';
  }
}

class Severity {
  const Severity(this.percent, this.level, this.leafPixels, this.lesionPixels);

  final double percent;
  final String level;
  final int leafPixels;
  final int lesionPixels;
}

/// Severity % = lesion pixels / (leaf pixels + lesion pixels) x 100.
Severity severityFromMask(Uint8List mask, {SeverityLevels levels = const SeverityLevels()}) {
  var leaf = 0;
  var lesion = 0;
  for (final v in mask) {
    if (v == maskLeaf) {
      leaf++;
    } else if (v == maskLesion) {
      lesion++;
    }
  }
  final total = leaf + lesion;
  final percent = total == 0 ? 0.0 : lesion * 100.0 / total;
  return Severity(percent, levels.levelOf(percent), leaf, lesion);
}

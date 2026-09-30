import 'severity.dart';

/// Mirrors configs/settings.yaml (exported to assets/knowledge_base/app_config.json).
class LiveSettings {
  const LiveSettings({
    this.processEveryNFrames = 6,
    this.smoothingWindow = 5,
    this.diseaseConfidence = 0.60,
    this.pestConfidence = 0.35,
    this.speakCooldownSeconds = 8,
  });

  factory LiveSettings.fromJson(Map<String, dynamic> j) => LiveSettings(
        processEveryNFrames: j['process_every_n_frames'] as int,
        smoothingWindow: j['smoothing_window'] as int,
        diseaseConfidence: (j['disease_confidence'] as num).toDouble(),
        pestConfidence: (j['pest_confidence'] as num).toDouble(),
        speakCooldownSeconds: (j['speak_cooldown_seconds'] as num).toInt(),
      );

  final int processEveryNFrames;
  final int smoothingWindow;
  final double diseaseConfidence;
  final double pestConfidence;
  final int speakCooldownSeconds;
}

class AppSettings {
  const AppSettings({this.live = const LiveSettings(), this.severity = const SeverityLevels()});

  factory AppSettings.fromJson(Map<String, dynamic> j) {
    final s = j['severity_levels'] as Map<String, dynamic>;
    return AppSettings(
      live: LiveSettings.fromJson(j['live'] as Map<String, dynamic>),
      severity: SeverityLevels(
        healthy: (s['healthy'] as num).toDouble(),
        mild: (s['mild'] as num).toDouble(),
        moderate: (s['moderate'] as num).toDouble(),
      ),
    );
  }

  final LiveSettings live;
  final SeverityLevels severity;
}

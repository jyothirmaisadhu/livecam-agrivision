import 'package:image/image.dart' as img;

import '../logic/engine.dart';
import '../logic/knowledge_base.dart';
import '../logic/settings.dart';
import '../logic/severity.dart';
import '../logic/smoothing.dart';
import 'models.dart';

class AnalysisResult {
  const AnalysisResult({required this.report, this.pests = const [], this.mask, this.severity, this.raw});

  final Report report;
  final List<PestDetection> pests;
  final SegMask? mask;
  final Severity? severity;
  final DiseaseResult? raw;
}

/// Same flow as agrivision.pipeline.live.LivePipeline.process.
class Analyzer {
  Analyzer(this.models, this.kb, this.settings);

  final AgriModels models;
  final KnowledgeBase kb;
  final AppSettings settings;

  /// Pass [vote] for live video (label must win the majority vote); omit it for a single photo.
  AnalysisResult analyze(img.Image image, {MajorityVote? vote}) {
    final threshold = settings.live.diseaseConfidence;
    DiseaseResult? raw, disease;
    if (models.disease != null) {
      raw = models.disease!.classify(image);
      final confident = raw.confidence >= threshold ? raw.label : null;
      if (vote == null) {
        disease = confident == null ? null : raw;
      } else {
        disease = vote.update(confident) == raw.label ? raw : null;
      }
    }
    final pests = models.pest?.detect(image) ?? const <PestDetection>[];
    SegMask? mask;
    Severity? severity;
    if (models.damage != null && disease != null && !disease.isHealthy) {
      mask = models.damage!.segment(image);
      severity = severityFromMask(mask.values, levels: settings.severity);
    }
    final report =
        buildReport(kb, disease, pests, severityPercent: severity?.percent, severityLevel: severity?.level);
    return AnalysisResult(report: report, pests: pests, mask: mask, severity: severity, raw: raw);
  }
}

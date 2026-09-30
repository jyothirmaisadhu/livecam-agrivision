import 'knowledge_base.dart';

class DiseaseResult {
  const DiseaseResult(this.label, this.confidence);

  final String label;
  final double confidence;

  bool get isHealthy => label.endsWith('_healthy');
}

class PestDetection {
  const PestDetection(this.label, this.confidence, this.left, this.top, this.right, this.bottom);

  final String label;
  final double confidence;

  /// Box corners normalised to 0..1 of the analysed image.
  final double left, top, right, bottom;
}

class Advice {
  const Advice(this.title, this.detail, {this.treatment = '', this.prevention = ''});

  final String title;
  final String detail;
  final String treatment;
  final String prevention;
}

class Report {
  const Report({
    required this.crop,
    required this.headline,
    required this.items,
    required this.spoken,
    this.severityPercent,
    this.severityLevel,
    this.treatmentSpoken = '',
  });

  final String? crop;
  final String headline;
  final double? severityPercent;
  final String? severityLevel;
  final List<Advice> items;
  final String spoken;
  final String treatmentSpoken;
}

String _title(String s) => s.isEmpty ? s : s[0].toUpperCase() + s.substring(1);

/// Dart port of agrivision.decision.engine.build_report.
Report buildReport(
  KnowledgeBase kb,
  DiseaseResult? disease,
  List<PestDetection> pests, {
  double? severityPercent,
  String? severityLevel,
}) {
  final crop = disease == null ? null : cropOf(disease.label);
  final relevant = crop == null ? null : kb.cropPests[crop] ?? <String>{};
  final items = <Advice>[];
  final spoken = <String>[];
  final treatments = <String>[];
  String headline;

  if (disease == null) {
    headline = 'No confident disease result';
  } else if (kb.healthyMessages.containsKey(disease.label)) {
    headline = kb.healthyMessages[disease.label]!;
    spoken.add(headline);
  } else {
    final info = kb.diseases[disease.label]!;
    headline = '${_title(crop!)}: ${info.name} (${(disease.confidence * 100).round()}%)';
    items.add(Advice(info.name, info.symptoms, treatment: info.treatment, prevention: info.prevention));
    spoken.add('${info.name} detected on $crop.');
    if (severityLevel != null && severityLevel != 'healthy' && severityPercent != null) {
      spoken.add('Severity is $severityLevel, about ${severityPercent.round()} percent of the leaf.');
      items.add(Advice('Severity: $severityLevel', kb.severityAdvice[severityLevel] ?? ''));
    }
    spoken.add('Treatment: ${info.treatment}');
    treatments.add('For ${info.name}: ${info.treatment}');
  }

  final counts = <String, int>{};
  for (final p in pests) {
    counts[p.label] = (counts[p.label] ?? 0) + 1;
  }
  final ordered = counts.entries.toList()..sort((a, b) => b.value.compareTo(a.value));
  for (final e in ordered) {
    final info = kb.pests[e.key];
    if (info == null) continue;
    final note = relevant == null || relevant.contains(e.key) ? '' : ' (not a common pest of $crop)';
    items.add(Advice('Pest: ${info.name} x${e.value}$note', info.symptoms,
        treatment: info.treatment, prevention: info.prevention));
    spoken.add('${e.value} ${info.name} found. ${info.treatment}');
    treatments.add('For ${info.name}: ${info.treatment}');
    for (final linked in kb.vectorLinks[e.key] ?? const <String>[]) {
      if (crop == null || cropOf(linked) == crop) {
        final virus = kb.diseases[linked]?.name ?? linked;
        items.add(Advice('Virus risk', '${info.name} can spread $virus. Control it early.'));
      }
    }
  }

  if (spoken.isEmpty) spoken.add('Point the camera at a leaf, closer and in good light.');
  return Report(
    crop: crop,
    headline: headline,
    severityPercent: severityPercent,
    severityLevel: severityLevel,
    items: items,
    spoken: spoken.join(' '),
    treatmentSpoken: treatments.isEmpty ? 'No treatment needed right now.' : treatments.join(' '),
  );
}

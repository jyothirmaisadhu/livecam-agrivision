class KbEntry {
  const KbEntry({required this.name, this.symptoms = '', this.treatment = '', this.prevention = ''});

  factory KbEntry.fromJson(Map<String, dynamic> j) => KbEntry(
        name: j['name'] as String,
        symptoms: (j['symptoms'] ?? '') as String,
        treatment: (j['treatment'] ?? '') as String,
        prevention: (j['prevention'] ?? '') as String,
      );

  final String name;
  final String symptoms;
  final String treatment;
  final String prevention;
}

/// Mirror of knowledge_base/recommendations.json plus the crop -> pest table from configs/classes.yaml.
class KnowledgeBase {
  KnowledgeBase({
    required this.disclaimer,
    required this.diseases,
    required this.pests,
    required this.healthyMessages,
    required this.severityAdvice,
    required this.vectorLinks,
    required this.cropPests,
  });

  factory KnowledgeBase.fromJson(Map<String, dynamic> kb, Map<String, dynamic> cropPests) {
    Map<String, KbEntry> entries(String key) => (kb[key] as Map<String, dynamic>)
        .map((k, v) => MapEntry(k, KbEntry.fromJson(v as Map<String, dynamic>)));
    return KnowledgeBase(
      disclaimer: kb['disclaimer'] as String,
      diseases: entries('diseases'),
      pests: entries('pests'),
      healthyMessages: (kb['healthy'] as Map<String, dynamic>)
          .map((k, v) => MapEntry(k, (v as Map<String, dynamic>)['message'] as String)),
      severityAdvice: (kb['severity_advice'] as Map<String, dynamic>).cast<String, String>(),
      vectorLinks:
          (kb['vector_links'] as Map<String, dynamic>).map((k, v) => MapEntry(k, (v as List).cast<String>())),
      cropPests: cropPests.map((k, v) => MapEntry(k, (v as List).cast<String>().toSet())),
    );
  }

  final String disclaimer;
  final Map<String, KbEntry> diseases;
  final Map<String, KbEntry> pests;
  final Map<String, String> healthyMessages;
  final Map<String, String> severityAdvice;
  final Map<String, List<String>> vectorLinks;
  final Map<String, Set<String>> cropPests;
}

String cropOf(String diseaseClass) => diseaseClass.split('_').first;

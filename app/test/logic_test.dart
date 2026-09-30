import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:agrivision_app/logic/engine.dart';
import 'package:agrivision_app/logic/knowledge_base.dart';
import 'package:agrivision_app/logic/settings.dart';
import 'package:agrivision_app/logic/severity.dart';
import 'package:agrivision_app/logic/smoothing.dart';
import 'package:agrivision_app/logic/yolo.dart';
import 'package:agrivision_app/services/voice_commands.dart';
import 'package:flutter_test/flutter_test.dart';

Map<String, dynamic> _json(String path) => jsonDecode(File(path).readAsStringSync()) as Map<String, dynamic>;

void main() {
  final config = _json('assets/knowledge_base/app_config.json');
  final kb = KnowledgeBase.fromJson(
    _json('assets/knowledge_base/recommendations.json'),
    config['crop_pests'] as Map<String, dynamic>,
  );

  group('severity', () {
    test('uses leaf area, not the whole image', () {
      final mask = Uint8List(100); // 50 background, 40 leaf, 10 lesion
      mask.fillRange(50, 90, maskLeaf);
      mask.fillRange(90, 100, maskLesion);
      final s = severityFromMask(mask);
      expect(s.percent, closeTo(20.0, 1e-9));
      expect(s.level, 'moderate');
    });

    test('levels follow settings thresholds', () {
      final levels = AppSettings.fromJson(config).severity;
      expect(levels.levelOf(0.5), 'healthy');
      expect(levels.levelOf(10), 'mild');
      expect(levels.levelOf(25), 'moderate');
      expect(levels.levelOf(26), 'severe');
      expect(severityFromMask(Uint8List(10)).percent, 0);
    });
  });

  test('majority vote needs a strict majority', () {
    final vote = MajorityVote(window: 5);
    expect(vote.update('a'), 'a');
    expect(vote.update('b'), isNull);
    expect(vote.update('b'), 'b');
    expect(vote.update(null), isNull);
    expect(vote.update('b'), 'b');
  });

  group('report', () {
    test('disease + severity + pests with vector warning', () {
      final r = buildReport(
        kb,
        const DiseaseResult('chilli_leaf_curl', 0.91),
        const [
          PestDetection('whitefly', 0.8, 0, 0, 0.1, 0.1),
          PestDetection('whitefly', 0.7, 0.2, 0.2, 0.3, 0.3),
          PestDetection('stem_borer', 0.6, 0.5, 0.5, 0.6, 0.6),
        ],
        severityPercent: 18,
        severityLevel: 'moderate',
      );
      expect(r.crop, 'chilli');
      expect(r.headline, startsWith('Chilli: '));
      expect(r.items.any((a) => a.title.startsWith('Severity: moderate')), isTrue);
      expect(r.items.any((a) => a.title.contains('x2')), isTrue);
      expect(r.items.any((a) => a.title.contains('not a common pest of chilli')), isTrue);
      expect(r.items.any((a) => a.title == 'Virus risk'), isTrue);
      expect(r.spoken, contains('about 18 percent'));
      expect(r.treatmentSpoken, contains('For '));
    });

    test('healthy and empty results', () {
      expect(buildReport(kb, const DiseaseResult('tomato_healthy', 0.9), const []).headline,
          'Tomato leaf looks healthy.');
      final none = buildReport(kb, null, const []);
      expect(none.headline, 'No confident disease result');
      expect(none.spoken, contains('Point the camera'));
    });

    test('every disease label has knowledge base text', () {
      for (final crop in kb.cropPests.keys) {
        expect(kb.healthyMessages.containsKey('${crop}_healthy'), isTrue, reason: crop);
      }
      for (final pests in kb.cropPests.values) {
        for (final p in pests) {
          expect(kb.pests.containsKey(p), isTrue, reason: p);
        }
      }
    });
  });

  group('yolo decode', () {
    const labels = ['aphids', 'thrips'];
    Float32List output(List<List<double>> anchors) {
      final n = anchors.length;
      final out = Float32List(6 * n);
      for (var a = 0; a < n; a++) {
        for (var r = 0; r < 6; r++) {
          out[r * n + a] = anchors[a][r];
        }
      }
      return out;
    }

    test('normalised output, NMS removes overlaps, low scores dropped', () {
      final out = output([
        [0.5, 0.5, 0.2, 0.2, 0.9, 0.1],
        [0.51, 0.5, 0.2, 0.2, 0.8, 0.1],
        [0.2, 0.2, 0.1, 0.1, 0.1, 0.7],
        [0.8, 0.8, 0.1, 0.1, 0.1, 0.2],
      ]);
      final d = decodeYolo(out, labels, numAnchors: 4, inputSize: 640);
      expect(d.map((e) => e.label), ['aphids', 'thrips']);
      expect(d.first.left, closeTo(0.4, 1e-6));
      expect(d.first.bottom, closeTo(0.6, 1e-6));
    });

    test('pixel output is rescaled', () {
      final out = output([
        [320, 320, 64, 64, 0.9, 0.0],
      ]);
      final d = decodeYolo(out, labels, numAnchors: 1, inputSize: 640);
      expect(d.single.left, closeTo(0.45, 1e-6));
    });
  });

  test('voice commands', () {
    expect(parseCommand('Please analyze this leaf'), VoiceCommand.analyze);
    expect(parseCommand('which spray should I use'), VoiceCommand.treatment);
    expect(parseCommand('say that again'), VoiceCommand.repeat);
    expect(parseCommand('STOP'), VoiceCommand.stop);
    expect(parseCommand('open the live camera'), VoiceCommand.live);
    expect(parseCommand('banana'), VoiceCommand.unknown);
  });
}

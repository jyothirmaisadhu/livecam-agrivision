import 'dart:convert';

import 'package:flutter/services.dart';

import '../logic/knowledge_base.dart';
import '../logic/settings.dart';
import '../ml/analyzer.dart';
import '../ml/models.dart';
import 'voice.dart';

class AppServices {
  AppServices(this.kb, this.settings, this.models, this.voice) : analyzer = Analyzer(models, kb, settings);

  final KnowledgeBase kb;
  final AppSettings settings;
  final AgriModels models;
  final VoiceService voice;
  final Analyzer analyzer;

  static Future<AppServices> load() async {
    final config = jsonDecode(await rootBundle.loadString('assets/knowledge_base/app_config.json'))
        as Map<String, dynamic>;
    final kbJson = jsonDecode(await rootBundle.loadString('assets/knowledge_base/recommendations.json'))
        as Map<String, dynamic>;
    final settings = AppSettings.fromJson(config);
    final kb = KnowledgeBase.fromJson(kbJson, config['crop_pests'] as Map<String, dynamic>);
    final models = await AgriModels.load(pestConfidence: settings.live.pestConfidence);
    final voice = VoiceService();
    await voice.init();
    return AppServices(kb, settings, models, voice);
  }
}

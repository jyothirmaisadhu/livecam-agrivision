import 'package:flutter_tts/flutter_tts.dart';
import 'package:speech_to_text/speech_to_text.dart';

import 'voice_commands.dart';

/// Text-to-speech for advice plus short voice commands (flutter_tts + speech_to_text).
class VoiceService {
  final FlutterTts _tts = FlutterTts();
  final SpeechToText _stt = SpeechToText();
  bool listeningAvailable = false;
  bool enabled = true;
  String lastSpoken = '';
  DateTime _lastTime = DateTime.fromMillisecondsSinceEpoch(0);

  Future<void> init() async {
    try {
      await _tts.setLanguage('en-IN');
      await _tts.setSpeechRate(0.45);
      listeningAvailable = await _stt.initialize();
    } catch (_) {
      listeningAvailable = false;
    }
  }

  /// Speaks [text]. Unless [force], nothing new is spoken within [cooldown] of the last sentence.
  Future<void> speak(String text,
      {bool force = false, Duration cooldown = const Duration(seconds: 8)}) async {
    if (!enabled && !force) return;
    final now = DateTime.now();
    if (!force && now.difference(_lastTime) < cooldown) return;
    lastSpoken = text;
    _lastTime = now;
    await _tts.stop();
    await _tts.speak(text);
  }

  Future<void> stop() => _tts.stop();

  bool get isListening => _stt.isListening;

  Future<void> listen(void Function(VoiceCommand command, String words) onCommand) async {
    if (!listeningAvailable) return;
    await _tts.stop();
    await _stt.listen(
      onResult: (r) {
        if (r.finalResult) onCommand(parseCommand(r.recognizedWords), r.recognizedWords);
      },
      listenOptions: SpeechListenOptions(
        listenFor: const Duration(seconds: 6),
        partialResults: false,
        localeId: 'en_IN',
      ),
    );
  }
}

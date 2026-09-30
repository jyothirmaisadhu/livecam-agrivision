enum VoiceCommand { analyze, repeat, treatment, stop, live, photo, help, unknown }

const _keywords = <VoiceCommand, List<String>>{
  VoiceCommand.stop: ['stop', 'quiet', 'silent', 'cancel'],
  VoiceCommand.treatment: ['treatment', 'medicine', 'spray', 'remedy', 'cure', 'what should i do'],
  VoiceCommand.repeat: ['repeat', 'again', 'say that'],
  VoiceCommand.analyze: ['analyze', 'analyse', 'scan', 'check', 'detect'],
  VoiceCommand.live: ['live', 'camera', 'video'],
  VoiceCommand.photo: ['photo', 'picture', 'gallery', 'image'],
  VoiceCommand.help: ['help', 'what can you do'],
};

/// Maps a recognised sentence to a command; earlier entries win when several keywords match.
VoiceCommand parseCommand(String words) {
  final text = words.toLowerCase();
  for (final entry in _keywords.entries) {
    if (entry.value.any(text.contains)) return entry.key;
  }
  return VoiceCommand.unknown;
}

const helpText = 'Say live camera, photo, analyze, repeat, treatment, or stop.';

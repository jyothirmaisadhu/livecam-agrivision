import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../services/app_services.dart';
import '../services/voice_commands.dart';
import 'live_screen.dart';
import 'photo_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key, required this.services});

  final AppServices services;

  void _open(BuildContext context, Widget screen) =>
      Navigator.of(context).push(MaterialPageRoute(builder: (_) => screen));

  void _onCommand(BuildContext context, VoiceCommand command) {
    switch (command) {
      case VoiceCommand.live || VoiceCommand.analyze:
        _open(context, LiveScreen(services: services));
      case VoiceCommand.photo:
        _open(context, PhotoScreen(services: services));
      default:
        services.voice.speak(helpText, force: true);
    }
  }

  @override
  Widget build(BuildContext context) {
    final m = services.models;
    final status = {
      'Disease (MobileNetV3)': m.disease != null,
      'Pests (YOLO11n)': m.pest != null,
      'Damage (U-Net)': m.damage != null,
    };
    return Scaffold(
      appBar: AppBar(title: const Text('LiveCam AgriVision')),
      floatingActionButton: services.voice.listeningAvailable
          ? FloatingActionButton.extended(
              onPressed: () => services.voice.listen((c, _) => _onCommand(context, c)),
              icon: const Icon(Icons.mic),
              label: const Text('Speak'),
            )
          : null,
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Text('Detect crop diseases and pests, see how much of the leaf is damaged, and hear what to do.',
              style: Theme.of(context).textTheme.bodyLarge),
          const SizedBox(height: 24),
          FilledButton.icon(
            style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(64)),
            onPressed: () => _open(context, LiveScreen(services: services)),
            icon: const Icon(Icons.videocam),
            label: const Text('Live camera scan'),
          ),
          const SizedBox(height: 12),
          OutlinedButton.icon(
            style: OutlinedButton.styleFrom(minimumSize: const Size.fromHeight(64)),
            onPressed: () => _open(context, PhotoScreen(services: services)),
            icon: const Icon(Icons.photo_camera),
            label: const Text('Take a photo'),
          ),
          const SizedBox(height: 12),
          OutlinedButton.icon(
            style: OutlinedButton.styleFrom(minimumSize: const Size.fromHeight(64)),
            onPressed: () => _open(context, PhotoScreen(services: services, source: ImageSource.gallery)),
            icon: const Icon(Icons.photo_library),
            label: const Text('Choose from gallery'),
          ),
          const SizedBox(height: 24),
          Text('Models', style: Theme.of(context).textTheme.titleMedium),
          for (final e in status.entries)
            ListTile(
              dense: true,
              leading: Icon(e.value ? Icons.check_circle : Icons.error_outline,
                  color: e.value ? Colors.green : Colors.orange),
              title: Text(e.key),
              subtitle: Text(e.value ? 'Loaded' : 'Not bundled - run scripts/export_app_assets.py'),
            ),
        ],
      ),
    );
  }
}

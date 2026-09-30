import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';

import '../ml/analyzer.dart';
import '../services/app_services.dart';
import '../widgets/overlay_painter.dart';
import '../widgets/report_card.dart';

/// Analyse a single photo taken with the camera or chosen from the gallery.
class PhotoScreen extends StatefulWidget {
  const PhotoScreen({super.key, required this.services, this.source = ImageSource.camera});

  final AppServices services;
  final ImageSource source;

  @override
  State<PhotoScreen> createState() => _PhotoScreenState();
}

class _PhotoScreenState extends State<PhotoScreen> {
  Uint8List? _bytes;
  double _aspect = 1;
  AnalysisResult? _result;
  bool _working = false;
  String? _error;

  AppServices get s => widget.services;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _pick(widget.source));
  }

  Future<void> _pick(ImageSource source) async {
    final file = await ImagePicker().pickImage(source: source, maxWidth: 1280);
    if (file == null) return;
    setState(() {
      _working = true;
      _error = null;
    });
    final bytes = await file.readAsBytes();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) {
      setState(() {
        _working = false;
        _error = 'Could not read that image.';
      });
      return;
    }
    final image = img.bakeOrientation(decoded);
    final result = s.analyzer.analyze(image);
    setState(() {
      _bytes = bytes;
      _aspect = image.width / image.height;
      _result = result;
      _working = false;
    });
    s.voice.speak(result.report.spoken, force: true);
  }

  @override
  void dispose() {
    s.voice.stop();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final result = _result;
    return Scaffold(
      appBar: AppBar(title: const Text('Photo analysis')),
      body: ListView(
        padding: const EdgeInsets.all(12),
        children: [
          Row(
            children: [
              Expanded(
                child: FilledButton.icon(
                  onPressed: _working ? null : () => _pick(ImageSource.camera),
                  icon: const Icon(Icons.photo_camera),
                  label: const Text('Camera'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: _working ? null : () => _pick(ImageSource.gallery),
                  icon: const Icon(Icons.photo_library),
                  label: const Text('Gallery'),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          if (_working) const Center(child: CircularProgressIndicator()),
          if (_error != null) Text(_error!, style: const TextStyle(color: Colors.red)),
          if (_bytes != null)
            AspectRatio(
              aspectRatio: _aspect,
              child: Stack(fit: StackFit.expand, children: [
                Image.memory(_bytes!, fit: BoxFit.fill),
                CustomPaint(painter: OverlayPainter(mask: result?.mask, pests: result?.pests ?? const [])),
              ]),
            ),
          if (result != null) ...[
            const SizedBox(height: 12),
            ReportCard(report: result.report, disclaimer: s.kb.disclaimer),
            const SizedBox(height: 8),
            Wrap(spacing: 8, children: [
              ActionChip(
                avatar: const Icon(Icons.record_voice_over),
                label: const Text('Read aloud'),
                onPressed: () => s.voice.speak(result.report.spoken, force: true),
              ),
              ActionChip(
                avatar: const Icon(Icons.healing),
                label: const Text('Treatment'),
                onPressed: () => s.voice.speak(result.report.treatmentSpoken, force: true),
              ),
            ]),
          ],
        ],
      ),
    );
  }
}

import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../logic/smoothing.dart';
import '../ml/analyzer.dart';
import '../ml/preprocess.dart';
import '../services/app_services.dart';
import '../services/voice_commands.dart';
import '../widgets/overlay_painter.dart';
import '../widgets/report_card.dart';

/// Real-time scan: every Nth camera frame -> disease + pest + damage -> overlay + spoken advice.
class LiveScreen extends StatefulWidget {
  const LiveScreen({super.key, required this.services});

  final AppServices services;

  @override
  State<LiveScreen> createState() => _LiveScreenState();
}

class _LiveScreenState extends State<LiveScreen> {
  CameraController? _controller;
  String? _error;
  AnalysisResult? _result;
  late final MajorityVote _vote = MajorityVote(window: widget.services.settings.live.smoothingWindow);
  int _frame = 0;
  bool _busy = false;
  bool _forceNext = false;

  AppServices get s => widget.services;

  @override
  void initState() {
    super.initState();
    _start();
  }

  Future<void> _start() async {
    try {
      final cameras = await availableCameras();
      final back =
          cameras.firstWhere((c) => c.lensDirection == CameraLensDirection.back, orElse: () => cameras.first);
      final controller = CameraController(
        back,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: Platform.isIOS ? ImageFormatGroup.bgra8888 : ImageFormatGroup.yuv420,
      );
      await controller.initialize();
      await controller.startImageStream((f) => _onFrame(f, back.sensorOrientation));
      if (!mounted) return;
      setState(() => _controller = controller);
    } catch (e) {
      setState(() => _error = 'Camera unavailable: $e');
    }
  }

  void _onFrame(CameraImage frame, int sensorOrientation) {
    _frame++;
    final due = _frame % s.settings.live.processEveryNFrames == 0 || _forceNext;
    if (_busy || !due) return;
    _busy = true;
    _forceNext = false;
    try {
      final image = cameraImageToImage(frame, rotationDegrees: Platform.isAndroid ? sensorOrientation : 0);
      final result = s.analyzer.analyze(image, vote: _vote);
      if (!mounted) return;
      setState(() => _result = result);
      s.voice.speak(result.report.spoken, cooldown: Duration(seconds: s.settings.live.speakCooldownSeconds));
    } finally {
      _busy = false;
    }
  }

  void _onCommand(VoiceCommand command, String words) {
    final report = _result?.report;
    switch (command) {
      case VoiceCommand.analyze:
        _vote.reset();
        _forceNext = true;
      case VoiceCommand.repeat:
        if (report != null) s.voice.speak(report.spoken, force: true);
      case VoiceCommand.treatment:
        if (report != null) s.voice.speak(report.treatmentSpoken, force: true);
      case VoiceCommand.stop:
        s.voice.stop();
        setState(() => s.voice.enabled = false);
      default:
        s.voice.speak(helpText, force: true);
    }
  }

  @override
  void dispose() {
    s.voice.stop();
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = _controller;
    final result = _result;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Live scan'),
        actions: [
          IconButton(
            tooltip: s.voice.enabled ? 'Mute voice' : 'Unmute voice',
            icon: Icon(s.voice.enabled ? Icons.volume_up : Icons.volume_off),
            onPressed: () => setState(() {
              s.voice.enabled = !s.voice.enabled;
              if (!s.voice.enabled) s.voice.stop();
            }),
          ),
        ],
      ),
      floatingActionButton: s.voice.listeningAvailable
          ? FloatingActionButton(
              tooltip: 'Voice command',
              onPressed: () => s.voice.listen(_onCommand),
              child: const Icon(Icons.mic),
            )
          : null,
      body: _error != null
          ? Center(child: Padding(padding: const EdgeInsets.all(24), child: Text(_error!)))
          : controller == null
              ? const Center(child: CircularProgressIndicator())
              : Column(
                  children: [
                    CameraPreview(
                      controller,
                      child: CustomPaint(
                        painter: OverlayPainter(mask: result?.mask, pests: result?.pests ?? const []),
                        child: const SizedBox.expand(),
                      ),
                    ),
                    Expanded(
                      child: SingleChildScrollView(
                        padding: const EdgeInsets.all(12),
                        child: result == null
                            ? const Text('Point the camera at a leaf...')
                            : ReportCard(report: result.report, disclaimer: s.kb.disclaimer),
                      ),
                    ),
                  ],
                ),
    );
  }
}

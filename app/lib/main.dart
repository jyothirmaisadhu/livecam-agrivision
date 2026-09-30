import 'package:flutter/material.dart';

import 'screens/home_screen.dart';
import 'services/app_services.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const AgriVisionApp());
}

class AgriVisionApp extends StatefulWidget {
  const AgriVisionApp({super.key});

  @override
  State<AgriVisionApp> createState() => _AgriVisionAppState();
}

class _AgriVisionAppState extends State<AgriVisionApp> {
  late final Future<AppServices> _services = AppServices.load();

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'LiveCam AgriVision',
      theme: ThemeData(colorSchemeSeed: Colors.green, useMaterial3: true),
      home: FutureBuilder<AppServices>(
        future: _services,
        builder: (context, snap) {
          if (snap.hasError) {
            return Scaffold(body: Center(child: Text('Failed to start: ${snap.error}')));
          }
          if (!snap.hasData) {
            return const Scaffold(body: Center(child: CircularProgressIndicator()));
          }
          return HomeScreen(services: snap.data!);
        },
      ),
    );
  }
}

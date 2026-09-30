import 'package:agrivision_app/logic/engine.dart';
import 'package:agrivision_app/widgets/report_card.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('ReportCard shows headline, severity and expandable advice', (tester) async {
    const report = Report(
      crop: 'tomato',
      headline: 'Tomato: Early Blight (88%)',
      severityPercent: 12.5,
      severityLevel: 'moderate',
      items: [Advice('Early Blight', 'Brown rings', treatment: 'Spray X', prevention: 'Rotate crops')],
      spoken: '',
    );
    await tester.pumpWidget(const MaterialApp(
      home: Scaffold(
          body: SingleChildScrollView(child: ReportCard(report: report, disclaimer: 'Follow labels.'))),
    ));
    expect(find.text('Tomato: Early Blight (88%)'), findsOneWidget);
    expect(find.text('Severity: moderate (12.5%)'), findsOneWidget);
    expect(find.text('Spray X'), findsNothing);
    await tester.tap(find.text('Early Blight'));
    await tester.pumpAndSettle();
    expect(find.text('Spray X'), findsOneWidget);
    expect(find.text('Follow labels.'), findsOneWidget);
  });
}

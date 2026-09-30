import 'package:flutter/material.dart';

import '../logic/engine.dart';

Color severityColor(String? level) => switch (level) {
      'mild' => Colors.amber,
      'moderate' => Colors.orange,
      'severe' => Colors.red,
      _ => Colors.green,
    };

class ReportCard extends StatelessWidget {
  const ReportCard({super.key, required this.report, required this.disclaimer});

  final Report report;
  final String disclaimer;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Text(report.headline, style: theme.textTheme.titleLarge),
        if (report.severityPercent != null && report.severityLevel != null)
          Padding(
            padding: const EdgeInsets.only(top: 8),
            child: Row(children: [
              Chip(
                backgroundColor: severityColor(report.severityLevel).withOpacity(0.2),
                label: Text(
                    'Severity: ${report.severityLevel} (${report.severityPercent!.toStringAsFixed(1)}%)'),
              ),
            ]),
          ),
        for (final a in report.items)
          Card(
            child: ExpansionTile(
              title: Text(a.title),
              childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
              expandedCrossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (a.detail.isNotEmpty) Text(a.detail),
                if (a.treatment.isNotEmpty) ...[
                  const SizedBox(height: 8),
                  Text('Treatment', style: theme.textTheme.labelLarge),
                  Text(a.treatment),
                ],
                if (a.prevention.isNotEmpty) ...[
                  const SizedBox(height: 8),
                  Text('Prevention', style: theme.textTheme.labelLarge),
                  Text(a.prevention),
                ],
              ],
            ),
          ),
        Padding(
          padding: const EdgeInsets.only(top: 8),
          child: Text(disclaimer, style: theme.textTheme.bodySmall),
        ),
      ],
    );
  }
}

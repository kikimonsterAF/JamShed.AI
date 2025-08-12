import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../common/widgets/usage_meter.dart';
import '../common/providers/analysis_provider.dart';

class ResultsScreen extends ConsumerWidget {
  const ResultsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final analysis = ref.watch(analysisProvider);
    return DefaultTabController(
      length: 4,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Results'),
          bottom: const TabBar(
            isScrollable: true,
            tabs: [
              Tab(text: 'Tab'),
              Tab(text: 'Chords + Form'),
              Tab(text: 'Strumming'),
              Tab(text: 'Scale Suggestions'),
            ],
          ),
        ),
        body: Column(
          children: [
            const Padding(
              padding: EdgeInsets.all(12),
              child: UsageMeter(),
            ),
            Expanded(
              child: analysis.when(
                data: (data) => TabBarView(
                  children: [
                    Center(child: Text('Tab view placeholder')),
                    Padding(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          if (data != null)
                            Wrap(
                              spacing: 8,
                              children: [
                                for (final s in data.form) Chip(label: Text(s)),
                              ],
                            ),
                          const SizedBox(height: 12),
                          Text(
                            data == null ? '—' : data.chords.join('  '),
                            style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
                          ),
                        ],
                      ),
                    ),
                    Center(child: Text('Strumming placeholder')),
                    Padding(
                      padding: const EdgeInsets.all(16),
                      child: ListView(
                        children: [
                          if (data != null)
                            ...data.scales.map(
                              (s) => ListTile(
                                title: Text('${s.sectionId} • ${s.chord}'),
                                subtitle: Wrap(
                                  spacing: 6,
                                  children: s.recommendedScales
                                      .map((rs) => Chip(label: Text(rs)))
                                      .toList(),
                                ),
                              ),
                            ),
                          if (data == null) const Text('No scale suggestions'),
                        ],
                      ),
                    ),
                  ],
                ),
                loading: () => const Center(child: CircularProgressIndicator()),
                error: (e, _) => Center(child: Text('Error: $e')),
              ),
            ),
          ],
        ),
      ),
    );
  }
}


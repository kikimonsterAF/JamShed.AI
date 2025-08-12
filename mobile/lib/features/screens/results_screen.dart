import 'package:flutter/material.dart';
import '../common/widgets/usage_meter.dart';

class ResultsScreen extends StatelessWidget {
  const ResultsScreen({super.key});

  @override
  Widget build(BuildContext context) {
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
            const Expanded(
              child: TabBarView(
                children: [
                  Center(child: Text('Tab view placeholder')),
                  Center(child: Text('Chords + Form placeholder')),
                  Center(child: Text('Strumming placeholder')),
                  Center(child: Text('Scale Suggestions placeholder')),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}


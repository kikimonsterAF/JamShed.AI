import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

class InstrumentScreen extends StatelessWidget {
  const InstrumentScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Instrument & Difficulty')),
      body: Center(
        child: FilledButton(
          onPressed: () => context.push('/results'),
          child: const Text('Analyze → Results'),
        ),
      ),
    );
  }
}


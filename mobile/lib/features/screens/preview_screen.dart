import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

class PreviewScreen extends StatelessWidget {
  const PreviewScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Preview & Timestamps')),
      body: Center(
        child: FilledButton(
          onPressed: () => context.push('/instrument'),
          child: const Text('Next: Instrument & Difficulty'),
        ),
      ),
    );
  }
}


import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../common/providers/analysis_provider.dart';

class PreviewScreen extends ConsumerWidget {
  const PreviewScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Scaffold(
      appBar: AppBar(title: const Text('Preview & Timestamps')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            FilledButton(
              onPressed: () async {
                final result = await FilePicker.platform.pickFiles(type: FileType.media);
                final path = result?.files.single.path;
                if (path == null) return;
                await ref.read(analysisProvider.notifier).uploadAndAnalyze(path);
                if (context.mounted) context.push('/instrument');
              },
              child: const Text('Choose file and Analyze'),
            ),
            const SizedBox(height: 12),
            Consumer(builder: (context, ref, _) {
              final state = ref.watch(analysisProvider);
              return state.when(
                data: (data) => Text(data == null ? 'No analysis yet' : 'Chords: ${data.chords.join(" ")}  Form: ${data.form.join(" ")}') ,
                loading: () => const CircularProgressIndicator(),
                error: (e, _) => Text('Error: $e'),
              );
            }),
          ],
        ),
      ),
    );
  }
}


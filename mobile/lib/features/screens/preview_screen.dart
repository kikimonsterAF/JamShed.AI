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
            const _MeterSelector(),
            FilledButton(
              onPressed: () async {
                final result = await FilePicker.platform.pickFiles(type: FileType.media);
                final path = result?.files.single.path;
                if (path == null) return;
                final bpb = _MeterSelector.of(context)?.selectedBpb;
                await ref.read(analysisProvider.notifier).uploadAndAnalyze(path, beatsPerBar: bpb, meter: bpb != null ? '${bpb}/4' : null);
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


class _MeterSelector extends StatefulWidget {
  const _MeterSelector();
  static _MeterSelectorState? of(BuildContext context) => context.findAncestorStateOfType<_MeterSelectorState>();
  @override
  State<_MeterSelector> createState() => _MeterSelectorState();
}

class _MeterSelectorState extends State<_MeterSelector> {
  int? selectedBpb; // null = Auto
  final options = const [null, 2, 3, 4];

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        const Text('Meter:'),
        const SizedBox(width: 8),
        Wrap(
          spacing: 8,
          children: options
              .map((o) => ChoiceChip(
                    label: Text(o == null ? 'Auto' : '${o}/4'),
                    selected: selectedBpb == o,
                    onSelected: (_) => setState(() => selectedBpb = o),
                  ))
              .toList(),
        ),
      ],
    );
  }
}


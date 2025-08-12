import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../common/widgets/usage_meter.dart';
import '../common/providers/analysis_provider.dart';
import '../common/providers/entitlement_provider.dart';

class HomeScreen extends ConsumerWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final entitlement = ref.watch(entitlementProvider);
    ref.read(entitlementProvider.notifier).refreshFromServer();
    return Scaffold(
      appBar: AppBar(title: const Text('Home')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const UsageMeter(),
            const SizedBox(height: 16),
            FilledButton(
              onPressed: () {
                if (entitlement.isSubscribed || entitlement.remainingFree > 0) {
                  context.push('/preview');
                } else {
                  context.push('/paywall');
                }
              },
              child: const Text('Upload/Link to Start'),
            ),
            const SizedBox(height: 8),
            // Simple YouTube URL input
            _YoutubeInputRow(),
            const SizedBox(height: 8),
            OutlinedButton(onPressed: () => context.push('/library'), child: const Text('Library')),
          ],
        ),
      ),
    );
  }
}


class _YoutubeInputRow extends ConsumerStatefulWidget {
  @override
  ConsumerState<_YoutubeInputRow> createState() => _YoutubeInputRowState();
}

class _YoutubeInputRowState extends ConsumerState<_YoutubeInputRow> {
  final _controller = TextEditingController();
  bool _loading = false;
  int? _beatsPerBar; // null = Auto
  final _meterOptions = const [null, 2, 3, 4];

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(analysisProvider);
    _loading = state.isLoading;
    return Row(
      children: [
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              TextField(
                controller: _controller,
                decoration: const InputDecoration(
                  labelText: 'YouTube URL',
                  hintText: 'https://www.youtube.com/watch?v=...'
                ),
              ),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8,
                children: _meterOptions
                    .map((o) => ChoiceChip(
                          label: Text(o == null ? 'Auto' : '${o}/4'),
                          selected: _beatsPerBar == o,
                          onSelected: (_) => setState(() => _beatsPerBar = o),
                        ))
                    .toList(),
              ),
            ],
          ),
        ),
        const SizedBox(width: 8),
        FilledButton(
          onPressed: _loading
              ? null
              : () async {
                  final url = _controller.text.trim();
                  if (url.isEmpty) return;
                  await ref.read(analysisProvider.notifier).analyzeYoutubeUrl(
                        url,
                        beatsPerBar: _beatsPerBar,
                        meter: _beatsPerBar != null ? '${_beatsPerBar}/4' : null,
                      );
                  if (context.mounted) context.push('/instrument');
                },
          child: _loading
              ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
              : const Text('Analyze'),
        )
      ],
    );
  }
}


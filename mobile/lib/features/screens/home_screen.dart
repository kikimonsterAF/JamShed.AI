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
          child: TextField(
            controller: _controller,
            decoration: const InputDecoration(
              labelText: 'YouTube URL',
              hintText: 'https://www.youtube.com/watch?v=...'
            ),
          ),
        ),
        const SizedBox(width: 8),
        FilledButton(
          onPressed: _loading
              ? null
              : () async {
                  final url = _controller.text.trim();
                  if (url.isEmpty) return;
                  await ref.read(analysisProvider.notifier).analyzeYoutubeUrl(url);
                  if (context.mounted) context.push('/instrument');
                },
          child: _loading ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2)) : const Text('Analyze'),
        )
      ],
    );
  }
}


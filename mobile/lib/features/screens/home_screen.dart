import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../common/widgets/usage_meter.dart';
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
            OutlinedButton(onPressed: () => context.push('/library'), child: const Text('Library')),
          ],
        ),
      ),
    );
  }
}


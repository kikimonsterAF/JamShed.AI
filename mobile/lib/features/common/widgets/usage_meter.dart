import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../providers/entitlement_provider.dart';

class UsageMeter extends ConsumerWidget {
  const UsageMeter({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(entitlementProvider);
    final bool gated = !state.isSubscribed && state.remainingFree <= 0;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Expanded(
              child: Text(
                state.isSubscribed
                    ? 'Subscribed: Unlimited transcriptions'
                    : 'Free remaining: ${state.remainingFree} of ${state.freeQuota}',
                style: const TextStyle(fontWeight: FontWeight.w600),
              ),
            ),
            if (!state.isSubscribed)
              ElevatedButton(
                onPressed: gated ? () => context.go('/paywall') : null,
                child: const Text('Upgrade'),
              ),
          ],
        ),
      ),
    );
  }
}


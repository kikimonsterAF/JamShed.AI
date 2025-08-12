import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import '../../../core/config.dart';

class EntitlementState {
  final bool isSubscribed;
  final int transcriptionsUsed;
  final int freeQuota;

  const EntitlementState({
    required this.isSubscribed,
    required this.transcriptionsUsed,
    required this.freeQuota,
  });

  int get remainingFree => isSubscribed ? 999999 : (freeQuota - transcriptionsUsed).clamp(0, freeQuota);

  EntitlementState copyWith({bool? isSubscribed, int? transcriptionsUsed, int? freeQuota}) {
    return EntitlementState(
      isSubscribed: isSubscribed ?? this.isSubscribed,
      transcriptionsUsed: transcriptionsUsed ?? this.transcriptionsUsed,
      freeQuota: freeQuota ?? this.freeQuota,
    );
  }
}

class EntitlementNotifier extends StateNotifier<EntitlementState> {
  EntitlementNotifier() : super(const EntitlementState(isSubscribed: false, transcriptionsUsed: 0, freeQuota: 5));

  void incrementUsage() {
    if (state.isSubscribed) return;
    state = state.copyWith(transcriptionsUsed: (state.transcriptionsUsed + 1));
  }

  void setSubscribed(bool value) {
    state = state.copyWith(isSubscribed: value);
  }

  Future<void> refreshFromServer() async {
    try {
      final uri = Uri.parse('${AppConfig.apiBaseUrl}/usage');
      final res = await http.get(uri).timeout(const Duration(seconds: 5));
      if (res.statusCode == 200) {
        final body = json.decode(res.body) as Map<String, dynamic>;
        state = state.copyWith(
          isSubscribed: body['is_subscribed'] as bool? ?? false,
          transcriptionsUsed: body['transcriptions_used'] as int? ?? 0,
          freeQuota: body['free_quota'] as int? ?? 5,
        );
      }
    } catch (_) {
      // Leave local state on network error
    }
  }
}

final StateNotifierProvider<EntitlementNotifier, EntitlementState> entitlementProvider =
    StateNotifierProvider<EntitlementNotifier, EntitlementState>((ref) => EntitlementNotifier());


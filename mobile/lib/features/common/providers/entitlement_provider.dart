import 'package:flutter_riverpod/flutter_riverpod.dart';

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
}

final StateNotifierProvider<EntitlementNotifier, EntitlementState> entitlementProvider =
    StateNotifierProvider<EntitlementNotifier, EntitlementState>((ref) => EntitlementNotifier());


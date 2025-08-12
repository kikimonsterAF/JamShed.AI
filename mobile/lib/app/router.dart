import 'package:flutter/foundation.dart' show defaultTargetPlatform, TargetPlatform;
import 'package:flutter/cupertino.dart' show CupertinoPageRoute;
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../features/screens/onboarding_screen.dart';
import '../features/screens/home_screen.dart';
import '../features/screens/preview_screen.dart';
import '../features/screens/instrument_screen.dart';
import '../features/screens/results_screen.dart';
import '../features/screens/paywall_screen.dart';
import '../features/screens/library_screen.dart';

PageRoute<T> _platformRouteBuilder<T>(Widget page) {
  final platform = defaultTargetPlatform;
  if (platform == TargetPlatform.iOS || platform == TargetPlatform.macOS) {
    return CupertinoPageRoute<T>(builder: (_) => page) as PageRoute<T>;
  }
  return MaterialPageRoute<T>(builder: (_) => page);
}

final GoRouter appRouter = GoRouter(
  initialLocation: '/',
  routes: <GoRoute>[
    GoRoute(path: '/', builder: (_, __) => const OnboardingScreen()),
    GoRoute(path: '/home', builder: (_, __) => const HomeScreen()),
    GoRoute(path: '/preview', builder: (_, __) => const PreviewScreen()),
    GoRoute(path: '/instrument', builder: (_, __) => const InstrumentScreen()),
    GoRoute(path: '/results', builder: (_, __) => const ResultsScreen()),
    GoRoute(path: '/paywall', builder: (_, __) => const PaywallScreen()),
    GoRoute(path: '/library', builder: (_, __) => const LibraryScreen()),
  ],
);


import 'dart:convert';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:http/http.dart' as http;
import '../../../core/config.dart';

class ScaleSuggestionModel {
  final String sectionId;
  final String chord;
  final String keyCenter;
  final List<String> recommendedScales;
  ScaleSuggestionModel({
    required this.sectionId,
    required this.chord,
    required this.keyCenter,
    required this.recommendedScales,
  });
}

class AnalysisResult {
  final List<String> chords;
  final List<String> form;
  final List<ScaleSuggestionModel> scales;
  AnalysisResult({required this.chords, required this.form, required this.scales});
}

class AnalysisNotifier extends StateNotifier<AsyncValue<AnalysisResult?>> {
  AnalysisNotifier() : super(const AsyncData(null));

  Future<void> uploadAndAnalyze(String filePath, {String? instrument, String? difficulty}) async {
    state = const AsyncLoading();
    try {
      final uri = Uri.parse('${AppConfig.apiBaseUrl}/analyze');
      final req = http.MultipartRequest('POST', uri)
        ..files.add(await http.MultipartFile.fromPath('file', filePath));
      if (instrument != null) req.fields['instrument'] = instrument;
      if (difficulty != null) req.fields['difficulty'] = difficulty;
      final res = await http.Response.fromStream(await req.send());
      if (res.statusCode == 200) {
        final body = json.decode(res.body) as Map<String, dynamic>;
        final chords = (body['chords'] as List).cast<String>();
        final form = (body['form'] as List).cast<String>();
        final scalesJson = (body['scale_suggestions'] as List? ?? []) as List;
        final scales = scalesJson
            .map((e) => e as Map<String, dynamic>)
            .map((m) => ScaleSuggestionModel(
                  sectionId: (m['section_id'] ?? '').toString(),
                  chord: (m['chord'] ?? '').toString(),
                  keyCenter: (m['key_center'] ?? '').toString(),
                  recommendedScales: ((m['recommended_scales'] as List?) ?? []).map((x) => x.toString()).toList(),
                ))
            .toList();
        state = AsyncData(AnalysisResult(chords: chords, form: form, scales: scales));
      } else {
        state = AsyncError('Server error: ${res.statusCode}', StackTrace.current);
      }
    } catch (e, st) {
      state = AsyncError(e, st);
    }
  }
}

final analysisProvider = StateNotifierProvider<AnalysisNotifier, AsyncValue<AnalysisResult?>>(
  (ref) => AnalysisNotifier(),
);


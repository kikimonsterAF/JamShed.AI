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
  final String? localFilePath;
  final int? beatsPerBar;
  final String? meter;
  AnalysisResult({required this.chords, required this.form, required this.scales, this.localFilePath, this.beatsPerBar, this.meter});
}

class AnalysisNotifier extends StateNotifier<AsyncValue<AnalysisResult?>> {
  AnalysisNotifier() : super(const AsyncData(null));

  Future<void> uploadAndAnalyze(String filePath, {String? instrument, String? difficulty, int? beatsPerBar, String? meter}) async {
    state = const AsyncLoading();
    try {
      final uri = Uri.parse('${AppConfig.apiBaseUrl}/analyze');
      final req = http.MultipartRequest('POST', uri)
        ..files.add(await http.MultipartFile.fromPath('file', filePath));
      if (instrument != null) req.fields['instrument'] = instrument;
      if (difficulty != null) req.fields['difficulty'] = difficulty;
      if (beatsPerBar != null) req.fields['beats_per_bar'] = beatsPerBar.toString();
      if (meter != null) req.fields['meter'] = meter;
      final res = await http.Response.fromStream(await req.send());
      // Debug: log the request/response for troubleshooting 404s
      // ignore: avoid_print
      print('POST ${uri.toString()} -> ${res.statusCode}');
      // ignore: avoid_print
      print(res.body);
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
        state = AsyncData(AnalysisResult(
          chords: chords,
          form: form,
          scales: scales,
          localFilePath: filePath,
          beatsPerBar: (body['beats_per_bar'] as int?),
          meter: body['meter'] as String?,
        ));
      } else {
        state = AsyncError('Server error: ${res.statusCode}: ${res.body}', StackTrace.current);
      }
    } catch (e, st) {
      state = AsyncError(e, st);
    }
  }

  Future<void> analyzeYoutubeUrl(String url, {int? beatsPerBar, String? meter}) async {
    state = const AsyncLoading();
    try {
      final uri = Uri.parse('${AppConfig.apiBaseUrl}/analyze_url').replace(queryParameters: {
        if (beatsPerBar != null) 'beats_per_bar': beatsPerBar.toString(),
        if (meter != null) 'meter': meter,
      });
      final req = http.MultipartRequest('POST', uri)
        ..fields['url'] = url;
      final res = await http.Response.fromStream(await req.send());
      // ignore: avoid_print
      print('POST ${uri.toString()} -> ${res.statusCode}');
      // ignore: avoid_print
      print(res.body);
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
        state = AsyncData(AnalysisResult(
          chords: chords,
          form: form,
          scales: scales,
          localFilePath: null,
          beatsPerBar: (body['beats_per_bar'] as int?),
          meter: body['meter'] as String?,
        ));
      } else {
        state = AsyncError('Server error: ${res.statusCode}: ${res.body}', StackTrace.current);
      }
    } catch (e, st) {
      state = AsyncError(e, st);
    }
  }
}

final analysisProvider = StateNotifierProvider<AnalysisNotifier, AsyncValue<AnalysisResult?>>(
  (ref) => AnalysisNotifier(),
);


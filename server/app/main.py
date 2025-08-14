from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import List, Optional, Tuple
from mutagen import File as MutagenFile
import tempfile
import shutil
import subprocess
import os
import numpy as np
import soundfile as sf
import librosa
import uuid
from tempfile import TemporaryDirectory

# AISU transformer integration
print("[DEBUG] Attempting to import AISU transformers...")
try:
    from .aisu_chord_detection import predict_chords_aisu
    AISU_AVAILABLE = True
    print("[DEBUG] AISU basic transformer available")
except ImportError as e:
    print(f"[DEBUG] AISU basic transformer failed: {e}")
    AISU_AVAILABLE = False
    def predict_chords_aisu(audio_path: str):
        return ["C", "F", "G", "C"]

# AISU deep transformer integration
try:
    from .aisu_deep_transformer import predict_chords_deep_transformer
    AISU_DEEP_AVAILABLE = True
    print("[DEBUG] AISU deep transformer available - competition-grade model loaded")
except ImportError as e:
    print(f"[DEBUG] AISU deep transformer not available: {e}")
    AISU_DEEP_AVAILABLE = False
    def predict_chords_deep_transformer(audio_path: str):
        print("[DEBUG] Using fallback deep transformer")
        return ["C", "F", "G", "C"]

app = FastAPI(title="JamTab API", version="0.1.0")


class ScaleSuggestion(BaseModel):
    section_id: str
    chord: str
    key_center: str
    recommended_scales: List[str]
    scale_degrees: List[str]
    target_notes: List[str]


class AnalysisResponse(BaseModel):
    chords: List[str]
    form: List[str]
    scale_suggestions: List[ScaleSuggestion]
    # Estimated meter information (auto or user-provided override echoed back)
    beats_per_bar: Optional[int] = None
    meter: Optional[str] = None
    meter_debug: Optional[dict] = None


class UsageResponse(BaseModel):
    is_subscribed: bool
    transcriptions_used: int
    free_quota: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/usage", response_model=UsageResponse)
def get_usage() -> UsageResponse:
    # TODO: replace with DB lookup and entitlement check
    return UsageResponse(is_subscribed=False, transcriptions_used=0, free_quota=5)


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    file: UploadFile = File(...),
    instrument: Optional[str] = None,
    difficulty: Optional[str] = None,
    beats_per_bar: Optional[int] = None,
    meter: Optional[str] = None,
    use_aisu: Optional[int] = None,
    use_deep: Optional[int] = None,
) -> AnalysisResponse:
    allowed_types = {
        "audio/mpeg",  # .mp3
        "audio/mp3",  # some devices report this
        "audio/mp4",  # .m4a
        "audio/aac",
        "audio/x-aac",
        "audio/3gpp",
        "audio/3gpp2",
        "audio/ogg",
        "audio/x-wav",  # .wav
        "audio/wav",
        "video/mp4",  # mp4 containers
    }
    if (file.content_type or "").lower() not in allowed_types:
        # Accept by extension as a fallback (some pickers omit content-type)
        filename = (file.filename or "").lower()
        allowed_exts = (".mp3", ".m4a", ".aac", ".wav", ".ogg", ".3gp", ".mp4")
        if not any(filename.endswith(ext) for ext in allowed_exts):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported content type: {file.content_type} (filename: {file.filename})",
            )

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or "")[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        src_path = tmp.name
    try:
        # Echo duration if available (non-critical)
        try:
            mf = MutagenFile(src_path)
            _ = float(getattr(mf.info, "length", 0.0) or 0.0)
        except Exception:
            pass
        # Delegate to path-based analyzer
        result = _analyze_path(
            src_path=src_path,
            instrument=instrument,
            difficulty=difficulty,
            beats_per_bar=beats_per_bar,
            meter=meter,
            use_aisu=use_aisu,
            use_deep=use_deep,
        )
        return result
    finally:
        try:
            file.file.close()
        except Exception:
            pass
        try:
            os.remove(src_path)
        except Exception:
            pass


@app.post("/analyze_url", response_model=AnalysisResponse)
async def analyze_url(
    url: str = Form(...),
    instrument: Optional[str] = None,
    difficulty: Optional[str] = None,
    beats_per_bar: Optional[int] = None,
    meter: Optional[str] = None,
    use_aisu: Optional[int] = None,
    use_deep: Optional[int] = None,
    meter_debug: Optional[int] = None,
) -> AnalysisResponse:
    """Download audio from YouTube (or other supported sites) and analyze it.
    Requires yt-dlp and ffmpeg to be present.
    """
    try:
        import yt_dlp  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"yt-dlp not installed: {e}")

    with TemporaryDirectory() as tmpdir:
        out_base = os.path.join(tmpdir, str(uuid.uuid4()))
        out_path = f"{out_base}.mp3"
        # Resolve ffmpeg/ffprobe location for yt-dlp (accept either the binary path or its parent directory)
        ffbin = os.environ.get("FFMPEG_BIN")
        ffmpeg_location = None
        if ffbin:
            ffmpeg_location = os.path.dirname(ffbin) if ffbin.lower().endswith("ffmpeg.exe") or ffbin.lower().endswith("ffmpeg") else ffbin
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": out_base,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "0",
                }
            ],
            "quiet": True,
            "nocheckcertificate": True,
        }
        if ffmpeg_location:
            ydl_opts["ffmpeg_location"] = ffmpeg_location
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download media: {e}")

        # Reuse the same analysis pipeline by opening the file as if it was uploaded
        # Determine actual downloaded audio file path
        candidate_path = out_path if os.path.exists(out_path) else None
        if candidate_path is None:
            # Fallback: search temp dir for common audio extensions, pick largest file
            exts = (".m4a", ".mp3", ".ogg", ".opus", ".webm", ".wav", ".mka")
            candidates = []
            for root, _, files in os.walk(tmpdir):
                for fn in files:
                    if os.path.splitext(fn)[1].lower() in exts:
                        fp = os.path.join(root, fn)
                        try:
                            size = os.path.getsize(fp)
                        except Exception:
                            size = 0
                        candidates.append((size, fp))
            if candidates:
                candidates.sort(reverse=True)
                candidate_path = candidates[0][1]
        if not candidate_path or not os.path.exists(candidate_path):
            raise HTTPException(status_code=400, detail="Downloaded file not found. Check ffmpeg/yt-dlp postprocessing.")
        # Directly analyze the downloaded audio path
        return _analyze_path(
            src_path=candidate_path,
            instrument=instrument,
            difficulty=difficulty,
            beats_per_bar=beats_per_bar,
            meter=meter,
            use_aisu=use_aisu,
            use_deep=use_deep,
            meter_debug=meter_debug,
        )


def _analyze_path(
    src_path: str,
    instrument: Optional[str],
    difficulty: Optional[str],
    beats_per_bar: Optional[int],
    meter: Optional[str],
    use_aisu: Optional[int] = None,
    use_deep: Optional[int] = None,
    meter_debug: Optional[int] = None,
) -> AnalysisResponse:
    # Decode to mono PCM WAV (22.05 kHz) via ffmpeg
    y, sr = _decode_audio_with_ffmpeg(src_path, target_sr=22050)
    if y.size == 0:
        try:
            size = os.path.getsize(src_path)
        except Exception:
            size = -1
        raise HTTPException(status_code=400, detail=f"Failed to decode audio (path={src_path}, size={size}). Ensure FFmpeg/ffprobe are available and the media is valid.")

    # Chroma features and beat-synchronous pooling
    chroma, beat_frames, tempo, onset_env, mel, f0, bass_chroma = _compute_beatsynced_chroma(y, sr)

    # Key estimation
    key_center, key_mode = _estimate_key_from_chroma(chroma)

    # Chord detection - Deep transformer, AISU transformer, or fallback
    print(f"[DEBUG] use_deep parameter: {use_deep}")
    print(f"[DEBUG] use_aisu parameter: {use_aisu}")
    print(f"[DEBUG] AISU_DEEP_AVAILABLE: {AISU_DEEP_AVAILABLE}")
    print(f"[DEBUG] AISU_AVAILABLE: {AISU_AVAILABLE}")
    
    use_deep_enabled = bool(use_deep) and int(use_deep or 0) == 1 and AISU_DEEP_AVAILABLE
    use_aisu_enabled = bool(use_aisu) and int(use_aisu or 0) == 1 and AISU_AVAILABLE and not use_deep_enabled
    
    print(f"[DEBUG] use_deep_enabled: {use_deep_enabled}")
    print(f"[DEBUG] use_aisu_enabled: {use_aisu_enabled}")
    
    if use_deep_enabled:
        try:
            print("[DEBUG] 🚀 Starting AISU DEEP TRANSFORMER chord detection...")
            chords_progression = predict_chords_deep_transformer(src_path)
            print(f"[DEBUG] 🎯 DEEP TRANSFORMER returned chords: {chords_progression}")
            # Create beat-level chords for form analysis
            # Do NOT fabricate 4-per-chord beats for meter; keep separate beat-level chroma chords
            beat_chords = _infer_chords_from_chroma(chroma)
            print(f"[DEBUG] 🎵 Deep transformer: using chroma-inferred beat_chords: {len(beat_chords)} beats")
        except Exception as e:
            print(f"[DEBUG] ❌ Deep transformer failed, falling back to AISU: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to AISU
            if AISU_AVAILABLE:
                try:
                    chords_progression = predict_chords_aisu(src_path)
                    beat_chords = []
                    for chord in chords_progression:
                        beat_chords.extend([chord] * 4)
                except:
                    beat_chords = _infer_chords_from_chroma(chroma)
                    chords_progression = _collapse_repeats(beat_chords)
            else:
                beat_chords = _infer_chords_from_chroma(chroma)
                chords_progression = _collapse_repeats(beat_chords)
    elif use_aisu_enabled:
        try:
            print("[DEBUG] Starting AISU transformer chord detection...")
            chords_progression = predict_chords_aisu(src_path)
            print(f"[DEBUG] AISU returned chords: {chords_progression}")
            # Use chroma-derived beat-level chords for meter/form to avoid 4/4 bias
            beat_chords = _infer_chords_from_chroma(chroma)
            print(f"[DEBUG] AISU: using chroma-inferred beat_chords: {len(beat_chords)} beats")
        except Exception as e:
            print(f"[DEBUG] AISU chord detection failed, falling back: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to original method
            beat_chords = _infer_chords_from_chroma(chroma)
            chords_progression = _collapse_repeats(beat_chords)
            print(f"[DEBUG] Fallback chords: {chords_progression}")
    else:
        print("[DEBUG] Using original chord detection method")
        # Original chord detection method
        beat_chords = _infer_chords_from_chroma(chroma)
        chords_progression = _collapse_repeats(beat_chords)
        print(f"[DEBUG] Original method chords: {chords_progression}")

    # Select phrase length (user override or auto)
    user_bpb: Optional[int] = None
    if beats_per_bar and beats_per_bar > 0:
        user_bpb = int(beats_per_bar)
    elif isinstance(meter, str) and "/" in meter:
        try:
            num = int(meter.split("/", 1)[0].strip())
            if num > 0:
                user_bpb = num
        except Exception:
            user_bpb = None
    debug_payload = {} if meter_debug else None
    selected_bpb = user_bpb or _auto_select_beats_per_bar(
        beat_chords=beat_chords,
        beat_frames=beat_frames,
        onset_env=onset_env,
        candidates=(2, 3, 4),
        mel=mel,
        f0_seq=f0,
        beat_chroma=bass_chroma,
        debug_dict=debug_payload,
    )

    # Form labels
    form_labels = _infer_form_from_chords(beat_chords, phrase_len=selected_bpb)

    # Scale suggestions
    unique_chords = list(dict.fromkeys(chords_progression))[:8]
    suggestions: List[ScaleSuggestion] = []
    for idx, chord in enumerate(unique_chords):
        root = chord.rstrip("m").upper()
        key_label = f"{key_center} {key_mode}"
        rec_scales = [f"{key_center} minor pentatonic", f"{root} mixolydian"]
        suggestions.append(
            ScaleSuggestion(
                section_id=chr(ord("A") + (idx % 4)),
                chord=chord,
                key_center=key_label,
                recommended_scales=rec_scales,
                scale_degrees=["1", "b3", "4", "5", "b7"],
                target_notes=[root],
            )
        )

    meter_label = f"{selected_bpb}/4" if selected_bpb else None
    # Optional debug payload for meter selection
    md = debug_payload if meter_debug else None
    return AnalysisResponse(
        chords=chords_progression,
        form=form_labels[:8],
        scale_suggestions=suggestions,
        beats_per_bar=selected_bpb,
        meter=meter_label,
        meter_debug=md,
    )


# ---------- Analysis helpers ----------

_NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _decode_audio_with_ffmpeg(src_path: str, target_sr: int = 22050) -> Tuple[np.ndarray, int]:
    """Decode arbitrary media to mono float32 PCM at target_sr using ffmpeg CLI.
    Resolves ffmpeg from env var FFMPEG_BIN if provided. Falls back to librosa/soundfile on failure.
    """
    out_fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(out_fd)
    try:
        ffmpeg_bin = os.environ.get("FFMPEG_BIN") or "ffmpeg"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            src_path,
            "-ac",
            "1",
            "-ar",
            str(target_sr),
            "-f",
            "wav",
            out_path,
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            data, sr = sf.read(out_path, dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            return data, sr
        except Exception:
            # Fallback 1: librosa (audioread) can handle many containers
            try:
                y, sr = librosa.load(src_path, sr=target_sr, mono=True)
                return y.astype(np.float32, copy=False), sr
            except Exception:
                # Fallback 2: try reading directly via soundfile
                try:
                    data, sr = sf.read(src_path, dtype="float32", always_2d=False)
                    if sr != target_sr:
                        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
                        sr = target_sr
                    if data.ndim > 1:
                        data = np.mean(data, axis=1)
                    return data, sr
                except Exception:
                    return np.array([], dtype=np.float32), target_sr
    finally:
        try:
            os.remove(out_path)
        except Exception:
            pass


def _compute_beatsynced_chroma(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute chroma-cqt and aggregate per beat for chord inference.
    Returns (beat_synced_chroma, beat_frames, tempo, onset_envelope).
    """
    y = librosa.effects.harmonic(y)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    # Bass-focused chroma (limit low fmin)
    try:
        bass_chroma = librosa.feature.chroma_cqt(y=y, sr=sr, fmin=librosa.note_to_hz('A1'))
    except Exception:
        bass_chroma = chroma
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    # Fundamental frequency (bass) track (mono). Prefer pYIN with voiced probability.
    try:
        f0, v_flag, v_prob = librosa.pyin(y, fmin=50, fmax=220, sr=sr)
        # Replace NaNs with 0
        f0 = np.nan_to_num(f0, nan=0.0)
    except Exception:
        try:
            f0 = librosa.yin(y, fmin=50, fmax=220, sr=sr)
        except Exception:
            f0 = np.zeros(1, dtype=float)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    if beats.size == 0:
        # Fallback: frame-wise pooling into fixed windows
        frames = chroma.shape[1]
        window = max(1, frames // 32)
        pools = [chroma[:, i : i + window].mean(axis=1) for i in range(0, frames, window)]
        # Approximate onset envelope at pooled resolution
        onset_pools = [onset_env[i : i + window].mean() for i in range(0, len(onset_env), window)]
        # Approximate mel pooling
        mel_pools = [mel[:, i : i + window].mean(axis=1) for i in range(0, mel.shape[1], window)]
        mel_bs = np.stack(mel_pools, axis=1) if mel_pools else np.zeros((64, 1), dtype=float)
        # Pool f0 roughly by selecting nearest sample per pooled frame (fallback zeros)
        f0_pools = []
        for i in range(0, len(onset_env), window):
            idx = min(i, len(f0) - 1) if len(f0) > 0 else 0
            f0_pools.append(float(f0[idx]) if len(f0) > 0 else 0.0)
        return (
            np.stack(pools, axis=1),
            np.arange(len(pools)),
            120.0,
            np.asarray(onset_pools, dtype=float),
            mel_bs,
            np.asarray(f0_pools, dtype=float),
            np.stack(pools, axis=1),  # fallback: use main chroma as bass chroma in no-beat case
        )
    beat_chroma = librosa.util.sync(chroma, beats, aggregate=np.mean)
    beat_bass_chroma = librosa.util.sync(bass_chroma, beats, aggregate=np.mean)
    # Keep mel and f0 at original frame-rate; we will sample them at beat frames
    return beat_chroma, beats, float(tempo), onset_env, mel, f0, beat_bass_chroma


def _estimate_key_from_chroma(chroma: np.ndarray) -> Tuple[str, str]:
    """Estimate global key (major/minor) from averaged chroma using Krumhansl profiles."""
    # Krumhansl-Schmuckler key profiles (normalized)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    major_profile = major_profile / major_profile.sum()
    minor_profile = minor_profile / minor_profile.sum()

    avg = chroma.mean(axis=1)
    scores = []
    for i in range(12):
        rot_major = np.roll(major_profile, i)
        rot_minor = np.roll(minor_profile, i)
        scores.append((np.dot(avg, rot_major), i, "major"))
        scores.append((np.dot(avg, rot_minor), i, "minor"))
    best = max(scores, key=lambda x: x[0])
    return _NOTE_NAMES_SHARP[best[1]], best[2]


def _infer_chords_from_chroma(chroma_bs: np.ndarray) -> List[str]:
    """Assign a simple chord label (major/minor triad) to each beat-synchronous chroma vector."""
    # Triad templates (C major/minor) then rotate
    major_triad = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)  # C E G
    minor_triad = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)  # C Eb G
    templates = []
    labels = []
    for i, name in enumerate(_NOTE_NAMES_SHARP):
        templates.append(np.roll(major_triad, i))
        labels.append(f"{name}")
    for i, name in enumerate(_NOTE_NAMES_SHARP):
        templates.append(np.roll(minor_triad, i))
        labels.append(f"{name}m")
    templates = np.stack(templates, axis=0)

    chords: List[str] = []
    for t in range(chroma_bs.shape[1]):
        v = chroma_bs[:, t]
        sims = templates @ v
        idx = int(np.argmax(sims))
        chords.append(labels[idx])
    return chords


def _collapse_repeats(seq: List[str]) -> List[str]:
    out: List[str] = []
    prev = None
    for x in seq:
        if x != prev:
            out.append(x)
            prev = x
    return out


def _infer_form_from_chords(beat_chords: List[str], phrase_len: int = 4) -> List[str]:
    """Form labeling: hash phrases of length phrase_len and assign A/B/C by first occurrence."""
    labels: List[str] = []
    seen = {}
    next_label_ord = ord("A")
    for i in range(0, len(beat_chords), phrase_len):
        phrase = tuple(beat_chords[i : i + phrase_len])
        if phrase not in seen:
            seen[phrase] = chr(next_label_ord)
            next_label_ord += 1
        labels.append(seen[phrase])
    return labels


def _select_phrase_len(beat_chords: List[str], candidates: Tuple[int, ...] = (2, 3, 4, 5, 6), tempo: Optional[float] = None) -> int:
    """Auto-select beats-per-bar by choosing phrase length with most repetition and lowest remainder.
    
    Enhanced with waltz detection: bias toward 3/4 time for waltz-like characteristics.
    Scoring: score = unique_phrases_ratio + remainder_penalty, lower is better.
    """
    if not beat_chords:
        return 4
    best_len = 4
    best_score = float("inf")
    total = len(beat_chords)
    
    # Detect waltz characteristics
    waltz_bias = _detect_waltz_characteristics(beat_chords, tempo)
    
    for n in candidates:
        if n <= 0:
            continue
        segments = [tuple(beat_chords[i : i + n]) for i in range(0, total, n)]
        if not segments:
            continue
        unique = len(set(segments))
        unique_ratio = unique / max(1, len(segments))
        remainder = total % n
        remainder_penalty = remainder / n
        score = unique_ratio + remainder_penalty
        
        # Apply waltz bias - strongly favor 3/4 time for waltz-like songs
        if waltz_bias and n == 3:
            score *= 0.1  # 90% bonus for 3/4 time when waltz detected (very aggressive)
            print(f"[DEBUG] 🎼 Waltz detected! Applying VERY strong 3/4 bias, score: {score:.3f}")
        elif waltz_bias and n != 3:
            score *= 1.5  # Penalize non-3/4 time when waltz detected
            print(f"[DEBUG] 🎼 Waltz detected! Penalizing {n}/4 time, score: {score:.3f}")
        
        if score < best_score:
            best_score = score
            best_len = n
    
    print(f"[DEBUG] 🥁 Beat detection - Tempo: {tempo}, Selected: {best_len}/4, Waltz bias: {waltz_bias}")
    return best_len


def _detect_waltz_characteristics(beat_chords: List[str], tempo: Optional[float] = None) -> bool:
    """Detect if this song has waltz-like characteristics (3/4 time) based on harmonic patterns."""
    if not beat_chords or len(beat_chords) < 12:
        print(f"[DEBUG] 🎼 Waltz detection: skipping - not enough chords ({len(beat_chords) if beat_chords else 0})")
        return False
    
    print(f"[DEBUG] 🎼 Waltz detection: analyzing {len(beat_chords)} beat chords (ignoring tempo)")
    
    waltz_score = 0
    total_tests = 0
    
    # Test 1: Does the song length divide better by 3 than by 2 or 4?
    total_tests += 1
    length = len(beat_chords)
    remainder_3 = length % 3
    remainder_2 = length % 2
    remainder_4 = length % 4
    
    if remainder_3 <= remainder_2 and remainder_3 <= remainder_4:
        waltz_score += 1
        print(f"[DEBUG] 🎼 Length test: {length} chords, 3-beat remainder={remainder_3} wins vs 2-beat={remainder_2}, 4-beat={remainder_4}")
    else:
        print(f"[DEBUG] 🎼 Length test: {length} chords, 3-beat remainder={remainder_3} loses vs 2-beat={remainder_2}, 4-beat={remainder_4}")
    
    # Test 2: Chord change patterns - do chords change in 3-beat groups?
    total_tests += 1
    three_beat_groups = []
    for i in range(0, len(beat_chords) - 2, 3):
        if i + 2 < len(beat_chords):
            group = beat_chords[i:i+3]
            three_beat_groups.append(group)
    
    if len(three_beat_groups) >= 4:
        # Count how often the first beat of each 3-beat group has a different chord
        chord_changes_on_downbeat = 0
        for i in range(1, len(three_beat_groups)):
            if three_beat_groups[i][0] != three_beat_groups[i-1][0]:
                chord_changes_on_downbeat += 1
        
        change_ratio = chord_changes_on_downbeat / max(1, len(three_beat_groups) - 1)
        if change_ratio > 0.4:  # At least 40% of measures start with chord changes
            waltz_score += 1
            print(f"[DEBUG] 🎼 Chord change pattern: {change_ratio:.2f} measures start with new chords (good for waltz)")
        else:
            print(f"[DEBUG] 🎼 Chord change pattern: {change_ratio:.2f} measures start with new chords (weak for waltz)")
    
    # Test 3: Repeated chord patterns in 3-beat groups
    total_tests += 1
    if len(three_beat_groups) >= 4:
        unique_3_patterns = len(set(tuple(group) for group in three_beat_groups))
        repetition_ratio = unique_3_patterns / len(three_beat_groups)
        
        # Compare with 2-beat patterns
        two_beat_groups = []
        for i in range(0, len(beat_chords) - 1, 2):
            if i + 1 < len(beat_chords):
                group = beat_chords[i:i+2]
                two_beat_groups.append(tuple(group))
        
        if two_beat_groups:
            unique_2_patterns = len(set(two_beat_groups))
            repetition_ratio_2 = unique_2_patterns / len(two_beat_groups)
            
            # Waltzes often have more repetitive 3-beat patterns than 2-beat patterns
            if repetition_ratio < repetition_ratio_2:
                waltz_score += 1
                print(f"[DEBUG] 🎼 Pattern repetition: 3-beat={repetition_ratio:.2f} more repetitive than 2-beat={repetition_ratio_2:.2f}")
            else:
                print(f"[DEBUG] 🎼 Pattern repetition: 3-beat={repetition_ratio:.2f} less repetitive than 2-beat={repetition_ratio_2:.2f}")
    
    # Test 4: Long chord holds (waltzes often hold chords for 3 or 6 beats)
    total_tests += 1
    chord_runs = []
    current_chord = beat_chords[0]
    current_run = 1
    
    for i in range(1, len(beat_chords)):
        if beat_chords[i] == current_chord:
            current_run += 1
        else:
            if current_run > 1:
                chord_runs.append(current_run)
            current_chord = beat_chords[i]
            current_run = 1
    
    if current_run > 1:
        chord_runs.append(current_run)
    
    # Count runs that are multiples of 3
    three_beat_runs = sum(1 for run in chord_runs if run % 3 == 0)
    if chord_runs and three_beat_runs / len(chord_runs) > 0.3:
        waltz_score += 1
        print(f"[DEBUG] 🎼 Chord holds: {three_beat_runs}/{len(chord_runs)} are 3-beat multiples")
    else:
        print(f"[DEBUG] 🎼 Chord holds: {three_beat_runs}/{len(chord_runs) if chord_runs else 0} are 3-beat multiples (weak)")
    
    # Final decision - balanced threshold to catch waltzes but avoid false positives
    threshold = max(2, int(total_tests * 0.6))  # Need at least 60% of tests to pass
    waltz_detected = waltz_score >= threshold
    
    print(f"[DEBUG] 🎼 Waltz verdict: {waltz_score}/{total_tests} tests passed, threshold={threshold}, detected={waltz_detected}")
    
    return waltz_detected


def _score_phrase_len(beat_chords: List[str], n: int) -> float:
    """Higher-is-better score for phrase repetition with bar length n."""
    if not beat_chords or n <= 0:
        return 0.0
    total = len(beat_chords)
    segments = [tuple(beat_chords[i : i + n]) for i in range(0, total, n)]
    if not segments:
        return 0.0
    unique = len(set(segments))
    unique_ratio = unique / max(1, len(segments))
    remainder_penalty = (total % n) / n
    raw = unique_ratio + remainder_penalty  # lower is better
    return 1.0 / (1e-6 + raw)


def _score_meter_accent(
    beat_chords: List[str], beat_frames: np.ndarray, onset_env: np.ndarray, n: int
) -> float:
    """Score how well meter n fits expected accent patterns using onset/chord-change profiles.

    Steps:
    - Aggregate onset envelope per beat and fold by bar position (0..n-1)
    - Compute chord-change frequency per bar position
    - Build a position profile vector by blending onset means and change frequencies
    - Compare against expected accent templates via cosine similarity
    Returns higher-is-better score.
    """
    if n <= 0 or len(beat_chords) == 0 or beat_frames.size == 0:
        return 0.0
    positions = np.arange(len(beat_chords)) % n
    if onset_env.ndim == 0:
        return 0.0
    # Sample onset env per beat
    beat_onsets: List[float] = []
    last_idx = onset_env.shape[0] - 1
    for f in beat_frames:
        idx = int(f)
        if idx < 0:
            idx = 0
        if idx > last_idx:
            idx = last_idx
        beat_onsets.append(float(onset_env[idx]))
    beat_onsets = np.asarray(beat_onsets, dtype=float)
    if len(beat_onsets) != len(positions):
        L = min(len(beat_onsets), len(positions))
        beat_onsets = beat_onsets[:L]
        positions = positions[:L]

    # Per-position onset means
    onset_means = np.zeros(n, dtype=float)
    counts = np.zeros(n, dtype=float)
    for val, pos in zip(beat_onsets, positions):
        onset_means[int(pos)] += val
        counts[int(pos)] += 1.0
    onset_means = onset_means / np.maximum(counts, 1e-6)
    if onset_means.sum() > 0:
        onset_means = onset_means / (onset_means.sum() + 1e-9)

    # Per-position chord-change frequency
    changes = [i for i in range(1, len(beat_chords)) if beat_chords[i] != beat_chords[i - 1]]
    change_freq = np.zeros(n, dtype=float)
    if changes:
        for i in changes:
            change_freq[int(i % n)] += 1.0
        change_freq = change_freq / (change_freq.sum() + 1e-9)

    # Blend onset and change profiles (favor chord-change structure over raw accents)
    alpha = 0.3
    profile = alpha * onset_means + (1.0 - alpha) * change_freq
    if profile.sum() > 0:
        profile = profile / (profile.sum() + 1e-9)

    # Expected accent templates (normalized)
    if n == 2:
        template = np.array([1.0, 0.5], dtype=float)
    elif n == 3:
        template = np.array([1.0, 0.3, 0.3], dtype=float)
    elif n == 4:
        template = np.array([1.0, 0.3, 0.6, 0.3], dtype=float)  # strong-weak-medium-weak
    else:
        # For other meters, fallback to downbeat emphasis only
        template = np.zeros(n, dtype=float)
        template[0] = 1.0
    template = template / (template.sum() + 1e-9)

    # Cosine similarity between profile and template
    denom = (np.linalg.norm(profile) * np.linalg.norm(template)) + 1e-9
    cos_sim = float(np.dot(profile, template) / denom)

    # Small preference to meters that reduce remainder (stability)
    remainder_bonus = 0.0
    total = len(beat_chords)
    remainder = total % n
    remainder_bonus = 1.0 - (remainder / max(1, n))  # closer to full bars is better

    # Final score
    return 0.8 * cos_sim + 0.2 * remainder_bonus


def _change_alignment_ratio(beat_chords: List[str], n: int) -> float:
    """Return fraction of chord changes that land on bar start for meter n."""
    if n <= 0 or not beat_chords:
        return 0.0
    changes = [i for i in range(1, len(beat_chords)) if beat_chords[i] != beat_chords[i - 1]]
    if not changes:
        return 0.0
    aligned = sum(1 for i in changes if (i % n) == 0)
    return aligned / len(changes)


def _auto_select_beats_per_bar(
    beat_chords: List[str],
    beat_frames: np.ndarray,
    onset_env: np.ndarray,
    candidates: Tuple[int, ...] = (2, 3, 4),
    mel: Optional[np.ndarray] = None,
    f0_seq: Optional[np.ndarray] = None,
    debug_dict: Optional[dict] = None,
) -> int:
    """Blend phrase repetition with rhythmic accent/chord-change alignment to pick the meter."""
    if not beat_chords:
        return 4
    phrase_scores = {n: _score_phrase_len(beat_chords, n) for n in candidates}
    accent_scores = {n: _score_meter_accent(beat_chords, beat_frames, onset_env, n) for n in candidates}

    # Bar homogeneity: for each bar, fewer within-bar chord changes is better for that meter
    def _score_bar_homogeneity(beat_chords_local: List[str], n_local: int) -> float:
        if n_local <= 0 or not beat_chords_local:
            return 0.0
        total_bars = 0
        homogeneous = 0
        for i in range(0, len(beat_chords_local), n_local):
            seg = beat_chords_local[i:i+n_local]
            if not seg:
                continue
            total_bars += 1
            if all(ch == seg[0] for ch in seg):
                homogeneous += 1
        if total_bars == 0:
            return 0.0
        return homogeneous / total_bars

    homo_scores = {n: _score_bar_homogeneity(beat_chords, n) for n in candidates}

    # Spectral flux per beat from mel and fold by bar positions
    flux_scores: dict = {}
    if mel is not None and mel.size > 0 and beat_frames.size > 1:
        # Compute spectral flux on mel frames
        mel_db = librosa.power_to_db(mel, ref=np.max)
        diff = np.diff(mel_db, axis=1)
        flux = np.maximum(diff, 0.0).sum(axis=0)  # positive changes only
        # Sample flux at beat frames
        last_idx = flux.shape[0] - 1
        beat_flux: List[float] = []
        for f in beat_frames:
            idx = int(f)
            if idx < 1:
                idx = 1
            if idx > last_idx:
                idx = last_idx
            beat_flux.append(float(flux[idx]))
        beat_flux = np.asarray(beat_flux, dtype=float)
        for n in candidates:
            if n <= 0:
                flux_scores[n] = 0.0
                continue
            pos = np.arange(len(beat_flux)) % n
            pos_means = np.zeros(n, dtype=float)
            counts = np.zeros(n, dtype=float)
            for val, p in zip(beat_flux, pos):
                pos_means[int(p)] += val
                counts[int(p)] += 1.0
            pos_means = pos_means / np.maximum(counts, 1e-6)
            if pos_means.sum() > 0:
                pos_means = pos_means / (pos_means.sum() + 1e-9)
            # Compare to expected templates (same as accent templates)
            if n == 2:
                tmpl = np.array([1.0, 0.5])
            elif n == 3:
                tmpl = np.array([1.0, 0.3, 0.3])
            elif n == 4:
                tmpl = np.array([1.0, 0.3, 0.6, 0.3])
            else:
                tmpl = np.zeros(n); tmpl[0] = 1.0
            tmpl = tmpl / (tmpl.sum() + 1e-9)
            denom = (np.linalg.norm(pos_means) * np.linalg.norm(tmpl)) + 1e-9
            flux_scores[n] = float(np.dot(pos_means, tmpl) / denom)
    else:
        flux_scores = {n: 0.0 for n in candidates}

    # Low-frequency downbeat contrast (bass emphasis at bar starts)
    lf_scores: dict = {}
    if mel is not None and mel.size > 0 and beat_frames.size > 0:
        # Take lowest mel bands (e.g., first 8)
        lf = mel[: min(8, mel.shape[0]), :]
        # Energy per frame
        lf_energy = lf.sum(axis=0)
        # Sample per beat
        last_idx_lf = lf_energy.shape[0] - 1
        beat_lf: List[float] = []
        for f in beat_frames:
            idx = int(f)
            if idx < 0:
                idx = 0
            if idx > last_idx_lf:
                idx = last_idx_lf
            beat_lf.append(float(lf_energy[idx]))
        beat_lf = np.asarray(beat_lf, dtype=float)
        for n in candidates:
            if n <= 0 or beat_lf.size == 0:
                lf_scores[n] = 0.0
                continue
            pos = np.arange(len(beat_lf)) % n
            pos0 = beat_lf[pos == 0]
            posO = beat_lf[pos != 0]
            if pos0.size == 0 or posO.size == 0:
                lf_scores[n] = 0.0
            else:
                contrast = (pos0.mean() - posO.mean()) / (1e-6 + beat_lf.mean())
                lf_scores[n] = float(max(contrast, 0.0))
    else:
        lf_scores = {n: 0.0 for n in candidates}

    # Root-bass downbeat: sample f0 per beat, then fold by bar to see if bass notes emphasize bar starts
    rootbass_scores: dict = {}
    if isinstance(f0_seq, np.ndarray) and f0_seq.size > 0 and beat_frames.size > 0:
        last_idx_f0 = f0_seq.shape[0] - 1
        beat_f0: List[float] = []
        for f in beat_frames:
            idx = int(f)
            if idx < 0:
                idx = 0
            if idx > last_idx_f0:
                idx = last_idx_f0
            beat_f0.append(float(f0_seq[idx]))
        beat_f0 = np.asarray(beat_f0, dtype=float)
        # Use voiced indicator (non-zero f0) as a proxy to avoid octave drift; measure voiced energy per position
        voiced = (beat_f0 > 0).astype(float)
        for n in candidates:
            if n <= 0 or beat_f0.size == 0:
                rootbass_scores[n] = 0.0
                continue
            pos = np.arange(len(beat_f0)) % n
            v0 = voiced[pos == 0].mean() if np.any(pos == 0) else 0.0
            vO = voiced[pos != 0].mean() if np.any(pos != 0) else 0.0
            # Contrast of voiced probability at bar starts vs others
            rootbass_scores[n] = float(max(v0 - vO, 0.0))
    else:
        rootbass_scores = {n: 0.0 for n in candidates}

    # Periodicity via autocorrelation at lag=n (compare 3-beat vs 2-beat strength)
    def _sample_onsets_per_beat(beat_frames_arr: np.ndarray, onset_env_arr: np.ndarray) -> np.ndarray:
        if onset_env_arr.ndim == 0 or beat_frames_arr.size == 0:
            return np.zeros(0, dtype=float)
        vals: List[float] = []
        last = onset_env_arr.shape[0] - 1
        for f in beat_frames_arr:
            idx = int(f)
            if idx < 0:
                idx = 0
            if idx > last:
                idx = last
            vals.append(float(onset_env_arr[idx]))
        return np.asarray(vals, dtype=float)

    def _norm_autocorr_at_lag(seq: np.ndarray, lag: int) -> float:
        L = int(seq.shape[0])
        if L < lag + 2 or lag <= 0:
            return 0.0
        s = (seq - (seq.mean() if L > 0 else 0.0))
        num = float(np.dot(s[:-lag], s[lag:]))
        den = float(np.dot(s, s)) + 1e-9
        val = num / den
        return max(val, 0.0)

    # Build sequences: onsets, flux, and chord-change impulse train
    beat_onsets_seq = _sample_onsets_per_beat(beat_frames, onset_env)
    beat_flux_seq = None
    if mel is not None and mel.size > 0 and beat_frames.size > 1:
        mel_db = librosa.power_to_db(mel, ref=np.max)
        diff = np.diff(mel_db, axis=1)
        flux_full = np.maximum(diff, 0.0).sum(axis=0)
        # Sample at beat frames
        last_idx_ff = flux_full.shape[0] - 1
        bf: List[float] = []
        for f in beat_frames:
            idx = int(f)
            if idx < 1:
                idx = 1
            if idx > last_idx_ff:
                idx = last_idx_ff
            bf.append(float(flux_full[idx]))
        beat_flux_seq = np.asarray(bf, dtype=float)
    # Changes impulse
    Lbeats = len(beat_chords)
    changes_imp = np.zeros(Lbeats, dtype=float)
    change_idxs = [i for i in range(1, Lbeats) if beat_chords[i] != beat_chords[i - 1]]
    for i in change_idxs:
        changes_imp[i] = 1.0

    periodicity_scores: dict = {}
    for n in candidates:
        if n <= 0:
            periodicity_scores[n] = 0.0
            continue
        p_on = _norm_autocorr_at_lag(beat_onsets_seq, n) if beat_onsets_seq.size else 0.0
        p_fx = _norm_autocorr_at_lag(beat_flux_seq, n) if isinstance(beat_flux_seq, np.ndarray) and beat_flux_seq.size else 0.0
        p_ch = _norm_autocorr_at_lag(changes_imp, n) if changes_imp.size else 0.0
        # Weighted average
        periodicity_scores[n] = 0.4 * p_on + 0.3 * p_fx + 0.3 * p_ch

    # Run-length multiples score: favor meters where chord runs are multiples of n
    def _run_multiple_fraction(beat_chords_local: List[str], n_local: int) -> float:
        if n_local <= 0 or not beat_chords_local:
            return 0.0
        runs: List[int] = []
        cur = beat_chords_local[0]
        cnt = 1
        for x in beat_chords_local[1:]:
            if x == cur:
                cnt += 1
            else:
                runs.append(cnt)
                cur = x
                cnt = 1
        runs.append(cnt)
        if not runs:
            return 0.0
        mult = sum(1 for r in runs if (r % n_local) == 0)
        return mult / len(runs)

    runmult_scores = {n: _run_multiple_fraction(beat_chords, n) for n in candidates}
    def normalize(d: dict) -> dict:
        vals = list(d.values())
        if not vals:
            return {k: 0.0 for k in d}
        vmin, vmax = min(vals), max(vals)
        if vmax - vmin < 1e-9:
            return {k: 0.0 for k in d}
        return {k: (v - vmin) / (vmax - vmin) for k, v in d.items()}
    phrase_n = normalize(phrase_scores)
    accent_n = normalize(accent_scores)
    homo_n = normalize(homo_scores)
    flux_n = normalize(flux_scores)
    periodicity_n = normalize(periodicity_scores)
    runmult_n = normalize(runmult_scores)
    lf_n = normalize(lf_scores)
    rb_n = normalize(rootbass_scores)
    # Blend (loosen 3/4 gates): reduce homogeneity weight, raise periodicity weight
    # New blend: phrase 0.05, accent 0.22, homogeneity 0.10, flux 0.15, periodicity 0.30, run-multiples 0.08, low-bass 0.10
    total_scores = {
        n: 0.05 * phrase_n.get(n, 0.0)
        + 0.22 * accent_n.get(n, 0.0)
        + 0.10 * homo_n.get(n, 0.0)
        + 0.15 * flux_n.get(n, 0.0)
        + 0.30 * periodicity_n.get(n, 0.0)
        + 0.08 * runmult_n.get(n, 0.0)
        + 0.07 * lf_n.get(n, 0.0)
        + 0.03 * rb_n.get(n, 0.0)
        for n in candidates
    }

    # Apply conservative gate for selecting 3/4: require strong bar-start change alignment
    if 3 in candidates and 3 in total_scores:
        align3 = _change_alignment_ratio(beat_chords, 3)
        if align3 < 0.5:
            total_scores[3] *= 0.8  # lighter penalty for weaker 3-beat alignment
        print(f"[DEBUG] 🥁 3/4 alignment ratio={align3:.2f} (>=0.50 to avoid penalty)")

    print(f"[DEBUG] 🥁 Meter scoring - phrase={phrase_scores}, accent={accent_scores}, homo={homo_scores}, flux={flux_scores}, periodicity={periodicity_scores}, runmult={runmult_scores}, lf={lf_scores}, rb={rootbass_scores}, total(pre)={total_scores}")
    if debug_dict is not None:
        debug_dict.update({
            "phrase": phrase_scores,
            "accent": accent_scores,
            "homogeneity": homo_scores,
            "flux": flux_scores,
            "periodicity": periodicity_scores,
            "run_multiples": runmult_scores,
            "low_bass": lf_scores,
            "root_bass": rootbass_scores,
        })

    # Require margin for 3/4 over others to reduce false positives
    if 3 in total_scores:
        best_other = max([v for k, v in total_scores.items() if k != 3] or [0.0])
        # Additional periodicity gate (looser): 3-beat periodicity should be at least comparable to 2-beat
        p3 = periodicity_scores.get(3, 0.0)
        p2 = periodicity_scores.get(2, 0.0)
        if p3 < p2 + 0.01:
            total_scores[3] *= 0.95
        # Remove strict margin requirement; only dampen slightly if very close
        if total_scores[3] < best_other + 0.01:
            total_scores[3] *= 0.99

    best_n = max(total_scores.items(), key=lambda kv: (kv[1], kv[0] == 4, kv[0] == 2))[0]

    # Final waltz check: if 3 beats shows clearly stronger structural evidence than 2, choose 3/4
    if 3 in candidates and 2 in candidates:
        align2 = _change_alignment_ratio(beat_chords, 2)
        align3 = _change_alignment_ratio(beat_chords, 3)
        per2 = periodicity_scores.get(2, 0.0)
        per3 = periodicity_scores.get(3, 0.0)
        homo2 = homo_scores.get(2, 0.0)
        homo3 = homo_scores.get(3, 0.0)
        flux2 = flux_scores.get(2, 0.0)
        flux3 = flux_scores.get(3, 0.0)
        # Require multiple cues to favor 3: stronger bar-start alignment and periodicity and not worse homogeneity
        if (align3 - align2) > 0.08 and (per3 - per2) > 0.02 and (homo3 + 1e-6) >= (homo2 - 1e-6) and (flux3 + 1e-6) >= (flux2 - 1e-6):
            print(f"[DEBUG] 🥁 Waltz override triggered: align3={align3:.2f} per3={per3:.2f} homo3={homo3:.2f} vs 2-beat align2={align2:.2f} per2={per2:.2f} homo2={homo2:.2f}")
            best_n = 3
        if debug_dict is not None:
            debug_dict.update({
                "align2": align2,
                "align3": align3,
                "per2": per2,
                "per3": per3,
            })

    print(f"[DEBUG] 🥁 Auto meter selection -> {best_n}/4 (scores={total_scores})")
    if debug_dict is not None:
        debug_dict["total_scores"] = total_scores
        debug_dict["selected"] = int(best_n)
    return int(best_n)


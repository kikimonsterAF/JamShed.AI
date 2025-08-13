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
        )


def _analyze_path(
    src_path: str,
    instrument: Optional[str],
    difficulty: Optional[str],
    beats_per_bar: Optional[int],
    meter: Optional[str],
    use_aisu: Optional[int] = None,
    use_deep: Optional[int] = None,
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
    chroma, beat_frames = _compute_beatsynced_chroma(y, sr)

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
            beat_chords = []
            for chord in chords_progression:
                beat_chords.extend([chord] * 4)  # 4 beats per chord
            print(f"[DEBUG] 🎵 Deep transformer beat_chords created: {len(beat_chords)} beats")
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
            # Create beat-level chords for form analysis (repeat each chord 4 times for 4/4)
            beat_chords = []
            for chord in chords_progression:
                beat_chords.extend([chord] * 4)  # 4 beats per chord
            print(f"[DEBUG] AISU beat_chords created: {len(beat_chords)} beats")
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
    selected_bpb = user_bpb or _select_phrase_len(beat_chords)

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
    return AnalysisResponse(
        chords=chords_progression[:16],
        form=form_labels[:8],
        scale_suggestions=suggestions,
        beats_per_bar=selected_bpb,
        meter=meter_label,
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


def _compute_beatsynced_chroma(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute chroma-cqt and aggregate per beat for chord inference."""
    y = librosa.effects.harmonic(y)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    if beats.size == 0:
        # Fallback: frame-wise pooling into fixed windows
        frames = chroma.shape[1]
        window = max(1, frames // 32)
        pools = [chroma[:, i : i + window].mean(axis=1) for i in range(0, frames, window)]
        return np.stack(pools, axis=1), np.arange(len(pools))
    beat_chroma = librosa.util.sync(chroma, beats, aggregate=np.mean)
    return beat_chroma, beats


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


def _select_phrase_len(beat_chords: List[str], candidates: Tuple[int, ...] = (2, 3, 4, 5, 6)) -> int:
    """Auto-select beats-per-bar by choosing phrase length with most repetition and lowest remainder.

    Scoring: score = unique_phrases_ratio + remainder_penalty, lower is better.
    """
    if not beat_chords:
        return 4
    best_len = 4
    best_score = float("inf")
    total = len(beat_chords)
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
        if score < best_score:
            best_score = score
            best_len = n
    return best_len



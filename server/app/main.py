from fastapi import FastAPI, UploadFile, File, HTTPException
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
        mf = MutagenFile(src_path)
        _ = float(getattr(mf.info, "length", 0.0) or 0.0)  # duration if available (not currently returned)
    except Exception:
        pass
    finally:
        try:
            file.file.close()
        except Exception:
            pass

    # Decode to mono PCM WAV (22.05 kHz) via ffmpeg
    y, sr = _decode_audio_with_ffmpeg(src_path, target_sr=22050)
    if y.size == 0:
        raise HTTPException(status_code=400, detail="Failed to decode audio. Ensure FFmpeg is installed and the file is valid.")

    # Chroma features and beat-synchronous pooling
    chroma, beat_frames = _compute_beatsynced_chroma(y, sr)

    # Key estimation using Krumhansl-Schmuckler profiles
    key_center, key_mode = _estimate_key_from_chroma(chroma)

    # Chord sequence (major/minor triads) per beat block; collapse repeats for readability
    beat_chords = _infer_chords_from_chroma(chroma)
    chords_progression = _collapse_repeats(beat_chords)

    # Simple form detection: chunk progression into 4-chord phrases, label repeats as A/B/C
    form_labels = _infer_form_from_chords(beat_chords)

    # Scale suggestions (basic): key minor pentatonic and Mixolydian per chord root
    unique_chords = list(dict.fromkeys(chords_progression))[:8]  # limit to first few unique for response brevity
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

    # Cleanup temp file
    try:
        os.remove(src_path)
    except Exception:
        pass

    return AnalysisResponse(chords=chords_progression[:16], form=form_labels[:8], scale_suggestions=suggestions)


# ---------- Analysis helpers ----------

_NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _decode_audio_with_ffmpeg(src_path: str, target_sr: int = 22050) -> Tuple[np.ndarray, int]:
    """Decode arbitrary media to mono float32 PCM at target_sr using ffmpeg CLI. Falls back to librosa/soundfile on failure."""
    out_fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(out_fd)
    try:
        cmd = [
            "ffmpeg",
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


def _infer_form_from_chords(beat_chords: List[str]) -> List[str]:
    """Naive form labeling: hash phrases of length 4 and assign A/B/C by first occurrence."""
    phrase_len = 4
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



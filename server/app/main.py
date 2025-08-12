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

app = FastAPI(title="JamTab API", version="0.1.0")


class ScaleSuggestion(BaseModel):
    section_id: str
    chord: str
    key_center: str
    recommended_scales: List[str]
    scale_degrees: List[str]
    target_notes: List[str]


class SectionSummary(BaseModel):
    label: str
    chords: List[str]
    roman_chords: List[str]


class AnalysisResponse(BaseModel):
    chords: List[str]
    form: List[str]
    scale_suggestions: List[ScaleSuggestion]
    # Estimated meter information (auto or user-provided override echoed back)
    beats_per_bar: Optional[int] = None
    meter: Optional[str] = None
    section_summaries: List[SectionSummary] = []


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
        )


def _analyze_path(
    src_path: str,
    instrument: Optional[str],
    difficulty: Optional[str],
    beats_per_bar: Optional[int],
    meter: Optional[str],
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

    # Chords and collapsed progression
    beat_chords = _infer_chords_from_chroma(chroma)
    chords_progression = _collapse_repeats(beat_chords)

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

    # Form labels: derive roman-degree tokens and group by larger multi-bar phrases (auto select 2/4/8 bars)
    roman_tokens = _map_chords_to_roman(beat_chords, key_center, key_mode)
    bars_per_phrase = _select_bars_per_phrase(roman_tokens, (selected_bpb or 4), candidates=(2, 4, 8))
    phrase_len_for_form = max(1, (selected_bpb or 4) * bars_per_phrase)
    off_tokens = _select_best_offset_tokens(roman_tokens, phrase_len_for_form)
    form_labels = _infer_form_from_tokens(roman_tokens[off_tokens:], phrase_len=phrase_len_for_form, similarity_threshold=0.75)

    # Build section summaries (first occurrence exemplar per label)
    section_map = {}
    section_summaries: List[SectionSummary] = []
    # Slice original chords/romans by the same offset and phrase window
    chords_offset = beat_chords[off_tokens:]
    for idx, lbl in enumerate(form_labels):
        start = idx * phrase_len_for_form
        end = start + phrase_len_for_form
        if start >= len(chords_offset):
            break
        ch_slice = chords_offset[start:end]
        rm_slice = roman_tokens[off_tokens + start: off_tokens + end]
        if lbl not in section_map:
            section_map[lbl] = True
            section_summaries.append(
                SectionSummary(label=lbl, chords=_collapse_repeats(ch_slice), roman_chords=_collapse_repeats(rm_slice))
            )

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
        section_summaries=section_summaries,
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


def _pcset_for_chord(label: str) -> List[int]:
    # Return pitch class set for a simple major/minor triad chord label like 'G' or 'Gm'
    root_name = label[0].upper()
    acc = label[1] if len(label) > 1 and label[1] in ['#', 'b'] else ''
    is_minor = label.endswith('m')
    root_idx = _NOTE_NAMES_SHARP.index(root_name + acc) if (root_name + acc) in _NOTE_NAMES_SHARP else _NOTE_NAMES_SHARP.index(root_name)
    third = 3 if is_minor else 4
    pcs = [(root_idx + x) % 12 for x in [0, third, 7]]
    return pcs


def _map_chords_to_roman(chords: List[str], key_center: str, key_mode: str) -> List[str]:
    # Map chord roots to roman numerals in the detected key to be invariant to transposition/quality
    scale = [0, 2, 4, 5, 7, 9, 11]
    root_idx = _NOTE_NAMES_SHARP.index(key_center)
    roman_names_major = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
    roman_names_minor = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii']
    roman_names = roman_names_major if key_mode == 'major' else roman_names_minor
    tokens: List[str] = []
    for ch in chords:
        # normalize to root pitch class
        rn = ch[0].upper()
        acc = ch[1] if len(ch) > 1 and ch[1] in ['#', 'b'] else ''
        name = rn + acc
        try:
            pc = _NOTE_NAMES_SHARP.index(name)
        except ValueError:
            pc = _NOTE_NAMES_SHARP.index(rn)
        rel = (pc - root_idx) % 12
        # pick closest scale degree
        diffs = [(abs((rel - s) % 12), idx) for idx, s in enumerate(scale)]
        deg_idx = min(diffs, key=lambda t: t[0])[1]
        tokens.append(roman_names[deg_idx])
    return tokens


def _select_best_offset_tokens(tokens: List[str], phrase_len: int) -> int:
    n = len(tokens)
    if n == 0 or phrase_len <= 1:
        return 0
    best_off = 0
    best_score = float('inf')
    for off in range(min(phrase_len, n)):
        segs = [tuple(tokens[i:i+phrase_len]) for i in range(off, n, phrase_len)]
        if not segs:
            continue
        unique_ratio = len(set(segs)) / len(segs)
        tail = (n - off) % phrase_len
        head = off % phrase_len
        score = unique_ratio + 0.25 * ((tail + head) / phrase_len)
        if score < best_score:
            best_score = score
            best_off = off
    return best_off


def _infer_form_from_tokens(tokens: List[str], phrase_len: int, similarity_threshold: float = 0.7) -> List[str]:
    def sim(a: List[str], b: List[str]) -> float:
        L = min(len(a), len(b))
        if L == 0:
            return 0.0
        return sum(1 for x, y in zip(a[:L], b[:L]) if x == y) / float(L)
    labels: List[str] = []
    exemplars: List[List[str]] = []
    next_label_ord = ord('A')
    for i in range(0, len(tokens), phrase_len):
        phrase = tokens[i:i+phrase_len]
        if not phrase:
            continue
        best_idx = -1
        best_sim = 0.0
        for idx, ex in enumerate(exemplars):
            s = sim(phrase, ex)
            if s > best_sim:
                best_sim = s
                best_idx = idx
        if best_idx >= 0 and best_sim >= similarity_threshold:
            labels.append(chr(ord('A') + best_idx))
        else:
            exemplars.append(phrase)
            labels.append(chr(next_label_ord))
            next_label_ord += 1
    return labels


def _select_bars_per_phrase(tokens: List[str], beats_per_bar: int, candidates: Tuple[int, ...] = (2, 4, 8)) -> int:
    if beats_per_bar <= 0:
        beats_per_bar = 4
    n = len(tokens)
    if n == 0:
        return 4
    best_bp = 4
    best_score = float('inf')
    for bars in candidates:
        L = beats_per_bar * bars
        if L <= 0:
            continue
        off = _select_best_offset_tokens(tokens, L)
        segs = [tuple(tokens[i:i+L]) for i in range(off, n, L)]
        if not segs:
            continue
        unique_ratio = len(set(segs)) / len(segs)
        remainder = (n - off) % L
        remainder_penalty = remainder / L
        score = unique_ratio + 0.25 * remainder_penalty
        if score < best_score:
            best_score = score
            best_bp = bars
    return best_bp

def _infer_form_from_chords(beat_chords: List[str], phrase_len: int = 4, similarity_threshold: float = 0.6) -> List[str]:
    """Form labeling with fuzzy matching over phrases of length phrase_len.

    - Normalize chords to roots (e.g., 'Gm' -> 'G') to reduce minor/quality noise
    - Reuse a label if normalized Hamming similarity >= threshold with any prior exemplar
    """
    def normalize(ch: str) -> str:
        ch = ch.strip()
        if not ch:
            return ch
        root = ch[0].upper()
        acc = ch[1] if len(ch) > 1 and ch[1] in ['#', 'b'] else ''
        return root + acc

    def similarity(a: List[str], b: List[str]) -> float:
        L = min(len(a), len(b))
        if L == 0:
            return 0.0
        matches = sum(1 for x, y in zip(a[:L], b[:L]) if normalize(x) == normalize(y))
        return matches / float(L)

    labels: List[str] = []
    exemplars: List[List[str]] = []
    next_label_ord = ord("A")
    for i in range(0, len(beat_chords), phrase_len):
        phrase = beat_chords[i : i + phrase_len]
        if not phrase:
            continue
        # Compare with existing exemplars
        best_idx = -1
        best_sim = 0.0
        for idx, ex in enumerate(exemplars):
            s = similarity(phrase, ex)
            if s > best_sim:
                best_sim = s
                best_idx = idx
        if best_idx >= 0 and best_sim >= similarity_threshold:
            labels.append(chr(ord('A') + best_idx))
        else:
            exemplars.append(phrase)
            labels.append(chr(next_label_ord))
            next_label_ord += 1
    return labels


def _select_phrase_len(beat_chords: List[str], candidates: Tuple[int, ...] = (2, 3, 4, 5, 6)) -> int:
    """Auto-select beats-per-bar by choosing phrase length with most repetition and lowest remainder.

    Scoring: score = unique_phrases_ratio + remainder_penalty, lower is better.
    """
    if not beat_chords:
        return 4
    best_len = 4
    best_score = float("inf")
    eps_tie = 1e-6
    near_tie_margin = 0.02  # prefer 3/4 when within this margin
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
        # Strictly better
        if score < best_score - eps_tie:
            best_score = score
            best_len = n
        else:
            # Tie or near-tie: prefer 3 beats per bar
            if abs(score - best_score) <= near_tie_margin:
                if n == 3 and best_len != 3:
                    best_score = score
                    best_len = n
    return best_len


def _normalize_chord_root(ch: str) -> str:
    ch = ch.strip()
    if not ch:
        return ch
    root = ch[0].upper()
    acc = ch[1] if len(ch) > 1 and ch[1] in ['#', 'b'] else ''
    return root + acc


def _select_best_offset(beat_chords: List[str], phrase_len: int) -> int:
    """Choose an offset (0..phrase_len-1) that maximizes repetition of phrases.

    We evaluate the number of unique normalized phrases for each offset and pick the lowest.
    Tie-breaker prefers smaller offsets.
    """
    n = len(beat_chords)
    if n == 0 or phrase_len <= 1:
        return 0
    best_offset = 0
    best_score = float("inf")
    for off in range(min(phrase_len, n)):
        phrases = []
        for i in range(off, n, phrase_len):
            seg = tuple(_normalize_chord_root(c) for c in beat_chords[i : i + phrase_len])
            if len(seg) == 0:
                continue
            phrases.append(seg)
        if not phrases:
            continue
        unique_ratio = len(set(phrases)) / len(phrases)
        # Penalize head/tail remainders lightly to prefer better alignment
        tail = (n - off) % phrase_len
        head = off % phrase_len
        remainder_penalty = 0.25 * ((tail + head) / phrase_len)
        score = unique_ratio + remainder_penalty
        if score < best_score:
            best_score = score
            best_offset = off
    return best_offset


def _infer_form_with_best_offset(beat_chords: List[str], phrase_len: int, similarity_threshold: float = 0.6) -> List[str]:
    if phrase_len <= 0:
        return []
    off = _select_best_offset(beat_chords, phrase_len)
    # Shift sequence by offset for segmentation
    seq = beat_chords[off:]
    labels = _infer_form_from_chords(seq, phrase_len=phrase_len, similarity_threshold=similarity_threshold)
    return labels


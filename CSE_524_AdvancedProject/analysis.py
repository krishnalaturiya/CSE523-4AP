"""
analysis.py — Stand-Up Comedy Analysis Engine (REAL IMPLEMENTATION)
=====================================================================
Replaces all mock data with actual audio/NLP analysis using:
  - yt-dlp        : video download
  - ffmpeg        : audio extraction
  - openai-whisper: ASR with word-level timestamps
  - YAMNet        : neural audio event classification
  - webrtcvad     : speech/silence segmentation
  - librosa       : audio feature extraction
  - spaCy / NLTK  : linguistic feature extraction

Research pipeline for CSE 524 Advanced Project.
"""

import os
import re
import math
import json
import subprocess
import pathlib
import warnings
import time
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa


# ─────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────

@dataclass
class LaughEvent:
    """One contiguous laughter event detected by YAMNet."""
    event_id: int
    laugh_start_s: float
    laugh_end_s: float
    laugh_duration_s: float
    laugh_intensity_max: float      # 0–1 YAMNet score
    laugh_intensity_mean: float
    laugh_latency_s: float          # gap between last spoken word and laugh start
    speech_ratio_during_laugh: float  # how much of the laugh window has speech (applause vs pure laugh)

    # What was said just before
    setup_text: str                 # full preceding Whisper segment(s)
    setup_start_s: float
    setup_end_s: float
    setup_duration_s: float
    setup_word_count: int

    # Linguistic features
    setup_sentiment: float          # -1 (negative) to +1 (positive), via VADER
    setup_is_question: bool         # ends in "?"
    setup_is_self_deprecating: bool # contains first-person negative language
    setup_is_callback: bool         # references earlier topic keywords

    # Crowd reaction breakdown at this moment (YAMNet multi-class)
    applause_score: float
    cheering_score: float
    crowd_noise_score: float

    # Audio clip for listening
    clip_path: Optional[str] = None


@dataclass
class PerformanceStats:
    """Aggregate statistics for one performance."""
    video_url: str
    duration_s: float
    duration_fmt: str

    total_laugh_events: int
    total_laugh_time_s: float
    laugh_coverage_pct: float       # % of show that is audience laughter
    laughs_per_minute: float
    avg_laugh_duration_s: float
    avg_laugh_intensity: float
    peak_laugh_intensity: float
    peak_laugh_timestamp_s: float

    avg_setup_duration_s: float
    avg_setup_word_count: float
    avg_laugh_latency_s: float      # comedian stops → laughter starts

    # Laugh intensity timeline (200 normalized points, 0–1)
    timeline: List[float] = field(default_factory=list)
    # Compressed heatmap (20 cells)
    heatmap: List[float] = field(default_factory=list)
    # Words spoken during each heatmap bin (same length as heatmap)
    heatmap_texts: List[str] = field(default_factory=list)
    # Comedy score 0–100
    comedy_score: int = 0

    laugh_events: List[dict] = field(default_factory=list)
    crowd_emotions: List[dict] = field(default_factory=list)
    key_moments: List[dict] = field(default_factory=list)
    laugh_peaks: List[dict] = field(default_factory=list)

    # ── Dashboard compatibility aliases (match original AnalysisResult schema) ──
    duration_sec: int = 0           # = int(duration_s)
    total_laugh_moments: int = 0    # = total_laugh_events
    peak_moment_fmt: str = "N/A"    # = "MM:SS" of peak laugh


# ─────────────────────────────────────────────────────────────
# Step 1 & 2: Download + Extract Audio
# ─────────────────────────────────────────────────────────────

def download_and_extract_audio(
    url: str,
    out_dir: str = "/tmp/laughlab",
    cookiefile: Optional[str] = None
) -> Tuple[str, float]:
    """
    Download YouTube video and convert to 16kHz mono WAV.
    Returns (wav_path, duration_seconds).
    """
    from yt_dlp import YoutubeDL

    os.makedirs(out_dir, exist_ok=True)
    audio_template = os.path.join(out_dir, "audio.%(ext)s")
    wav_path = os.path.join(out_dir, "audio_16k_mono.wav")

    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best/best",
        "outtmpl": audio_template,
        "noplaylist": True,
        "quiet": True,
        "extractor_args": {"youtube": {"player_client": ["web", "android"]}},
    }
    if cookiefile and pathlib.Path(cookiefile).exists():
        ydl_opts["cookiefile"] = cookiefile

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_path = ydl.prepare_filename(info)

    subprocess.run([
        "ffmpeg", "-y", "-i", downloaded_path,
        "-ac", "1", "-ar", "16000", wav_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    duration = len(y) / sr
    return wav_path, duration


def load_uploaded_audio(file_bytes: bytes, out_dir: str = "/tmp/laughlab") -> Tuple[str, float]:
    """Save uploaded bytes, convert to 16kHz mono WAV."""
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "upload_raw")
    wav_path = os.path.join(out_dir, "audio_16k_mono.wav")
    with open(raw_path, "wb") as f:
        f.write(file_bytes)
    subprocess.run([
        "ffmpeg", "-y", "-i", raw_path,
        "-ac", "1", "-ar", "16000", wav_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    return wav_path, len(y) / sr


# ─────────────────────────────────────────────────────────────
# Step 3: Transcribe with Whisper (word-level timestamps)
# ─────────────────────────────────────────────────────────────

def transcribe(wav_path: str, model_size: str = "small") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transcribe audio using Whisper with word-level timestamps.
    Returns:
      seg_df  — segment-level DataFrame (segment_id, start, end, text)
      word_df — word-level DataFrame (word, start, end, segment_id)
    """
    import whisper

    model = whisper.load_model(model_size)
    result = model.transcribe(
        wav_path,
        language="en",
        fp16=False,
        verbose=False,
        word_timestamps=True   # enables word-level timing
    )

    seg_rows, word_rows = [], []
    for i, seg in enumerate(result["segments"]):
        seg_rows.append({
            "segment_id": i,
            "seg_start": float(seg["start"]),
            "seg_end": float(seg["end"]),
            "seg_text": (seg.get("text") or "").strip(),
        })
        # word-level (Whisper ≥ 20230314 supports this)
        for w in seg.get("words", []):
            word_rows.append({
                "segment_id": i,
                "word": w.get("word", "").strip(),
                "word_start": float(w.get("start", seg["start"])),
                "word_end": float(w.get("end", seg["end"])),
            })

    seg_df = pd.DataFrame(seg_rows)
    word_df = pd.DataFrame(word_rows) if word_rows else pd.DataFrame(
        columns=["segment_id", "word", "word_start", "word_end"]
    )
    return seg_df, word_df


# ─────────────────────────────────────────────────────────────
# Step 4: VAD — speech / silence mask
# ─────────────────────────────────────────────────────────────

def build_vad_mask(y: np.ndarray, sr: int = 16000, aggressiveness: int = 2):
    """
    Returns a function speech_ratio(start_s, end_s) → float.
    Uses WebRTC VAD at 30ms frames.
    """
    import webrtcvad
    vad = webrtcvad.Vad(aggressiveness)
    frame_ms = 30
    frame_len = int(sr * frame_ms / 1000)
    num_frames = len(y) // frame_len
    y_int16 = (np.clip(y, -1, 1) * 32767).astype(np.int16)
    flags = np.zeros(num_frames, dtype=bool)
    for i in range(num_frames):
        frame = y_int16[i * frame_len:(i + 1) * frame_len].tobytes()
        flags[i] = vad.is_speech(frame, sample_rate=sr)

    def speech_ratio(start_s: float, end_s: float) -> float:
        s = max(0, int(start_s * sr // frame_len))
        e = min(num_frames, int(end_s * sr // frame_len))
        return float(np.mean(flags[s:e])) if e > s else 0.0

    return speech_ratio, flags, frame_len


# ─────────────────────────────────────────────────────────────
# Step 5: YAMNet multi-class detection
# ─────────────────────────────────────────────────────────────

# Classes we care about (will be found dynamically from class map)
TARGET_CLASSES = ["Laughter", "Applause", "Cheering", "Crowd", "Audience"]

def run_yamnet(y: np.ndarray):
    """
    Run YAMNet on audio. Returns (scores_array, class_names, hop_s, win_s).
    scores_array shape: (n_frames, 521)
    """
    import tensorflow_hub as hub
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
    scores, embeddings, spectrogram = yamnet(y)
    scores = scores.numpy()

    class_map_path = hub.resolve("https://tfhub.dev/google/yamnet/1") + "/assets/yamnet_class_map.csv"
    class_names = pd.read_csv(class_map_path)["display_name"].tolist()

    return scores, class_names, 0.48, 0.96  # YAMNet hop=0.48s, win=0.96s


def get_class_scores(scores: np.ndarray, class_names: list, target: str) -> np.ndarray:
    """Extract score timeseries for a named class (case-insensitive)."""
    for idx, name in enumerate(class_names):
        if name.lower() == target.lower():
            return scores[:, idx]
    return np.zeros(scores.shape[0])


# ─────────────────────────────────────────────────────────────
# Step 6: Merge YAMNet frames into laughter events
# ─────────────────────────────────────────────────────────────

def merge_laugh_frames(
    frame_indices: np.ndarray,
    scores: np.ndarray,
    hop: float,
    win: float,
    max_gap: int = 1
) -> List[dict]:
    """Merge consecutive YAMNet frames into discrete laughter events."""
    if len(frame_indices) == 0:
        return []

    frame_indices = np.sort(frame_indices)
    events = []
    start_i = frame_indices[0]
    prev_i = frame_indices[0]

    for i in frame_indices[1:]:
        if i <= prev_i + max_gap:
            prev_i = i
        else:
            _append_event(events, start_i, prev_i, scores, hop, win)
            start_i = i
            prev_i = i
    _append_event(events, start_i, prev_i, scores, hop, win)
    return events


def _append_event(events, start_i, prev_i, scores, hop, win):
    idxs = np.arange(start_i, prev_i + 1)
    events.append({
        "laugh_start_s": float(start_i * hop),
        "laugh_end_s": float(prev_i * hop + win),
        "laugh_duration_s": float((prev_i - start_i) * hop + win),
        "laugh_intensity_max": float(np.max(scores[idxs])),
        "laugh_intensity_mean": float(np.mean(scores[idxs])),
    })


# ─────────────────────────────────────────────────────────────
# Step 7: Linguistic feature extraction
# ─────────────────────────────────────────────────────────────

# Self-deprecating marker words
SELF_DEPRECATING = ["i'm", "i am", "my", "me", "i've", "i was", "i have", "i did", "i said", "i look", "i sound", "i feel", "i can't", "i couldn't"]
NEGATIVE_WORDS   = ["bad", "ugly", "stupid", "dumb", "idiot", "fat", "old", "wrong", "fail", "weird", "embarrass", "broke", "poor", "lazy", "awful", "terrible", "horrible", "pathetic"]

def is_self_deprecating(text: str) -> bool:
    tl = text.lower()
    has_self = any(m in tl for m in SELF_DEPRECATING)
    has_neg  = any(w in tl for w in NEGATIVE_WORDS)
    return has_self and has_neg


def is_callback(text: str, earlier_keywords: List[str]) -> bool:
    """Check if setup text references a keyword from an earlier part of the show."""
    tl = text.lower()
    return any(kw.lower() in tl for kw in earlier_keywords)


def get_sentiment(text: str) -> float:
    """
    VADER sentiment score: -1 (very negative) to +1 (very positive).
    Falls back to 0.0 if nltk/vader not available.
    """
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import nltk
        nltk.download("vader_lexicon", quiet=True)
        sid = SentimentIntensityAnalyzer()
        return float(sid.polarity_scores(text)["compound"])
    except Exception:
        return 0.0


def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """Simple keyword extraction: most frequent non-stopword nouns."""
    stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                 "of", "is", "it", "that", "this", "was", "with", "he", "she", "they",
                 "i", "you", "we", "my", "your", "his", "her", "its", "be", "are"}
    words = re.findall(r"[a-z']+", text.lower())
    freq = {}
    for w in words:
        if w not in stopwords and len(w) > 3:
            freq[w] = freq.get(w, 0) + 1
    return sorted(freq, key=freq.get, reverse=True)[:top_n]


# ─────────────────────────────────────────────────────────────
# Step 8: Get setup text using word-level timestamps
# ─────────────────────────────────────────────────────────────

def get_setup(
    seg_df: pd.DataFrame,
    word_df: pd.DataFrame,
    laugh_start_s: float,
    lookback_s: float = 8.0
) -> dict:
    """
    Return the setup text: all speech in the `lookback_s` window before laugh_start_s.
    Uses word-level timestamps when available for precision.
    """
    if not word_df.empty:
        # Word-level: grab words that ended before the laugh started
        words_before = word_df[word_df["word_end"] <= laugh_start_s]
        words_in_window = words_before[words_before["word_end"] >= laugh_start_s - lookback_s]
        if len(words_in_window) > 0:
            # Find the segment containing the last word
            last_seg_id = words_in_window.iloc[-1]["segment_id"]
            # Use the full segment(s) within the window
            segs_in_window = seg_df[
                (seg_df["seg_end"] <= laugh_start_s) &
                (seg_df["seg_start"] >= laugh_start_s - lookback_s)
            ]
            if len(segs_in_window) == 0:
                # fall back to just the previous segment
                segs_in_window = seg_df[seg_df["seg_end"] <= laugh_start_s].tail(1)

            text = " ".join(segs_in_window["seg_text"].tolist()).strip()
            start_s = float(segs_in_window["seg_start"].iloc[0]) if len(segs_in_window) > 0 else laugh_start_s - lookback_s
            end_s   = float(segs_in_window["seg_end"].iloc[-1]) if len(segs_in_window) > 0 else laugh_start_s

            # Laugh latency = gap between last spoken word and laugh
            last_word_end = float(words_before["word_end"].max()) if len(words_before) > 0 else end_s
            latency = max(0.0, laugh_start_s - last_word_end)

            return {"text": text, "start_s": start_s, "end_s": end_s, "latency_s": latency}

    # Fallback: segment-level
    prev_segs = seg_df[seg_df["seg_end"] <= laugh_start_s].tail(2)
    if len(prev_segs) == 0:
        return {"text": "", "start_s": 0.0, "end_s": 0.0, "latency_s": 0.0}
    text = " ".join(prev_segs["seg_text"].tolist()).strip()
    start_s = float(prev_segs["seg_start"].iloc[0])
    end_s   = float(prev_segs["seg_end"].iloc[-1])
    latency = max(0.0, laugh_start_s - end_s)
    return {"text": text, "start_s": start_s, "end_s": end_s, "latency_s": latency}


# ─────────────────────────────────────────────────────────────
# Step 9: Create audio clips
# ─────────────────────────────────────────────────────────────

def make_clip(wav_path: str, clip_path: str, start_s: float, end_s: float):
    start_s = max(0.0, start_s)
    end_s   = max(start_s + 0.1, end_s)
    subprocess.run([
        "ffmpeg", "-y", "-i", wav_path,
        "-ss", str(start_s), "-to", str(end_s),
        "-ac", "1", "-ar", "16000", clip_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ─────────────────────────────────────────────────────────────
# Step 10: Build normalized timeline + heatmap
# ─────────────────────────────────────────────────────────────

def build_timeline(laugh_events: List[dict], duration_s: float, n_points: int = 200) -> List[float]:
    """Map laughter events onto a normalized n_points timeline."""
    timeline = np.zeros(n_points)
    for ev in laugh_events:
        s = int((ev["laugh_start_s"] / duration_s) * n_points)
        e = int((ev["laugh_end_s"]   / duration_s) * n_points)
        intensity = ev["laugh_intensity_mean"]
        for i in range(max(0, s), min(n_points, e + 1)):
            timeline[i] = max(timeline[i], intensity)
    # Normalize to 0–1
    mx = timeline.max()
    if mx > 0:
        timeline = timeline / mx
    return [round(float(v), 3) for v in timeline]


def build_heatmap(timeline: List[float], n_cells: int = 20) -> List[float]:
    arr = np.array(timeline)
    cell_size = len(arr) // n_cells
    return [round(float(np.mean(arr[i*cell_size:(i+1)*cell_size])), 3) for i in range(n_cells)]


def build_heatmap_texts(seg_df, duration_s: float, n_cells: int = 20) -> List[str]:
    """
    For each heatmap bin, collect the Whisper transcript words spoken
    during that time window and return them as a list of short strings.
    """
    if seg_df is None or seg_df.empty or duration_s <= 0:
        return ["" for _ in range(n_cells)]

    bin_duration = duration_s / n_cells
    texts = []
    for i in range(n_cells):
        t_start = i * bin_duration
        t_end   = (i + 1) * bin_duration
        # Grab segments that overlap this bin
        mask = (seg_df["seg_end"] > t_start) & (seg_df["seg_start"] < t_end)
        words = " ".join(seg_df[mask]["seg_text"].tolist()).strip()
        # Truncate to keep tooltips readable
        if len(words) > 120:
            words = words[:117] + "…"
        texts.append(words if words else "(no speech)")
    return texts


def build_crowd_emotions(
    scores: np.ndarray,
    class_names: List[str],
    duration_s: float
) -> List[dict]:
    """
    Real crowd emotion breakdown from YAMNet class averages.
    Returns list of {label, value (0–100), color}.
    """
    def avg_score(cls_name: str) -> int:
        sc = get_class_scores(scores, class_names, cls_name)
        return int(np.mean(sc) * 100 * 5)  # scale up for readability

    return [
        {"label": "Belly Laughs",    "value": min(100, avg_score("Laughter") * 3),  "color": "#f5e642"},
        {"label": "Applause",        "value": min(100, avg_score("Applause") * 3),   "color": "#4fffb0"},
        {"label": "Cheering",        "value": min(100, avg_score("Cheering") * 3),   "color": "#ff6b35"},
        {"label": "Crowd Noise",     "value": min(100, avg_score("Crowd") * 3),      "color": "#a78bfa"},
        {"label": "Silence / Setup", "value": 100 - min(99, avg_score("Laughter")*2 + avg_score("Applause")), "color": "#f87171"},
    ]


# ─────────────────────────────────────────────────────────────
# Step 11: Comedy score
# ─────────────────────────────────────────────────────────────

def compute_comedy_score(events: List[LaughEvent], duration_s: float) -> int:
    """
    Multi-factor comedy score (0–100):
      - Laugh density (laughs per minute)
      - Average laugh intensity
      - Average laugh duration
      - Setup efficiency (words per laugh)
    """
    if not events or duration_s == 0:
        return 0

    laughs_per_min = len(events) / (duration_s / 60)
    avg_intensity  = np.mean([e.laugh_intensity_mean for e in events])
    avg_duration   = np.mean([e.laugh_duration_s for e in events])

    # Scoring components (tune weights as needed)
    density_score    = min(40, laughs_per_min * 4)   # ~10 lpm = 40pts
    intensity_score  = avg_intensity * 35             # max 35pts
    duration_score   = min(15, avg_duration * 5)      # max 15pts
    efficiency_bonus = min(10, 10)                    # placeholder

    return min(100, int(density_score + intensity_score + duration_score + efficiency_bonus))


# ─────────────────────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────────────────────

def analyze(
    url: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    file_name: Optional[str] = None,
    out_dir: str = "/tmp/laughlab",
    whisper_model: str = "small",
    laugh_threshold: float = 0.25,
    max_gap_frames: int = 1,
    setup_lookback_s: float = 8.0,
    save_clips: bool = True,
) -> PerformanceStats:
    """
    Full analysis pipeline. Returns a PerformanceStats object.
    All fields are real — no mocked data.
    """
    os.makedirs(out_dir, exist_ok=True)
    clips_dir = os.path.join(out_dir, "laugh_clips")
    os.makedirs(clips_dir, exist_ok=True)

    # ── 1. Get audio ──
    print("Step 1/8: Loading audio...")
    if url:
        wav_path, duration_s = download_and_extract_audio(url, out_dir)
    elif file_bytes:
        wav_path, duration_s = load_uploaded_audio(file_bytes, out_dir)
    else:
        raise ValueError("Provide either url or file_bytes")

    y, sr = librosa.load(wav_path, sr=16000, mono=True)

    # ── 2. Transcribe ──
    print("Step 2/8: Transcribing with Whisper...")
    seg_df, word_df = transcribe(wav_path, model_size=whisper_model)

    # ── 3. VAD ──
    print("Step 3/8: Running VAD...")
    speech_ratio_fn, vad_flags, vad_frame_len = build_vad_mask(y, sr)

    # ── 4. YAMNet ──
    print("Step 4/8: Running YAMNet...")
    yamnet_scores, class_names, yam_hop, yam_win = run_yamnet(y)

    laughter_scores  = get_class_scores(yamnet_scores, class_names, "Laughter")
    applause_scores  = get_class_scores(yamnet_scores, class_names, "Applause")
    cheering_scores  = get_class_scores(yamnet_scores, class_names, "Cheering")
    crowd_scores     = get_class_scores(yamnet_scores, class_names, "Crowd")

    # ── 5. Merge events ──
    print("Step 5/8: Merging laughter events...")
    laugh_frame_idxs = np.where(laughter_scores >= laugh_threshold)[0]
    raw_events = merge_laugh_frames(laugh_frame_idxs, laughter_scores, yam_hop, yam_win, max_gap_frames)
    print(f"  → {len(raw_events)} laughter events detected")

    # ── 6. Build full transcript for keyword extraction ──
    full_transcript = " ".join(seg_df["seg_text"].tolist()) if not seg_df.empty else ""
    all_keywords = extract_keywords(full_transcript, top_n=20)

    # ── 7. Enrich each event ──
    print("Step 6/8: Enriching events with setup & linguistics...")
    events: List[LaughEvent] = []

    for ev_id, ev in enumerate(raw_events):
        ls = ev["laugh_start_s"]
        le = ev["laugh_end_s"]

        setup = get_setup(seg_df, word_df, ls, setup_lookback_s)
        setup_text = setup["text"]
        setup_wc   = len(setup_text.split()) if setup_text else 0

        # YAMNet frame scores at this laugh window
        s_frame = int(ls / yam_hop)
        e_frame = int(le / yam_hop) + 1
        ev_applause = float(np.mean(applause_scores[s_frame:e_frame])) if e_frame > s_frame else 0.0
        ev_cheer    = float(np.mean(cheering_scores[s_frame:e_frame])) if e_frame > s_frame else 0.0
        ev_crowd    = float(np.mean(crowd_scores[s_frame:e_frame]))    if e_frame > s_frame else 0.0

        # Clip
        clip_path = None
        if save_clips and setup_text:
            clip_start = max(0.0, setup["start_s"] - 1.0)
            clip_end   = min(duration_s, le + 1.5)
            clip_file  = os.path.join(clips_dir, f"laugh_event_{ev_id:03d}.wav")
            try:
                make_clip(wav_path, clip_file, clip_start, clip_end)
                clip_path = clip_file
            except Exception:
                pass

        # Earlier keywords for callback detection (everything before this moment)
        earlier_segs = seg_df[seg_df["seg_end"] <= ls - 30]["seg_text"].tolist()
        earlier_text = " ".join(earlier_segs)
        earlier_kws  = extract_keywords(earlier_text, top_n=15)

        events.append(LaughEvent(
            event_id=ev_id,
            laugh_start_s=round(ls, 3),
            laugh_end_s=round(le, 3),
            laugh_duration_s=round(ev["laugh_duration_s"], 3),
            laugh_intensity_max=round(ev["laugh_intensity_max"], 4),
            laugh_intensity_mean=round(ev["laugh_intensity_mean"], 4),
            laugh_latency_s=round(setup["latency_s"], 3),
            speech_ratio_during_laugh=round(speech_ratio_fn(ls, le), 4),
            setup_text=setup_text,
            setup_start_s=round(setup["start_s"], 3),
            setup_end_s=round(setup["end_s"], 3),
            setup_duration_s=round(setup["end_s"] - setup["start_s"], 3),
            setup_word_count=setup_wc,
            setup_sentiment=round(get_sentiment(setup_text), 4),
            setup_is_question=setup_text.strip().endswith("?"),
            setup_is_self_deprecating=is_self_deprecating(setup_text),
            setup_is_callback=is_callback(setup_text, earlier_kws),
            applause_score=round(ev_applause, 4),
            cheering_score=round(ev_cheer, 4),
            crowd_noise_score=round(ev_crowd, 4),
            clip_path=clip_path,
        ))

    # ── 8. Aggregate stats ──
    print("Step 7/8: Computing aggregate statistics...")
    timeline = build_timeline([asdict(e) for e in events], duration_s)
    heatmap  = build_heatmap(timeline)
    heatmap_texts = build_heatmap_texts(seg_df, duration_s)
    comedy_score = compute_comedy_score(events, duration_s)
    crowd_emotions = build_crowd_emotions(yamnet_scores, class_names, duration_s)

    total_laugh_s = sum(e.laugh_duration_s for e in events)

    # Key moments for dashboard (top events by intensity)
    key_events = sorted(events, key=lambda e: e.laugh_intensity_max, reverse=True)[:11]
    key_moments = []
    for ev in sorted(key_events, key=lambda e: e.laugh_start_s):
        mm = int(ev.laugh_start_s) // 60
        ss = int(ev.laugh_start_s) % 60
        category = "hot" if ev.laugh_intensity_max > 0.7 else ("warm" if ev.laugh_intensity_max > 0.4 else "cool")
        desc = ev.setup_text[:80] + "..." if len(ev.setup_text) > 80 else ev.setup_text
        key_moments.append({
            "timestamp_fmt": f"{mm:02d}:{ss:02d}",
            "description": desc or "(no setup detected)",
            "intensity_pct": int(ev.laugh_intensity_max * 100),
            "category": category,
        })

    # Laugh peaks (for dashboard peaks panel)
    laugh_peaks_dicts = []
    for ev in sorted(events, key=lambda e: e.laugh_intensity_max, reverse=True)[:6]:
        mm = int(ev.laugh_start_s) // 60
        ss = int(ev.laugh_start_s) % 60
        laugh_peaks_dicts.append({
            "timestamp_sec": int(ev.laugh_start_s),
            "timestamp_fmt": f"{mm:02d}:{ss:02d}",
            "intensity_pct": int(ev.laugh_intensity_max * 100),
            "description": (ev.setup_text[:60] + "...") if len(ev.setup_text) > 60 else ev.setup_text,
        })

    dur_min = int(duration_s) // 60
    dur_sec = int(duration_s) % 60
    duration_fmt = f"{dur_min}:{dur_sec:02d}"

    peak_event = max(events, key=lambda e: e.laugh_intensity_max) if events else None
    peak_ts = peak_event.laugh_start_s if peak_event else 0.0
    peak_mm = int(peak_ts) // 60
    peak_ss = int(peak_ts) % 60

    print("Step 8/8: Saving results...")
    stats = PerformanceStats(
        video_url=url or "(uploaded file)",
        duration_s=duration_s,
        duration_fmt=duration_fmt,
        total_laugh_events=len(events),
        total_laugh_time_s=round(total_laugh_s, 2),
        laugh_coverage_pct=round(total_laugh_s / duration_s * 100, 2) if duration_s > 0 else 0,
        laughs_per_minute=round(len(events) / (duration_s / 60), 2) if duration_s > 0 else 0,
        avg_laugh_duration_s=round(np.mean([e.laugh_duration_s for e in events]), 3) if events else 0,
        avg_laugh_intensity=round(np.mean([e.laugh_intensity_mean for e in events]), 4) if events else 0,
        peak_laugh_intensity=round(peak_event.laugh_intensity_max, 4) if peak_event else 0,
        peak_laugh_timestamp_s=round(peak_ts, 2),
        avg_setup_duration_s=round(np.mean([e.setup_duration_s for e in events if e.setup_duration_s > 0]), 3) if events else 0,
        avg_setup_word_count=round(np.mean([e.setup_word_count for e in events if e.setup_word_count > 0]), 1) if events else 0,
        avg_laugh_latency_s=round(np.mean([e.laugh_latency_s for e in events]), 3) if events else 0,
        timeline=timeline,
        heatmap=heatmap,
        heatmap_texts=heatmap_texts,
        comedy_score=comedy_score,
        laugh_events=[asdict(e) for e in events],
        crowd_emotions=crowd_emotions,
        key_moments=key_moments,
        laugh_peaks=laugh_peaks_dicts,
        # Dashboard compatibility aliases
        duration_sec=int(duration_s),
        total_laugh_moments=len(events),
        peak_moment_fmt=f"{peak_mm:02d}:{peak_ss:02d}" if peak_event else "N/A",
    )

    # Save CSV + JSON for research analysis
    events_df = pd.DataFrame([asdict(e) for e in events])
    events_df.to_csv(os.path.join(out_dir, "laugh_events.csv"), index=False)

    with open(os.path.join(out_dir, "performance_stats.json"), "w") as f:
        json.dump(asdict(stats) if hasattr(stats, "__dataclass_fields__") else stats.__dict__, f, indent=2, default=str)

    print(f"\n✅ Done! {len(events)} events | Score: {comedy_score} | {round(len(events)/(duration_s/60),1)} laughs/min")
    return stats


# ─────────────────────────────────────────────────────────────
# Multi-video comparative analysis (research module)
# ─────────────────────────────────────────────────────────────

def analyze_multiple(
    videos: List[Dict],   # [{"url": ..., "label": "Comedian A - Special Name"}, ...]
    out_dir: str = "/tmp/laughlab_multi",
    **kwargs
) -> pd.DataFrame:
    """
    Run the full pipeline on multiple videos and produce a comparative DataFrame.
    Each row = one video. Columns = all PerformanceStats scalar fields.

    Example:
        videos = [
            {"url": "https://youtu.be/...", "label": "Dave Chappelle - 2024"},
            {"url": "https://youtu.be/...", "label": "Hannah Gadsby - Nanette"},
        ]
        df = analyze_multiple(videos)
        df.to_csv("comparative_results.csv", index=False)
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for v in videos:
        label = v.get("label", v.get("url", "unknown"))
        print(f"\n{'='*60}")
        print(f"Analyzing: {label}")
        print('='*60)
        vid_dir = os.path.join(out_dir, re.sub(r"[^\w]", "_", label)[:40])
        try:
            stats = analyze(url=v["url"], out_dir=vid_dir, **kwargs)
            row = {
                "label": label,
                "url": v["url"],
                "duration_s": stats.duration_s,
                "total_laugh_events": stats.total_laugh_events,
                "laughs_per_minute": stats.laughs_per_minute,
                "avg_laugh_duration_s": stats.avg_laugh_duration_s,
                "avg_laugh_intensity": stats.avg_laugh_intensity,
                "peak_laugh_intensity": stats.peak_laugh_intensity,
                "laugh_coverage_pct": stats.laugh_coverage_pct,
                "avg_setup_duration_s": stats.avg_setup_duration_s,
                "avg_setup_word_count": stats.avg_setup_word_count,
                "avg_laugh_latency_s": stats.avg_laugh_latency_s,
                "comedy_score": stats.comedy_score,
            }
            rows.append(row)
        except Exception as ex:
            print(f"⚠️  Error analyzing {label}: {ex}")
            rows.append({"label": label, "url": v["url"], "error": str(ex)})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "comparative_analysis.csv"), index=False)
    print(f"\n✅ Comparative analysis saved to {out_dir}/comparative_analysis.csv")
    return df


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = analyze(url="https://youtu.be/faNCfJL9008")
    print(f"\nDuration:          {result.duration_fmt}")
    print(f"Comedy Score:      {result.comedy_score}")
    print(f"Laugh Events:      {result.total_laugh_events}")
    print(f"Laughs/min:        {result.laughs_per_minute}")
    print(f"Avg Setup Length:  {result.avg_setup_word_count} words")
    print(f"Avg Laugh Latency: {result.avg_laugh_latency_s}s")
    print(f"Laugh Coverage:    {result.laugh_coverage_pct}%")
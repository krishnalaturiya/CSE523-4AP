"""
analyze_segments.py — Segment-level analysis for multi-part videos.
Add this to analysis.py, or import from here.
"""

from analysis import (
    download_and_extract_audio, transcribe, build_vad_mask,
    run_yamnet, get_class_scores, merge_laugh_frames,
    get_setup, get_sentiment, is_self_deprecating, is_callback,
    extract_keywords, build_timeline, build_heatmap, build_heatmap_texts,
    build_crowd_emotions, compute_comedy_score,
    LaughEvent, PerformanceStats
)
from punchline_detector import (
    detect_punchlines_llm,
    load_cached_punchlines,
    save_punchlines_cache,
    score_punchlines_per_segment,
)
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Dict
import numpy as np
import librosa
import logging
import os
import subprocess

log = logging.getLogger("laughlab")


def download_video_file(url: str, out_dir: str) -> Optional[str]:
    """
    Download the best MP4 video (with audio) from a YouTube URL.
    Returns path to downloaded file, or None on failure.
    """
    from yt_dlp import YoutubeDL
    os.makedirs(out_dir, exist_ok=True)
    video_path = os.path.join(out_dir, "video_full.mp4")
    if os.path.exists(video_path):
        print("Video already downloaded, reusing.")
        return video_path
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": video_path,
        "noplaylist": True,
        "quiet": True,
        "merge_output_format": "mp4",
        "extractor_args": {"youtube": {"player_client": ["web", "android"]}},
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return video_path
    except Exception as e:
        print(f"⚠️  Video download failed (clips won't be available): {e}")
        return None


def cut_segment_clip(video_path: str, start_s: float, end_s: float, out_path: str) -> bool:
    """Cut a segment from video_path using ffmpeg. Returns True on success."""
    duration = end_s - start_s
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_s),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264", "-c:a", "aac",
        "-movflags", "+faststart",
        out_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


@dataclass
class SegmentResult:
    label: str           # e.g. "Part 1"
    start_s: float
    end_s: float
    duration_s: float
    total_laugh_events: int
    laughs_per_minute: float
    avg_laugh_intensity: float
    peak_laugh_intensity: float
    laugh_coverage_pct: float
    avg_laugh_latency_s: float
    avg_setup_word_count: float
    comedy_score: int
    timeline: List[float]
    heatmap: List[float]
    heatmap_texts: List[str]
    crowd_emotions: List[dict]
    laugh_events: List[dict]
    winner: bool = False   # True for the segment with highest audience reaction
    clip_url: Optional[str] = None   # URL to serve the video clip for this segment


def analyze_video_segments(
    url: str,
    segments: List[Dict],   # [{"label": "Part 1", "start_s": 0, "end_s": 51}, ...]
    out_dir: str = "/tmp/laughlab_segments",
    whisper_model: str = "small",
    laugh_threshold: float = 0.25,
) -> Dict:
    """
    Download video once, then analyse each segment slice independently.
    Returns dict with keys: segments (List[SegmentResult]), overall_winner_index (int)
    """
    import warnings
    warnings.filterwarnings("ignore")

    log.info(f"[AUDIO] analyze_video_segments called — url={url}, segments={len(segments)}, out_dir={out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    # --- Step 1: Download once ---
    log.info("[AUDIO] Step 1: Downloading audio...")
    print("Downloading audio...")
    wav_path, total_duration = download_and_extract_audio(url, out_dir)
    log.info(f"[AUDIO] Audio downloaded: {wav_path}, duration={total_duration:.1f}s")
    y_full, sr = librosa.load(wav_path, sr=16000, mono=True)
    log.info(f"[AUDIO] Loaded waveform: {len(y_full)} samples @ {sr}Hz")

    # --- Step 1b: Download video for segment clips ---
    clips_dir = os.path.join(out_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    log.info("[AUDIO] Step 1b: Downloading video for clips...")
    print("Downloading video for segment clips...")
    video_path = download_video_file(url, out_dir) if url else None
    log.info(f"[AUDIO] Video path: {video_path}")

    # --- Step 2: Transcribe once ---
    log.info("[AUDIO] Step 2: Transcribing with Whisper...")
    print("Transcribing...")
    seg_df, word_df = transcribe(wav_path, model_size=whisper_model)
    log.info(f"[AUDIO] Transcription done: {len(seg_df)} segments, {len(word_df)} words")

    # --- Step 3: VAD once ---
    log.info("[AUDIO] Step 3: Running VAD...")
    print("Running VAD...")
    speech_ratio_fn, _, _ = build_vad_mask(y_full, sr)
    log.info("[AUDIO] VAD done")

    # --- Step 4: YAMNet once ---
    log.info("[AUDIO] Step 4: Running YAMNet...")
    print("Running YAMNet...")
    yamnet_scores, class_names, yam_hop, yam_win = run_yamnet(y_full)
    laughter_scores = get_class_scores(yamnet_scores, class_names, "Laughter")
    applause_scores = get_class_scores(yamnet_scores, class_names, "Applause")
    cheering_scores = get_class_scores(yamnet_scores, class_names, "Cheering")
    crowd_scores    = get_class_scores(yamnet_scores, class_names, "Crowd")
    log.info(f"[AUDIO] YAMNet done: {len(laughter_scores)} laugh frames, hop={yam_hop:.3f}s")

    full_transcript = " ".join(seg_df["seg_text"].tolist()) if not seg_df.empty else ""

    # --- Step 5: Analyse each segment ---
    results: List[SegmentResult] = []
    seg_df_slices: List = []          # collect time-shifted seg_dfs for punchline matching

    for seg in segments:
        label   = seg["label"]
        start_s = float(seg["start_s"])
        end_s   = float(seg["end_s"])
        dur     = end_s - start_s
        log.info(f"[AUDIO] Processing segment: {label} ({start_s:.0f}s – {end_s:.0f}s)")
        print(f"\nAnalysing {label} ({start_s:.0f}s – {end_s:.0f}s)...")

        # Slice YAMNet frames for this segment
        frame_start = int(start_s / yam_hop)
        frame_end   = int(end_s   / yam_hop) + 1
        seg_laugh_scores = laughter_scores[frame_start:frame_end]

        # Laugh frame indices (relative to segment start frame)
        laugh_idxs_rel = np.where(seg_laugh_scores >= laugh_threshold)[0]
        # Convert back to absolute frame indices for merging
        laugh_idxs_abs = laugh_idxs_rel + frame_start

        from analysis import merge_laugh_frames as _merge
        raw_events = _merge(laugh_idxs_abs, laughter_scores, yam_hop, yam_win, max_gap=1)

        # Filter events to within this segment window
        raw_events = [e for e in raw_events
                      if e["laugh_start_s"] >= start_s and e["laugh_end_s"] <= end_s + 1.0]

        # Enrich events
        events: List[LaughEvent] = []
        earlier_kws = extract_keywords(full_transcript, top_n=15)

        for ev_id, ev in enumerate(raw_events):
            ls, le = ev["laugh_start_s"], ev["laugh_end_s"]
            setup  = get_setup(seg_df, word_df, ls, lookback_s=8.0)
            setup_text = setup["text"]

            s_f = int(ls / yam_hop)
            e_f = int(le / yam_hop) + 1
            ev_applause = float(np.mean(applause_scores[s_f:e_f])) if e_f > s_f else 0.0
            ev_cheer    = float(np.mean(cheering_scores[s_f:e_f])) if e_f > s_f else 0.0
            ev_crowd    = float(np.mean(crowd_scores[s_f:e_f]))    if e_f > s_f else 0.0

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
                setup_word_count=len(setup_text.split()) if setup_text else 0,
                setup_sentiment=round(get_sentiment(setup_text), 4),
                setup_is_question=setup_text.strip().endswith("?"),
                setup_is_self_deprecating=is_self_deprecating(setup_text),
                setup_is_callback=is_callback(setup_text, earlier_kws),
                applause_score=round(ev_applause, 4),
                cheering_score=round(ev_cheer, 4),
                crowd_noise_score=round(ev_crowd, 4),
            ))

        # Build timeline relative to segment (0 → dur)
        shifted_events = []
        for e in events:
            d = asdict(e)
            d["laugh_start_s"] = d["laugh_start_s"] - start_s
            d["laugh_end_s"]   = d["laugh_end_s"]   - start_s
            shifted_events.append(d)

        timeline = build_timeline(shifted_events, dur, n_points=100)
        heatmap  = build_heatmap(timeline, n_cells=10)

        # Slice seg_df to this segment's time window, shift timestamps to 0-based
        seg_df_slice = seg_df[
            (seg_df["seg_end"] > start_s) & (seg_df["seg_start"] < end_s)
        ].copy()
        seg_df_slice["seg_start"] = seg_df_slice["seg_start"] - start_s
        seg_df_slice["seg_end"]   = seg_df_slice["seg_end"]   - start_s
        heatmap_texts = build_heatmap_texts(seg_df_slice, dur, n_cells=10)
        seg_df_slices.append(seg_df_slice)   # keep for punchline matching

        # Crowd emotions for this segment slice
        seg_yamnet_slice = yamnet_scores[frame_start:frame_end]
        crowd_emotions = build_crowd_emotions(seg_yamnet_slice, class_names, dur)

        comedy_score = compute_comedy_score(events, dur)
        total_laugh_s = sum(e.laugh_duration_s for e in events)

        log.info(f"[AUDIO]   {label}: {len(events)} laugh events, score={comedy_score}")
        results.append(SegmentResult(
            label=label,
            start_s=start_s,
            end_s=end_s,
            duration_s=round(dur, 2),
            total_laugh_events=len(events),
            laughs_per_minute=round(len(events) / (dur / 60), 2) if dur > 0 else 0,
            avg_laugh_intensity=round(np.mean([e.laugh_intensity_mean for e in events]), 4) if events else 0,
            peak_laugh_intensity=round(max((e.laugh_intensity_max for e in events), default=0), 4),
            laugh_coverage_pct=round(total_laugh_s / dur * 100, 2) if dur > 0 else 0,
            avg_laugh_latency_s=round(np.mean([e.laugh_latency_s for e in events]), 3) if events else 0,
            avg_setup_word_count=round(np.mean([e.setup_word_count for e in events if e.setup_word_count > 0]), 1) if events else 0,
            comedy_score=comedy_score,
            timeline=timeline,
            heatmap=heatmap,
            heatmap_texts=heatmap_texts,
            crowd_emotions=crowd_emotions,
            laugh_events=[asdict(e) for e in events],
        ))

        # --- Cut video clip for this segment ---
        if video_path:
            safe_label = label.replace(" ", "_").replace("/", "-")
            clip_filename = f"clip_{safe_label}.mp4"
            clip_out = os.path.join(clips_dir, clip_filename)
            if not os.path.exists(clip_out):
                print(f"  Cutting clip for {label}...")
                success = cut_segment_clip(video_path, start_s, end_s, clip_out)
                if success:
                    results[-1].clip_url = f"/api/clips/{clip_filename}"
                else:
                    print(f"  ⚠️  ffmpeg clip failed for {label}")
            else:
                print(f"  Clip already exists for {label}, reusing.")
                results[-1].clip_url = f"/api/clips/{clip_filename}"

    # Pick winner by peak laugh intensity
    if results:
        best_idx = max(range(len(results)), key=lambda i: results[i].comedy_score)
        results[best_idx].winner = True

    # --- Step 6: Punchline detection & scoring ---
    punchline_results = []
    try:
        # Build full transcript for LLM (from whole video seg_df)
        full_transcript = " ".join(seg_df["seg_text"].tolist()) if not seg_df.empty else ""

        # Try cache first (keyed by URL so re-runs skip the LLM call)
        punchlines = load_cached_punchlines(out_dir, url) if url else None
        if punchlines is None:
            if full_transcript.strip():
                punchlines = detect_punchlines_llm(full_transcript)
                if url:
                    save_punchlines_cache(out_dir, url, punchlines)
            else:
                punchlines = []

        if punchlines:
            print(f"Scoring {len(punchlines)} punchlines across {len(segments)} segments...")
            punchline_results = score_punchlines_per_segment(
                punchlines=punchlines,
                segments=segments,
                seg_dfs=seg_df_slices,
                laughter_scores=laughter_scores,
                applause_scores=applause_scores,
                yam_hop=yam_hop,
                window_s=8.0,
            )
            print(f"✅ Punchline scoring complete.")

            # --- Cut punchline sub-clips ---
            if video_path and punchline_results:
                print("Cutting punchline sub-clips...")
                for pl in punchline_results:
                    pl_id = pl["id"]
                    pl_label_safe = pl["label"].replace(" ", "_").replace("/", "-")[:30]
                    for seg_data in pl["segments"]:
                        if not seg_data.get("found"):
                            continue
                        seg_label = seg_data["label"]
                        seg_label_safe = seg_label.replace(" ", "_")
                        win_start = seg_data.get("win_start_s", 0)
                        win_end   = seg_data.get("win_end_s",   8)
                        # Find absolute times by looking up segment start
                        seg_meta = next((s for s in segments if s["label"] == seg_label), None)
                        if not seg_meta:
                            continue
                        abs_start = float(seg_meta["start_s"]) + win_start
                        abs_end   = float(seg_meta["start_s"]) + win_end
                        clip_filename = f"pl_{pl_id}_{pl_label_safe}_{seg_label_safe}.mp4"
                        clip_out = os.path.join(clips_dir, clip_filename)
                        if not os.path.exists(clip_out):
                            success = cut_segment_clip(video_path, abs_start, abs_end, clip_out)
                            if success:
                                seg_data["punchline_clip_url"] = f"/api/clips/{clip_filename}"
                        else:
                            seg_data["punchline_clip_url"] = f"/api/clips/{clip_filename}"
    except Exception as ex:
        log.error(f"[AUDIO] Punchline detection failed (non-fatal): {ex}", exc_info=True)
        print(f"⚠️  Punchline detection failed (non-fatal): {ex}")
        punchline_results = []

    log.info(f"[AUDIO] All segments processed. Writing segment_analysis.json to {out_dir}")
    import json
    out = {
        "url": url,
        "segments": [r.__dict__ for r in results],
        "overall_winner_index": next((i for i, r in enumerate(results) if r.winner), 0),
        "punchline_analysis": punchline_results,
    }
    with open(os.path.join(out_dir, "segment_analysis.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    log.info(f"[AUDIO] Done. Winner: {results[best_idx].label}")
    print(f"\n✅ Segment analysis complete. Winner: {results[best_idx].label}")
    return out
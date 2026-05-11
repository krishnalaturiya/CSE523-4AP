"""
punchline_detector.py — LLM-based punchline detection for LaughLab
====================================================================
Uses the groq API to identify recurring punchlines across a
comedian's full transcript, then measures audience reaction
(laugh + applause intensity) around each punchline in each segment.

Results are cached to disk so the LLM is only called once per video.
"""

import os
import re
import json
import hashlib
from typing import List, Dict, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────

def _cache_key(url: str) -> str:
    """Short stable cache key from URL."""
    return hashlib.md5(url.encode()).hexdigest()[:12]


def _punchline_cache_path(cache_dir: str, url: str) -> str:
    return os.path.join(cache_dir, f"punchlines_{_cache_key(url)}.json")


def load_cached_punchlines(cache_dir: str, url: str) -> Optional[List[dict]]:
    path = _punchline_cache_path(cache_dir, url)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} punchlines from cache: {path}")
        return data
    return None


def save_punchlines_cache(cache_dir: str, url: str, punchlines: List[dict]):
    os.makedirs(cache_dir, exist_ok=True)
    path = _punchline_cache_path(cache_dir, url)
    with open(path, "w") as f:
        json.dump(punchlines, f, indent=2)
    print(f"💾 Punchlines cached to: {path}")


# ─────────────────────────────────────────────────────────────
# LLM: detect punchlines from transcript
# ─────────────────────────────────────────────────────────────

def detect_punchlines_llm(full_transcript: str) -> List[dict]:
    """
    Call Groq (llama-3.3-70b-versatile) to find recurring punchlines.

    Returns a list of dicts:
      {
        "id": 1,
        "label": "Short name for the punchline",
        "description": "What the punchline is about",
        "keywords": ["keyword1", "keyword2", ...]
      }
    """
    from groq import Groq

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    system_prompt = """You are a comedy analyst. Given a transcript of a stand-up comedian
telling the SAME JOKE multiple times in different ways across 5 segments, your job is to
identify the 3–6 distinct recurring PUNCHLINES or comedic beats that appear (in different
wordings) across all or most segments.

A punchline is the climax/payoff of a comedic setup — the moment the audience is expected
to laugh. Even if the wording changes each time, it is the SAME punchline if it delivers the
same core joke or observation.

Respond ONLY with a valid JSON array. No preamble, no markdown, no explanation.
Each element must have:
  - "id": integer starting at 1
  - "label": short label (max 6 words) summarising the punchline
  - "description": one sentence explaining what the punchline is about
  - "keywords": array of 3–6 keywords or short phrases that would appear in transcript text
    near THIS punchline (used for fuzzy matching). Be specific — avoid generic words.

Example output format:
[
  {
    "id": 1,
    "label": "Douchebag/cult escalation",
    "description": "The observation that 1 person in a hat looks normal, 2 are douchebags, 3 is a cult.",
    "keywords": ["douchebag", "cult", "two people", "three people", "wearing a hat"]
  }
]"""

    user_prompt = f"""Here is the full transcript of the comedian's set. Identify the recurring punchlines:

---TRANSCRIPT START---
{full_transcript[:12000]}
---TRANSCRIPT END---

Return ONLY a JSON array as described."""

    print("🤖 Calling Groq API to detect punchlines...")
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1500,
    )

    raw = response.choices[0].message.content.strip()

    # Strip any accidental markdown fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    punchlines = json.loads(raw)
    print(f"✅ Groq detected {len(punchlines)} punchlines")
    return punchlines


# ─────────────────────────────────────────────────────────────
# Match punchline to a segment's transcript
# ─────────────────────────────────────────────────────────────

def find_punchline_windows(
    seg_df,          # segment-level DataFrame for one segment (already time-shifted to 0-based)
    punchline: dict,
    window_s: float = 8.0,
) -> List[Tuple[float, float]]:
    """
    Search the segment transcript for text matching this punchline's keywords.
    Returns list of (match_start_s, match_end_s) windows where the punchline
    likely occurs — each window is the transcript segment containing a keyword match
    ± window_s/2.

    We look for the highest-density keyword match within the segment text.
    """
    if seg_df is None or seg_df.empty:
        return []

    keywords = [kw.lower() for kw in punchline.get("keywords", [])]
    if not keywords:
        return []

    matches = []
    for _, row in seg_df.iterrows():
        text_lower = (row["seg_text"] or "").lower()
        hit_count = sum(1 for kw in keywords if kw in text_lower)
        if hit_count >= 1:
            # Center the reaction window on this transcript segment
            mid = (float(row["seg_start"]) + float(row["seg_end"])) / 2
            win_start = max(0.0, mid - window_s / 2)
            win_end   = mid + window_s / 2
            matches.append((win_start, win_end, hit_count))

    if not matches:
        return []

    # Return the window(s) with the most keyword hits (top 1)
    matches.sort(key=lambda x: -x[2])
    best = matches[0]
    return [(best[0], best[1])]


# ─────────────────────────────────────────────────────────────
# Measure audience reaction in a time window
# ─────────────────────────────────────────────────────────────

def measure_reaction_in_window(
    laughter_scores: np.ndarray,
    applause_scores: np.ndarray,
    yam_hop: float,
    seg_start_s: float,       # absolute start of segment in full audio
    win_start_s: float,       # relative to segment start
    win_end_s: float,         # relative to segment start
) -> dict:
    """
    Given a time window (relative to segment start), extract the
    max laughter + applause YAMNet scores, and a combined reaction score.

    Returns dict with: laugh_peak, applause_peak, combined_peak, laugh_mean, applause_mean
    """
    abs_start = seg_start_s + win_start_s
    abs_end   = seg_start_s + win_end_s

    f_start = max(0, int(abs_start / yam_hop))
    f_end   = max(f_start + 1, int(abs_end / yam_hop) + 1)

    laugh_slice   = laughter_scores[f_start:f_end]
    applause_slice = applause_scores[f_start:f_end]

    if len(laugh_slice) == 0:
        return {
            "laugh_peak": 0.0, "applause_peak": 0.0,
            "combined_peak": 0.0, "laugh_mean": 0.0, "applause_mean": 0.0,
            "found": False,
        }

    laugh_peak    = float(np.max(laugh_slice))
    applause_peak = float(np.max(applause_slice))
    laugh_mean    = float(np.mean(laugh_slice))
    applause_mean = float(np.mean(applause_slice))
    combined_peak = float(np.max(laugh_slice + applause_slice))

    return {
        "laugh_peak": round(laugh_peak, 4),
        "applause_peak": round(applause_peak, 4),
        "combined_peak": round(combined_peak, 4),
        "laugh_mean": round(laugh_mean, 4),
        "applause_mean": round(applause_mean, 4),
        "found": True,
    }


# ─────────────────────────────────────────────────────────────
# Main entry: score all punchlines across all segments
# ─────────────────────────────────────────────────────────────

def score_punchlines_per_segment(
    punchlines: List[dict],
    segments: List[dict],          # list of {"label", "start_s", "end_s"}
    seg_dfs: List,                 # one seg_df per segment (time-shifted to 0-based)
    laughter_scores: np.ndarray,
    applause_scores: np.ndarray,
    yam_hop: float,
    window_s: float = 8.0,
) -> List[dict]:
    """
    For each punchline, for each segment: find where it appears and
    measure the audience reaction there.

    Returns a list of punchline result dicts:
    {
      "id": 1,
      "label": "...",
      "description": "...",
      "segments": [
        {
          "label": "Part 1",
          "found": bool,
          "laugh_peak": 0.xx,
          "applause_peak": 0.xx,
          "combined_peak": 0.xx,
          "laugh_mean": 0.xx,
          "applause_mean": 0.xx,
        }, ...
      ]
    }
    """
    results = []

    for pl in punchlines:
        pl_result = {
            "id": pl["id"],
            "label": pl["label"],
            "description": pl["description"],
            "keywords": pl.get("keywords", []),
            "segments": [],
        }

        for i, (seg_meta, seg_df) in enumerate(zip(segments, seg_dfs)):
            seg_label   = seg_meta["label"]
            seg_start_s = float(seg_meta["start_s"])

            # Find where this punchline appears in this segment's transcript
            windows = find_punchline_windows(seg_df, pl, window_s=window_s)

            if not windows:
                pl_result["segments"].append({
                    "label": seg_label,
                    "found": False,
                    "laugh_peak": 0.0,
                    "applause_peak": 0.0,
                    "combined_peak": 0.0,
                    "laugh_mean": 0.0,
                    "applause_mean": 0.0,
                })
                continue

            # Use the best window
            win_start, win_end = windows[0]
            reaction = measure_reaction_in_window(
                laughter_scores, applause_scores,
                yam_hop, seg_start_s, win_start, win_end,
            )
            reaction["label"] = seg_label
            reaction["win_start_s"] = round(win_start, 3)   # relative to segment start
            reaction["win_end_s"]   = round(win_end,   3)   # relative to segment start
            pl_result["segments"].append(reaction)

        results.append(pl_result)

    return results
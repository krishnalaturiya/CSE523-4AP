"""
app.py — LaughLab FastAPI Server
"""

import asyncio
import sys

# asyncio.create_subprocess_exec requires ProactorEventLoop on Windows.
# SelectorEventLoop (which uvicorn reload=True switches to) raises
# NotImplementedError with an empty message when subprocess_exec is called.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from dataclasses import asdict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import logging
import os
import json
import tempfile

from analysis import analyze
from analyze_segments import analyze_video_segments, download_video_file
from punchline_detector import load_cached_punchlines

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("laughlab")

app = FastAPI(title="LaughLab API", version="2.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_VIDEO_URL = "https://www.youtube.com/watch?v=WSx2gqeEyLE"

DEFAULT_SEGMENTS = [
    {"label": "Part 1", "start_s":  0,  "end_s":  51},
    {"label": "Part 2", "start_s": 52,  "end_s": 108},
    {"label": "Part 3", "start_s": 109, "end_s": 151},
    {"label": "Part 4", "start_s": 152, "end_s": 202},
    {"label": "Part 5", "start_s": 203, "end_s": 262},
]

VIDEO2_URL = "https://www.youtube.com/watch?v=faNCfJL9008"

VIDEO2_SEGMENTS = [
    {"label": "Part 1", "start_s":  82,  "end_s": 133},   # 01:22 – 02:13
    {"label": "Part 2", "start_s": 134,  "end_s": 174},   # 02:14 – 02:54
    {"label": "Part 3", "start_s": 175,  "end_s": 215},   # 02:55 – 03:35
    {"label": "Part 4", "start_s": 216,  "end_s": 272},   # 03:36 – 04:32
    {"label": "Part 5", "start_s": 273,  "end_s": 320},   # 04:33 – 05:20
]

# Shared output dir for /api/analyze/full — the video MP4 is downloaded here once
# so both the audio task (which caches it) and the gesture subprocess (--video-file)
# avoid making a second simultaneous yt-dlp request.
SHARED_OUT_DIR = "C:\\tmp\\laughlab_full"

# ── Gesture analysis paths ────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
GESTURE_DIR     = os.path.normpath(os.path.join(_HERE, "..", "gestureanalysis"))
GESTURE_PYTHON  = os.path.join(GESTURE_DIR, "mediapipe_env", "Scripts", "python.exe")
GESTURE_SCRIPT  = os.path.join(GESTURE_DIR, "hand_body_movement_detectionnew.py")
GESTURE_RESULTS = os.path.join(GESTURE_DIR, "results.json")

log.info("=== PATH CHECK ===")
log.info(f"  GESTURE_DIR     exists={os.path.isdir(GESTURE_DIR)}    : {GESTURE_DIR}")
log.info(f"  GESTURE_PYTHON  exists={os.path.isfile(GESTURE_PYTHON)} : {GESTURE_PYTHON}")
log.info(f"  GESTURE_SCRIPT  exists={os.path.isfile(GESTURE_SCRIPT)} : {GESTURE_SCRIPT}")
log.info(f"  GESTURE_RESULTS exists={os.path.isfile(GESTURE_RESULTS)}: {GESTURE_RESULTS}")
log.info("===================")


async def _run_gesture(url: str, segs: list, video_path: str = None) -> dict:
    """Run the gesture pipeline in a subprocess and return parsed results.json."""
    import sys as _sys
    import platform as _platform
    import threading as _threading
    import traceback as _tb
    import stat as _stat
    import subprocess as _subprocess

    # ── Environment snapshot ────────────────────────────────────────────────
    log.info("[GESTURE] ============================================================")
    log.info("[GESTURE] _run_gesture() ENTRY")
    log.info(f"[GESTURE]   Python version : {_sys.version}")
    log.info(f"[GESTURE]   Platform       : {_platform.platform()}")
    log.info(f"[GESTURE]   Thread         : {_threading.current_thread().name}"
             f"  is_main={_threading.current_thread() is _threading.main_thread()}")
    try:
        _loop = asyncio.get_running_loop()
        log.info(f"[GESTURE]   Event loop type: {type(_loop).__name__}")
    except Exception as _le:
        log.error(f"[GESTURE]   get_running_loop() FAILED: {_le!r}")

    # ── Path diagnostics ────────────────────────────────────────────────────
    log.info("[GESTURE] --- PATH CHECKS ---")
    for _label, _p in [
        ("GESTURE_PYTHON",  GESTURE_PYTHON),
        ("GESTURE_SCRIPT",  GESTURE_SCRIPT),
        ("GESTURE_RESULTS", GESTURE_RESULTS),
        ("video_path",      video_path or "(not provided)"),
    ]:
        _exists = os.path.exists(_p) if _p else False
        _isfile = os.path.isfile(_p) if _p else False
        log.info(f"[GESTURE]   {_label}")
        log.info(f"[GESTURE]     path    : {_p!r}")
        log.info(f"[GESTURE]     exists  : {_exists}  isfile: {_isfile}")
        if _exists and _isfile:
            try:
                _st = os.stat(_p)
                log.info(f"[GESTURE]     size    : {_st.st_size} bytes")
                log.info(f"[GESTURE]     mode    : {oct(_stat.S_IMODE(_st.st_mode))}")
            except Exception as _se:
                log.error(f"[GESTURE]     stat() failed: {_se!r}")

    if not os.path.isfile(GESTURE_PYTHON):
        raise RuntimeError(f"Gesture Python not found: {GESTURE_PYTHON}")
    if not os.path.isfile(GESTURE_SCRIPT):
        raise RuntimeError(f"Gesture script not found: {GESTURE_SCRIPT}")

    # ── Quick sanity-check: can this Python executable actually run? ─────────
    log.info("[GESTURE] --- PYTHON EXECUTABLE SMOKE TEST (subprocess.run) ---")
    try:
        _tr = _subprocess.run(
            [GESTURE_PYTHON, "--version"],
            capture_output=True, text=True, timeout=15,
        )
        log.info(f"[GESTURE]   returncode : {_tr.returncode}")
        log.info(f"[GESTURE]   stdout     : {_tr.stdout.strip()!r}")
        log.info(f"[GESTURE]   stderr     : {_tr.stderr.strip()!r}")
    except Exception as _te:
        log.error(f"[GESTURE]   smoke test FAILED: {type(_te).__name__}: {_te!r}")
        log.error(f"[GESTURE]   traceback:\n{_tb.format_exc()}")

    # ── Build command ────────────────────────────────────────────────────────
    ranges = [f"{int(s['start_s'])}-{int(s['end_s'])}" for s in segs]
    log.info(f"[GESTURE] URL   : {url}")
    log.info(f"[GESTURE] Ranges: {ranges}")

    cmd = [GESTURE_PYTHON, GESTURE_SCRIPT, url, *ranges, "--output", GESTURE_RESULTS]
    if video_path and os.path.isfile(video_path):
        cmd.extend(["--video-file", video_path])
        log.info(f"[GESTURE] Using pre-downloaded video: {video_path}")

    log.info(f"[GESTURE] --- FULL COMMAND ({len(cmd)} args) ---")
    for _i, _arg in enumerate(cmd):
        log.info(f"[GESTURE]   cmd[{_i}] = {_arg!r}")

    # ── Launch via asyncio.create_subprocess_exec ────────────────────────────
    log.info("[GESTURE] Calling asyncio.create_subprocess_exec() ...")
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        log.info(f"[GESTURE] Subprocess launched — PID={proc.pid}")
    except Exception as e:
        log.error("[GESTURE] *** asyncio.create_subprocess_exec() FAILED ***")
        log.error(f"[GESTURE]   type(e)   : {type(e).__name__}")
        log.error(f"[GESTURE]   repr(e)   : {repr(e)}")
        log.error(f"[GESTURE]   str(e)    : {str(e)!r}")
        log.error(f"[GESTURE]   e.args    : {e.args!r}")
        log.error(f"[GESTURE]   errno     : {getattr(e, 'errno', 'N/A')}")
        log.error(f"[GESTURE]   winerror  : {getattr(e, 'winerror', 'N/A')}")
        log.error(f"[GESTURE]   strerror  : {getattr(e, 'strerror', 'N/A')}")
        log.error(f"[GESTURE]   full traceback:\n{_tb.format_exc()}")
        raise RuntimeError(
            f"Could not launch gesture subprocess: {type(e).__name__}: {repr(e)}"
        )

    # ── Wait for completion ──────────────────────────────────────────────────
    log.info(f"[GESTURE] Waiting for PID={proc.pid} (timeout=1200s) ...")
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=1200)
    except asyncio.TimeoutError:
        proc.kill()
        log.error("[GESTURE] Subprocess timed out after 20 min")
        raise RuntimeError("Gesture analysis timed out (20-min limit).")

    stdout_text = stdout.decode("utf-8", errors="replace").strip()
    stderr_text = stderr.decode("utf-8", errors="replace").strip()

    log.info(f"[GESTURE] Return code : {proc.returncode}")
    log.info(f"[GESTURE] STDOUT ({len(stdout_text)} chars):\n{stdout_text}")
    if stderr_text:
        log.warning(f"[GESTURE] STDERR ({len(stderr_text)} chars):\n{stderr_text}")

    if proc.returncode != 0:
        raise RuntimeError(
            f"Gesture script exited {proc.returncode}.\n"
            f"STDERR: {stderr_text}\n"
            f"STDOUT: {stdout_text}"
        )

    if not os.path.exists(GESTURE_RESULTS):
        log.error(f"[GESTURE] results.json NOT FOUND at: {GESTURE_RESULTS}")
        raise RuntimeError("Gesture script produced no results.json.")

    log.info(f"[GESTURE] Loading results.json: {GESTURE_RESULTS}")
    with open(GESTURE_RESULTS, "r", encoding="utf-8") as f:
        raw = json.load(f)
    log.info(f"[GESTURE] Raw JSON top-level keys: {list(raw.keys())}")

    if "segments" in raw and "per_second_metrics" not in raw:
        log.info("[GESTURE] Normalizing format: segments.*.seconds → per_second_metrics.*")
        data = {
            "per_second_metrics": {
                seg_key: seg_val["seconds"]
                for seg_key, seg_val in raw["segments"].items()
            }
        }
    else:
        data = raw

    seg_count = len(data.get("per_second_metrics", {}))
    log.info(f"[GESTURE] Done — {seg_count} segment(s) in per_second_metrics")
    log.info("[GESTURE] ============================================================")
    return data

@app.get("/")
def serve_dashboard():
    return FileResponse("dashboard.html")

@app.get("/dashboard.css")
def serve_css():
    return FileResponse("dashboard.css", media_type="text/css")

@app.get("/dashboard.js")
def serve_js():
    return FileResponse("dashboard.js", media_type="application/javascript")

@app.get("/api/analyze/default")
async def analyze_default():
    try:
        result = analyze(url=DEFAULT_VIDEO_URL)
        data = asdict(result)
        data["_source"] = "default"
        data["_default_url"] = DEFAULT_VIDEO_URL
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analyze/segments")
async def analyze_segments_default():
    try:
        result = analyze_video_segments(
            url=DEFAULT_VIDEO_URL,
            segments=DEFAULT_SEGMENTS,
            out_dir="C:\\tmp\\laughlab_segments",
            laugh_threshold=0.15
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analyze/segments/video2")
async def analyze_segments_video2():
    try:
        result = analyze_video_segments(
            url=VIDEO2_URL,
            segments=VIDEO2_SEGMENTS,
            out_dir="C:\\tmp\\laughlab_video2",
            laugh_threshold=0.15,
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/segments/custom")
async def analyze_segments_custom(url: str = Form(...), segments: str = Form(...)):
    import json
    try:
        segs = json.loads(segments)
        result = analyze_video_segments(url=url, segments=segs)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/full")
async def analyze_full(url: str = Form(...), segments: str = Form(...)):
    """Run audio + gesture analysis concurrently; return merged results."""
    log.info("=== /api/analyze/full called ===")
    log.info(f"  URL     : {url}")
    log.info(f"  Segments: {segments[:200]}")

    try:
        segs = json.loads(segments)
    except Exception as e:
        log.error(f"  Bad segments JSON: {e}")
        raise HTTPException(status_code=400, detail="segments must be valid JSON.")
    if not segs:
        raise HTTPException(status_code=400, detail="No segments provided.")

    log.info(f"  Parsed {len(segs)} segment(s): {[s['label'] for s in segs]}")

    # Download the video MP4 once before launching either task.
    # - The audio task calls download_video_file internally; it finds the file
    #   already on disk and skips the network request entirely.
    # - The gesture subprocess receives --video-file and never calls yt-dlp.
    # This eliminates two simultaneous downloads of the same URL.
    log.info("[SHARED] Pre-downloading video...")
    shared_video = None
    try:
        shared_video = await asyncio.to_thread(
            lambda: download_video_file(url, SHARED_OUT_DIR)
        )
        log.info(f"[SHARED] Video ready: {shared_video}")
    except Exception as e:
        log.warning(f"[SHARED] Pre-download failed — tasks will download independently: {e}")

    log.info("[AUDIO] Starting audio analysis in thread...")
    audio_task = asyncio.to_thread(
        lambda: analyze_video_segments(url=url, segments=segs, out_dir=SHARED_OUT_DIR)
    )

    log.info("[GESTURE] Starting gesture analysis...")
    gesture_task = _run_gesture(url, segs, video_path=shared_video)

    log.info("Waiting for both tasks to complete...")
    audio_result, gesture_result = await asyncio.gather(
        audio_task, gesture_task, return_exceptions=True
    )

    # Audio result
    if isinstance(audio_result, Exception):
        log.error(f"[AUDIO] FAILED: {audio_result}")
        raise HTTPException(status_code=500, detail=str(audio_result))
    log.info(f"[AUDIO] Success — {len(audio_result.get('segments', []))} segment(s) returned")

    # Gesture result
    if isinstance(gesture_result, Exception):
        log.error(f"[GESTURE] FAILED: {gesture_result}")
        payload = dict(audio_result)
        payload["gesture"] = None
        payload["gesture_error"] = str(gesture_result)
    else:
        log.info("[GESTURE] Success — merging into response")
        payload = dict(audio_result)
        payload["gesture"] = gesture_result

    log.info("=== /api/analyze/full complete ===")
    return JSONResponse(content=payload)

@app.post("/api/analyze/url")
async def analyze_url(url: str = Form(...)):
    if not url or not url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid URL")
    try:
        result = analyze(url=url)
        data = asdict(result)
        data["_source"] = "url"
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/upload")
async def analyze_upload(file: UploadFile = File(...)):
    allowed_types = ["video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")
    try:
        file_bytes = await file.read()
        result = analyze(file_bytes=file_bytes, file_name=file.filename)
        data = asdict(result)
        data["_source"] = "upload"
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/punchlines/video1")
async def get_punchlines_video1():
    cache_path = os.path.join("C:\\tmp\\laughlab_segments", "segment_analysis.json")
    if not os.path.exists(cache_path):
        raise HTTPException(status_code=404, detail="Run /api/analyze/segments first.")
    with open(cache_path) as f:
        data = json.load(f)
    pa = data.get("punchline_analysis", [])
    if not pa:
        raise HTTPException(status_code=404, detail="No punchline data found. Re-run analysis.")
    return JSONResponse(content={"punchline_analysis": pa})

@app.get("/api/punchlines/video2")
async def get_punchlines_video2():
    cache_path = os.path.join("C:\\tmp\\laughlab_video2", "segment_analysis.json")
    if not os.path.exists(cache_path):
        raise HTTPException(status_code=404, detail="Run /api/analyze/segments/video2 first.")
    with open(cache_path) as f:
        data = json.load(f)
    pa = data.get("punchline_analysis", [])
    if not pa:
        raise HTTPException(status_code=404, detail="No punchline data found. Re-run analysis.")
    return JSONResponse(content={"punchline_analysis": pa})

@app.get("/api/clips/{filename}")
async def serve_clip(filename: str):
    """Serve a video clip cut from a segment."""
    for out_dir in [SHARED_OUT_DIR, "C:\\tmp\\laughlab_segments", "C:\\tmp\\laughlab_video2"]:
        clip_path = os.path.join(out_dir, "clips", filename)
        if os.path.exists(clip_path):
            return FileResponse(clip_path, media_type="video/mp4")
    raise HTTPException(status_code=404, detail=f"Clip not found: {filename}")


@app.get("/api/gesture/results")
async def get_gesture_results():
    results_path = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gestureanalysis", "results.json")
    )
    if not os.path.exists(results_path):
        raise HTTPException(
            status_code=404,
            detail="gestureanalysis/results.json not found — run the gesture pipeline first."
        )
    with open(results_path, "r", encoding="utf-8") as f:
        return JSONResponse(content=json.load(f))

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "LaughLab", "default_video": DEFAULT_VIDEO_URL}

if __name__ == "__main__":
    # reload=False is required on Windows — reload=True forces SelectorEventLoop
    # which raises NotImplementedError when asyncio.create_subprocess_exec is called.
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
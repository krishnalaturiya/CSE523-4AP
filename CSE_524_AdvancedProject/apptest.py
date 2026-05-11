"""
apptest.py — Gesture Analysis Subprocess Debugger
==================================================
A minimal FastAPI app for isolating and debugging the gesture analysis pipeline.
Runs the same subprocess call as app.py/_run_gesture() with exhaustive logging.

Start: python apptest.py
Open:  http://localhost:8001
"""

import asyncio
import json
import logging
import os
import traceback
from datetime import datetime
from typing import List

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [APPTEST] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("apptest")

app = FastAPI(title="Gesture Analysis Debugger", version="1.0")

# ─────────────────────────────────────────────────────────────────────────────
# Path configuration — mirrors app.py exactly.
# _HERE resolves to the directory containing this file (CSE_524_AdvancedProject/).
# GESTURE_DIR is one level up, then into gestureanalysis/.
# ─────────────────────────────────────────────────────────────────────────────
_HERE           = os.path.dirname(os.path.abspath(__file__))
GESTURE_DIR     = os.path.normpath(os.path.join(_HERE, "..", "gestureanalysis"))
GESTURE_PYTHON  = os.path.join(GESTURE_DIR, "mediapipe_env", "Scripts", "python.exe")
GESTURE_SCRIPT  = os.path.join(GESTURE_DIR, "hand_body_movement_detectionnew.py")
GESTURE_RESULTS = os.path.join(GESTURE_DIR, "results.json")

# ─────────────────────────────────────────────────────────────────────────────
# Test parameters — fixed inputs so this script is self-contained.
# The URL is intentionally a dummy; the goal is to see how far the subprocess
# gets before failing (path errors, import errors, download errors, etc.).
# ─────────────────────────────────────────────────────────────────────────────
TEST_URL    = "https://www.youtube.com/watch?v=WSx2gqeEyLE"
TEST_RANGES = ["0-30", "45-90"]

# 20-minute timeout, same as app.py
TIMEOUT_SECONDS = 1200


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def now() -> str:
    """HH:MM:SS.mmm timestamp for inline log messages."""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def check_paths() -> dict:
    """
    Verify every path required by the gesture pipeline.
    Returns {name: {path: str, exists: bool}} so the caller can log and inspect.
    GESTURE_DIR uses isdir(), everything else uses isfile().
    """
    return {
        "GESTURE_DIR":     {"path": GESTURE_DIR,     "exists": os.path.isdir(GESTURE_DIR)},
        "GESTURE_PYTHON":  {"path": GESTURE_PYTHON,  "exists": os.path.isfile(GESTURE_PYTHON)},
        "GESTURE_SCRIPT":  {"path": GESTURE_SCRIPT,  "exists": os.path.isfile(GESTURE_SCRIPT)},
        "GESTURE_RESULTS": {"path": GESTURE_RESULTS, "exists": os.path.isfile(GESTURE_RESULTS)},
    }


# Directory containing this file — used to locate the sibling HTML/CSS/JS files
_UI_DIR = os.path.dirname(os.path.abspath(__file__))


def sse_event(event_type: str, data: str) -> str:
    """
    Format one SSE event as a string.
    `data` must not contain raw newlines — SSE uses newlines as delimiters.
    Multi-line content must be split by the caller before calling this.
    """
    return f"event: {event_type}\ndata: {data}\n\n"


def sse_log(msg: str) -> str:
    """Wrap a single-line message as an SSE 'log' event."""
    return sse_event("log", msg)


def sse_result(payload: dict) -> str:
    """Serialize the final result dict and wrap it as an SSE 'result' event."""
    # json.dumps without indent so there are no embedded newlines in the JSON string
    return sse_event("result", json.dumps(payload))




# ─────────────────────────────────────────────────────────────────────────────
# Core async generator
# Each phase yields SSE events that the browser receives in real time.
# The final yield is always a 'result' event with the complete summary dict.
# ─────────────────────────────────────────────────────────────────────────────

async def run_gesture_generator():
    """
    Async generator powering the SSE endpoint.

    Phases:
      1. Path checks      — log whether each required file/dir exists
      2. Command build    — construct the exact cmd list used by app.py
      3. Subprocess launch — asyncio.create_subprocess_exec (same as app.py)
      4. Output streaming — drain stdout+stderr via a shared asyncio.Queue
      5. Return code      — log exit status
      6. results.json     — check existence, parse, and log top-level keys
      7. Final result     — emit 'result' SSE event with complete summary dict
    """

    # Accumulated state that goes into the final result event
    stdout_lines:   List[str] = []
    stderr_lines:   List[str] = []
    error_message   = ""
    return_code     = None
    results_found   = False
    results_content = None

    # ── Phase 1: Path checks ──────────────────────────────────────────────
    yield sse_log(f"[{now()}] ===== GESTURE ANALYSIS DEBUG START =====")
    yield sse_log(f"[{now()}] Checking required paths...")

    paths_checked = check_paths()
    for name, info in paths_checked.items():
        marker = "✅ EXISTS " if info["exists"] else "❌ MISSING"
        yield sse_log(f"[{now()}]   {name}: {marker}")
        yield sse_log(f"[{now()}]          → {info['path']}")
        log.info("PATH  %-20s  exists=%-5s  %s", name, info["exists"], info["path"])

    # Abort before launching the subprocess if either executable is absent —
    # that would just produce a confusing "file not found" OS error.
    if not paths_checked["GESTURE_PYTHON"]["exists"]:
        error_message = f"GESTURE_PYTHON not found: {GESTURE_PYTHON}"
        yield sse_log(f"[{now()}] ❌ FATAL: {error_message}")
        yield sse_result({
            "success": False, "return_code": None,
            "stdout": "", "stderr": "",
            "results_found": False, "results_content": None,
            "error_message": error_message,
            "paths_checked": paths_checked,
        })
        return

    if not paths_checked["GESTURE_SCRIPT"]["exists"]:
        error_message = f"GESTURE_SCRIPT not found: {GESTURE_SCRIPT}"
        yield sse_log(f"[{now()}] ❌ FATAL: {error_message}")
        yield sse_result({
            "success": False, "return_code": None,
            "stdout": "", "stderr": "",
            "results_found": False, "results_content": None,
            "error_message": error_message,
            "paths_checked": paths_checked,
        })
        return

    # ── Phase 2: Build command ────────────────────────────────────────────
    # Exact replica of _run_gesture() in app.py:
    #   cmd = [GESTURE_PYTHON, GESTURE_SCRIPT, url, *ranges, "--output", GESTURE_RESULTS]
    cmd = [GESTURE_PYTHON, GESTURE_SCRIPT, TEST_URL, *TEST_RANGES, "--output", GESTURE_RESULTS]

    # Display the command with quotes around paths that contain spaces
    cmd_display = " ".join(f'"{p}"' if (" " in p or not p) else p for p in cmd)

    yield sse_log(f"[{now()}] Command to execute:")
    yield sse_log(f"[{now()}]   {cmd_display}")
    yield sse_log(f"[{now()}] TEST_URL    : {TEST_URL}")
    yield sse_log(f"[{now()}] TEST_RANGES : {TEST_RANGES}")
    yield sse_log(f"[{now()}] Output path : {GESTURE_RESULTS}")
    log.info("CMD: %s", cmd_display)

    # ── Phase 3: Launch subprocess ────────────────────────────────────────
    yield sse_log(f"[{now()}] Launching subprocess...")
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        yield sse_log(f"[{now()}] ✅ Process launched — PID={proc.pid}")
        log.info("Subprocess launched  PID=%s", proc.pid)

    except Exception as exc:
        tb = traceback.format_exc()
        error_message = f"Failed to launch subprocess: {exc}"
        yield sse_log(f"[{now()}] ❌ Launch error: {exc}")
        for tb_line in tb.splitlines():
            yield sse_log(f"[{now()}]   {tb_line}")
        log.error("Subprocess launch failed", exc_info=True)
        yield sse_result({
            "success": False, "return_code": None,
            "stdout": "", "stderr": "",
            "results_found": False, "results_content": None,
            "error_message": f"{error_message}\n{tb}",
            "paths_checked": paths_checked,
        })
        return

    # ── Phase 4: Stream stdout + stderr via asyncio.Queue ─────────────────
    # We cannot yield from inside a spawned Task, so we use a shared queue.
    # Each stream task pushes (tag, line) tuples; None signals end-of-stream.
    queue: asyncio.Queue = asyncio.Queue()

    async def drain(stream, tag: str, buf: List[str]) -> None:
        """Read one stream line-by-line; push each decoded line to the queue."""
        try:
            async for raw in stream:
                line = raw.decode("utf-8", errors="replace").rstrip()
                buf.append(line)
                log.info("[GESTURE %s] %s", tag, line)
                await queue.put((tag, line))
        except Exception as exc:
            # Push the error as a line so it appears in the live log
            await queue.put((tag, f"[READ ERROR] {exc}"))
        finally:
            await queue.put((tag, None))  # sentinel: this stream is done

    out_task = asyncio.create_task(drain(proc.stdout, "STDOUT", stdout_lines))
    err_task = asyncio.create_task(drain(proc.stderr, "STDERR", stderr_lines))

    yield sse_log(f"[{now()}] Streaming output (timeout={TIMEOUT_SECONDS}s)...")

    loop       = asyncio.get_running_loop()
    deadline   = loop.time() + TIMEOUT_SECONDS
    out_done   = False
    err_done   = False

    try:
        while not (out_done and err_done):
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError()

            try:
                tag, line = await asyncio.wait_for(queue.get(), timeout=min(remaining, 2.0))
            except asyncio.TimeoutError:
                if loop.time() >= deadline:
                    raise
                # Send an SSE comment (": keepalive") so browsers don't close the
                # connection during long quiet periods between subprocess lines.
                yield ": keepalive\n\n"
                # If the process already exited and both drains are idle, stop waiting
                if proc.returncode is not None:
                    break
                continue

            if line is None:
                # Sentinel: this particular stream finished
                if tag == "STDOUT":
                    out_done = True
                else:
                    err_done = True
            else:
                yield sse_log(f"[{now()}] [{tag}] {line}")

    except asyncio.TimeoutError:
        yield sse_log(f"[{now()}] ❌ TIMEOUT: subprocess exceeded {TIMEOUT_SECONDS}s — killing")
        log.error("Gesture subprocess timed out after %ss", TIMEOUT_SECONDS)
        try:
            proc.kill()
        except Exception:
            pass
        await asyncio.gather(out_task, err_task, return_exceptions=True)
        yield sse_result({
            "success": False, "return_code": None,
            "stdout": "\n".join(stdout_lines),
            "stderr": "\n".join(stderr_lines),
            "results_found": False, "results_content": None,
            "error_message": f"Subprocess timed out after {TIMEOUT_SECONDS}s",
            "paths_checked": paths_checked,
        })
        return

    # ── Phase 5: Collect return code ──────────────────────────────────────
    await asyncio.gather(out_task, err_task, return_exceptions=True)
    await proc.wait()
    return_code = proc.returncode

    yield sse_log(f"[{now()}] Return code: {return_code}")
    log.info("Subprocess return code: %s", return_code)

    if return_code != 0:
        error_message = f"Subprocess exited with non-zero code: {return_code}"
        yield sse_log(f"[{now()}] ❌ {error_message}")

    # ── Phase 6: Check results.json ───────────────────────────────────────
    yield sse_log(f"[{now()}] Checking for results.json:")
    yield sse_log(f"[{now()}]   → {GESTURE_RESULTS}")
    results_found = os.path.isfile(GESTURE_RESULTS)
    yield sse_log(f"[{now()}] results.json: {'✅ FOUND' if results_found else '❌ NOT FOUND'}")
    log.info("results.json exists: %s", results_found)

    if results_found:
        try:
            with open(GESTURE_RESULTS, "r", encoding="utf-8") as fh:
                results_content = json.load(fh)
            top_keys = (
                list(results_content.keys())
                if isinstance(results_content, dict)
                else type(results_content).__name__
            )
            yield sse_log(f"[{now()}] ✅ Loaded results.json — top-level keys: {top_keys}")
            log.info("results.json top-level keys: %s", top_keys)
        except Exception as exc:
            tb = traceback.format_exc()
            yield sse_log(f"[{now()}] ❌ Failed to parse results.json: {exc}")
            for tb_line in tb.splitlines():
                yield sse_log(f"[{now()}]   {tb_line}")
            error_message = error_message or f"results.json parse error: {exc}"
            log.error("results.json load error", exc_info=True)

    # ── Phase 7: Emit final result event ──────────────────────────────────
    success = (return_code == 0) and results_found
    yield sse_log(f"[{now()}] ===== {'SUCCESS' if success else 'FAILED'} =====")
    log.info("Analysis %s", "SUCCESS" if success else "FAILED")

    yield sse_result({
        "success": success,
        "return_code": return_code,
        "stdout": "\n".join(stdout_lines),
        "stderr": "\n".join(stderr_lines),
        "results_found": results_found,
        "results_content": results_content,
        "error_message": error_message or None,
        "paths_checked": paths_checked,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def dashboard():
    """Serve the debug dashboard HTML."""
    return FileResponse(os.path.join(_UI_DIR, "apptest.html"))


@app.get("/apptest.css")
async def serve_css():
    return FileResponse(os.path.join(_UI_DIR, "apptest.css"), media_type="text/css")


@app.get("/apptest.js")
async def serve_js():
    return FileResponse(os.path.join(_UI_DIR, "apptest.js"), media_type="application/javascript")


@app.get("/run-gesture-stream")
async def run_gesture_stream():
    """
    SSE endpoint consumed by the dashboard's EventSource.
    Streams 'log' events (one per output line) then a single 'result' event.
    The 'result' event payload matches the JSON documented in the module docstring.
    """
    return StreamingResponse(
        run_gesture_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Tell nginx/proxies not to buffer the SSE stream
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=== APPTEST STARTUP PATH CHECK ===")
    for name, info in check_paths().items():
        status = "EXISTS " if info["exists"] else "MISSING"
        log.info("  %-20s  %s  %s", name, status, info["path"])
    log.info("==================================")
    log.info("Listening on http://0.0.0.0:8001")
    # reload=False so we don't accidentally re-launch subprocesses on file save
    uvicorn.run("apptest:app", host="0.0.0.0", port=8001, reload=False)

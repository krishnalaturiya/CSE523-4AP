# LaughLab — Setup Guide
### CSE 524 Advanced Project

Stand-up comedy analysis platform. Runs audio analysis (Whisper + YAMNet), gesture/body movement detection (MediaPipe), and LLM-based punchline detection (Groq) concurrently across video segments.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Project Structure](#2-project-structure)
3. [Environment 1 — Audio/API Server (Python 3.12)](#3-environment-1--audioapi-server-python-312)
4. [Environment 2 — Gesture Analysis (Python 3.10)](#4-environment-2--gesture-analysis-python-310)
5. [API Keys](#5-api-keys)
6. [Running the Server](#6-running-the-server)
7. [Using the Dashboard](#7-using-the-dashboard)
8. [Output Directories](#8-output-directories)
9. [Running the Debug Server](#9-running-the-debug-server-apptest)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

Install all of the following before setting up either Python environment.

### Python Versions (both required)

| Version | Download |
|---|---|
| Python 3.12.x | https://www.python.org/downloads/ |
| Python 3.10.11 | https://www.python.org/downloads/release/python-31011/ |

> Install both. Do **not** use 3.11 or 3.13 — MediaPipe requires exactly 3.10.x, and the audio stack is tested on 3.12.x.

### ffmpeg

Required for audio extraction and video clip cutting.

```
winget install ffmpeg
```

Or download from https://ffmpeg.org/download.html and add the `bin/` folder to your system PATH.

Verify:
```
ffmpeg -version
```

### Git

```
winget install Git.Git
```

---

## 2. Project Structure

```
AP/
├── CSE_524_AdvancedProject/        ← FastAPI server + audio pipeline
│   ├── app.py                      ← Main server (run this)
│   ├── analysis.py                 ← Single-video audio analysis engine
│   ├── analyze_segments.py         ← Multi-segment analysis + punchline detection
│   ├── punchline_detector.py       ← Groq LLM punchline detection
│   ├── dashboard.html/css/js       ← Frontend dashboard
│   ├── apptest.py                  ← Gesture debug server (port 8001)
│   ├── requirements.txt            ← Dependencies for envforaudio
│   └── envforaudio/                ← Python 3.12 virtual environment
│
└── gestureanalysis/                ← Gesture/body movement pipeline
    ├── hand_body_movement_detectionnew.py   ← MediaPipe analysis script
    ├── results.json                ← Output written here after each run
    ├── requirements.txt            ← Dependencies for mediapipe_env
    └── mediapipe_env/              ← Python 3.10.11 virtual environment
```

---

## 3. Environment 1 — Audio/API Server (Python 3.12)

This environment runs the FastAPI server, Whisper transcription, YAMNet audio classification, and all API routes.

### Create the virtual environment

Open a terminal in the project root (`AP/`):

```powershell
cd "CSE_524_AdvancedProject"
C:\Users\<you>\AppData\Local\Programs\Python\Python312\python.exe -m venv envforaudio
```

### Activate it

```powershell
envforaudio\Scripts\activate
```

You should see `(envforaudio)` in your prompt.

### Install dependencies

```powershell
pip install -r requirements.txt
```

> First install takes 10–20 minutes — TensorFlow and Whisper are large downloads.

### Verify key packages

```powershell
python -c "import fastapi, whisper, tensorflow, librosa; print('OK')"
```

---

## 4. Environment 2 — Gesture Analysis (Python 3.10)

This environment runs the MediaPipe pose and hand detection pipeline as a subprocess. It must be Python **3.10.11** — MediaPipe 0.10.x does not support 3.11+.

### Create the virtual environment

Open a terminal in the project root (`AP/`):

```powershell
cd "gestureanalysis"
C:\Users\<you>\AppData\Local\Programs\Python\Python310\python.exe -m venv mediapipe_env
```

### Activate it

```powershell
mediapipe_env\Scripts\activate
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

> PyTorch (`torch==2.11.0`) is the largest download here — ~2 GB.

### Verify key packages

```powershell
python -c "import mediapipe, cv2, numpy; print('OK')"
```

### Test the gesture script directly

```powershell
python hand_body_movement_detectionnew.py "https://www.youtube.com/watch?v=WSx2gqeEyLE" 0-30 --output test_results.json
```

---

## 5. API Keys

### Groq API Key (required for punchline detection)

Punchline detection uses the Groq API (llama-3.3-70b-versatile) to identify recurring punchlines from the transcript.

1. Sign up at https://console.groq.com
2. Create an API key
3. Set the environment variable before running the server:

```powershell
$env:GROQ_API_KEY = "gsk_your_key_here"
```

To make it permanent, add it to your system environment variables via:
`Settings → System → About → Advanced system settings → Environment Variables`

> Without this key, punchline detection silently skips and returns empty results. All other analysis still works.

---

## 6. Running the Server

**Always run via `python app.py`, never via `uvicorn app:app --reload`.**

> On Windows, `uvicorn --reload` forces `_WindowsSelectorEventLoop` which breaks `asyncio.create_subprocess_exec` (the gesture subprocess call). `app.py` sets `WindowsProactorEventLoopPolicy` at startup and runs with `reload=False`.

### Steps

```powershell
# 1. Navigate to the app folder
cd "CSE_524_AdvancedProject"

# 2. Activate the audio environment
envforaudio\Scripts\activate

# 3. Set Groq API key (if not set permanently)
$env:GROQ_API_KEY = "gsk_your_key_here"

# 4. Start the server
python app.py
```

Server starts at: **http://localhost:8000**

Dashboard opens at: **http://localhost:8000**

### Expected startup output

```
INFO  PATH CHECK
INFO    GESTURE_DIR     exists=True
INFO    GESTURE_PYTHON  exists=True
INFO    GESTURE_SCRIPT  exists=True
INFO    GESTURE_RESULTS exists=False   ← OK, created after first run
INFO  Uvicorn running on http://0.0.0.0:8000
```

---

## 7. Using the Dashboard

Open **http://localhost:8000** in your browser.

### Running an analysis

1. Select the **Video 1** or **Video 2** tab
2. Optionally paste a custom YouTube URL (leave blank for the default video)
3. Optionally edit the segment time ranges
4. Click **⚡ Analyze**

First run downloads the video and runs Whisper + YAMNet + MediaPipe — expect **10–25 minutes**. Subsequent runs use cached files and take **1–3 minutes**.

### What gets rendered after analysis

| Section | Description |
|---|---|
| Winner Banner | Segment with the strongest audience reaction |
| Punchline Analysis | Bar chart of audience reaction per punchline, per segment. Click any bar to play the clip. |
| Performance Review | Per-segment accordion cards with stats, video clip, laugh events table, and gesture metric charts |
| Laugh Intensity Heatmaps | Overlaid laugh intensity across all segments |
| Gesture Analysis | Cross-segment comparison chart (appears after all segment charts finish rendering) |

### Gesture metric selector (per-segment)

Each segment card has colored chip toggles for all 11 body metrics. Click to add/remove metrics from that segment's chart. At least one must remain selected.

### Global gesture comparison

After analysis completes, the bottom panel shows all segments plotted for one metric. Click a chip to switch the metric — all segments update simultaneously.

---

## 8. Output Directories

All output is cached to disk. Re-running analysis reuses cached files automatically.

```
C:\tmp\
├── laughlab_full\              ← Primary output dir (/api/analyze/full)
│   ├── audio_16k_mono.wav      ← Extracted audio
│   ├── video_full.mp4          ← Downloaded video
│   ├── segment_analysis.json   ← Full pipeline results cache
│   ├── punchlines_<hash>.json  ← Groq punchline cache (per URL)
│   └── clips\
│       ├── clip_Part_1.mp4     ← Segment video clips
│       └── pl_1_..._Part_1.mp4 ← Punchline sub-clips
│
├── laughlab_segments\          ← Legacy Video 1 cache
└── laughlab_video2\            ← Legacy Video 2 cache

AP\gestureanalysis\
└── results.json                ← Gesture pipeline output
```

---

## 9. Running the Debug Server (apptest)

`apptest.py` is a standalone debug tool that runs the gesture subprocess in isolation with real-time log streaming. Useful for diagnosing gesture pipeline issues without running the full analysis.

```powershell
cd "CSE_524_AdvancedProject"
envforaudio\Scripts\activate
python apptest.py
```

Open **http://localhost:8001**, then click **Run Gesture Analysis**.

Streams live stdout/stderr from the gesture subprocess and shows a full JSON result summary.

---

## 10. Troubleshooting

### Gesture subprocess fails silently / `Failed to launch subprocess:`

**Cause:** Running `uvicorn app:app --reload` instead of `python app.py`.  
**Fix:** Always use `python app.py`.

### `GESTURE_PYTHON not found`

**Cause:** `mediapipe_env` does not exist or was created in the wrong location.  
**Fix:** Recreate the venv inside `gestureanalysis/` using Python 3.10.11 exactly.

```powershell
cd gestureanalysis
C:\Users\<you>\AppData\Local\Programs\Python\Python310\python.exe -m venv mediapipe_env
mediapipe_env\Scripts\activate
pip install -r requirements.txt
```

### `ffmpeg: command not found`

**Fix:** Install ffmpeg and ensure it is on your system PATH. Restart the terminal after installing.

### Punchline detection returns empty results

**Cause:** `GROQ_API_KEY` is not set.  
**Fix:** `$env:GROQ_API_KEY = "gsk_your_key_here"` before running the server.

### YAMNet download fails on first run

YAMNet is downloaded from TensorFlow Hub on the first run. Requires internet access. If it fails, delete `C:\tmp\laughlab_full\` and retry.

### Video clip is blank in the punchline modal

**Cause:** Clips were generated in an older cache directory before the storage refactor.  
**Fix:** Re-run analysis — clips will be written to `C:\tmp\laughlab_full\clips\` and served correctly.

### Analysis takes too long / times out

The gesture subprocess has a 20-minute timeout. On first run with a long video, MediaPipe processes every frame — this is expected. Subsequent runs skip the download step and are much faster.

---

## Quick Reference

| Task | Command |
|---|---|
| Start main server | `cd CSE_524_AdvancedProject && envforaudio\Scripts\activate && python app.py` |
| Start debug server | `cd CSE_524_AdvancedProject && envforaudio\Scripts\activate && python apptest.py` |
| Run gesture script standalone | `cd gestureanalysis && mediapipe_env\Scripts\activate && python hand_body_movement_detectionnew.py "<url>" 0-60` |
| Check gesture output | `cat gestureanalysis\results.json` |
| Clear cache for fresh run | `Remove-Item C:\tmp\laughlab_full -Recurse -Force` |

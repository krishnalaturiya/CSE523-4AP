# LaughLab 🎤 — Stand-Up Comedy Analyzer

## Project Structure

```
laughlab/
├── dashboard.html   ← Frontend only (HTML/CSS/JS). Never touches video.
├── analysis.py      ← All Python analysis logic. This is where YOU work.
├── app.py           ← FastAPI server connecting frontend ↔ analysis.py
└── README.md
```

## Quick Start

### 1. Install dependencies
```bash
pip install fastapi uvicorn python-multipart
```

### 2. Run the server
```bash
.\venv\Scripts\Activate.ps1
uvicorn app:app --reload --port 8000
```

### 3. Open the dashboard
Visit http://localhost:8000 in your browser.

---

## Where to Add Your Analysis Logic

Everything is in `analysis.py`. Each function has a clear TODO comment:

| Function | What it does | Replace with |
|---|---|---|
| `download_video()` | Downloads from URL | `yt-dlp` |
| `extract_audio()` | Pulls audio from video | `ffmpeg` |
| `detect_laughter()` | Core laugh detection | `librosa`, `pyAudioAnalysis`, custom ML |
| `find_laugh_peaks()` | Finds peak moments | Already works once `detect_laughter` is real |
| `classify_crowd_emotions()` | Emotion categories | Audio classifier or Whisper |
| `generate_key_moments()` | Timestamps + descriptions | Whisper transcript + NLP |

## Recommended Python Libraries

```bash
pip install librosa          # Audio analysis
pip install openai-whisper   # Speech-to-text for transcript analysis  
pip install yt-dlp           # Download YouTube/Vimeo videos
pip install numpy scipy      # Numerical processing
# ffmpeg must be installed system-wide: brew install ffmpeg / apt install ffmpeg
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves dashboard.html |
| POST | `/api/analyze/url` | Analyze video from URL |
| POST | `/api/analyze/upload` | Analyze uploaded video file |
| GET | `/api/health` | Health check |

Interactive API docs available at: http://localhost:8000/docs

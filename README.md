# Real-Time Voice Over Translator

A multi-tool AI audio/voice web app: translate speech, generate speech from text, clean up recordings, and enhance voice quality — all from a single dashboard.

## Pages / Frontend Files

| File | Route (expected) | Purpose |
|---|---|---|
| `index.html` | `/` | Login/signup landing page + dashboard linking to all five tools |
| `voice_to_voice.html` | `/voice_to_voice` | Translate spoken audio into another language, output as AI-generated speech |
| `voice_to_text.html` | `/voice_to_text` | Transcribe and translate speech into text |
| `text_to_voice.html` | `/text_to_voice` | Convert typed text (or uploaded TXT/DOCX/PDF) into natural-sounding speech |
| `noise_removal.html` | `/noise_removal` | Separate vocals from background noise/music using Demucs |
| `voice_enhancer.html` | `/voice_enhancer` | Adjust speed, clarity, volume, and noise reduction on audio |

Every frontend is a self-contained HTML/CSS/JS file (Poppins font, glassmorphism gradient theme) that posts audio/text to a backend endpoint of the same name and renders the returned result inline.

## Features by Tool

### 🔁 Voice → Voice
- Upload or record audio
- Select spoken language and target language (English, Hindi, Tamil, Telugu, Malayalam, Japanese, French, Spanish)
- Choose a voice mode (Female/Male Neural, Child)
- Returns translated audio in the chosen voice

### 📝 Voice → Text
- Upload or record audio
- Select spoken language and desired output language
- Returns a transcript, translated as needed

### 🔊 Text → Voice
- Type text directly, or upload a TXT/DOCX/PDF (content is auto-extracted into the text box)
- Select output language and voice mode (Female/Male Neural, Child)
- Returns synthesized speech audio

### 🔇 Noise Removal
- Upload or record audio
- Backend performs vocal/background separation (Demucs-based)
- Returns two playable/downloadable tracks: **Vocals** and **Background/Noise**
- Frontend shows a simulated progress bar while processing

### 🎛️ Voice Enhancer
- Upload or record audio
- Auto-detects spoken language (`/detect_language` endpoint) with a manual override dropdown
- Sliders for:
  - Noise reduction (0–1)
  - Clarity (0–1)
  - Speed (0.5×–2.0×)
  - Volume (-10 to +10 dB)
- Returns the processed/enhanced audio

## Authentication

`index.html` includes a simple login/signup flow that stores credentials in **browser `localStorage`** (no backend auth call). This is a front-end-only placeholder:
- Sign up stores `username`/`password` in `localStorage`
- Login checks the entered credentials against those stored values
- `loggedIn` flag in `localStorage` controls whether the dashboard or the auth screen is shown on load

⚠️ This is not secure for production — credentials are stored in plaintext client-side and there's no real session/auth on the backend. Swap this out for a real auth system (hashed passwords, server-side sessions/JWT) before deploying.

## Expected Backend Endpoints

The frontend expects a server exposing (at minimum):

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/voice_to_voice` | Accepts audio + form fields, returns translated audio |
| `POST` | `/voice_to_text` | Accepts audio + form fields, returns transcript/translation |
| `POST` | `/text_to_voice` | Accepts text + form fields, returns synthesized audio |
| `POST` | `/noise_removal` | Accepts audio, returns `{ vocals_url, background_url }` (or `{ error }`) |
| `POST` | `/detect_language` | Accepts audio, returns detected language |
| `POST` | `/voice_enhancer` | Accepts audio + enhancement params, returns processed audio |

None of these backend handlers were included in this upload — only the five tool pages and the landing page. You'll need a server (e.g., Flask/FastAPI) implementing the above routes, likely wrapping models/libraries such as Demucs (source separation), a speech-to-text engine (e.g., Whisper), a TTS engine, and a translation service.

## Requirements

Frontend has no build step or dependencies beyond a modern browser (uses `MediaRecorder`, `fetch`, `FormData`, `localStorage`).

Backend (not included) would typically need something like:
```
flask (or fastapi + uvicorn)
demucs
openai-whisper (or similar STT)
a TTS library/service
a translation library/service
```

## Setup & Running

1. Serve the five tool pages and `index.html` from a backend that also implements the endpoints above (e.g., Flask routes returning `render_template` for each page and JSON for the POST endpoints).
2. Open `index.html` (served at `/`) in a browser.
3. Sign up, log in, and navigate to any of the five tools from the dashboard.

## Suggested Improvements

- **Auth is client-side only.** Move credential storage and session verification to the backend; never store passwords in `localStorage`.
- **Simulated progress bars** (e.g., in `noise_removal.html`) use fixed `setTimeout` steps rather than real backend progress — fine for a demo, but consider streaming actual progress (e.g., via polling or WebSockets) for longer jobs.
- **Consistent language lists**: the language dropdown (English, Hindi, Tamil, Telugu, Malayalam, Japanese, French, Spanish) is duplicated across four files — extracting it into a shared JS snippet/template would reduce drift if you add more languages later.
- **Error handling** in each tool generally checks `data.error` from the JSON response — make sure the backend consistently returns that shape on failure across all endpoints.

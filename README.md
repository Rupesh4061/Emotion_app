# Text → Emotion Analyzer

A simple Streamlit app that predicts emotion from text using Hugging Face transformers.

## Features
- Predicts emotions (joy, sadness, anger, fear, surprise, etc.)
- Shows top-3 predictions + confidence and emojis
- Bar chart of probabilities
- Saves predictions to a CSV and allows download
- Supports English (fast model) and a multilingual model for Hindi

## Files
- `app_text_emotion.py` — main Streamlit app
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — optional Streamlit settings
- `.gitignore` — recommended ignores

## How to run locally (Windows)
1. Open terminal in the project folder.
2. Create and activate venv:
   ```bash
   python -m venv venv
   venv\Scripts\activate

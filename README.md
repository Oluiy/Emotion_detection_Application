## Emotion Detection Web App 🎭

A small Flask-based web application that detects emotions from images using a pretrained model.

This README explains how to set up, run, and deploy the app that is included in this repository.

## Repository at a glance

Project structure:

```
EMOTION_DETECTOR/
├── app.py                        
├── model.py                     
├── simple_emotion_detector.py  
├── emotion_model.joblib        
├── templates/
│   └── index.html               
├── requirements.txt            
├── Procfile                   
├── runtime.txt                 
├── app.json                  
├── link.txt                    
└── README.md          
```

Key files and folders:

- `app.py` — Flask application (entry point). The app runs by default on port 8000.
- `model.py` — wrapper to initialize and use the trained model.
- `simple_emotion_detector.py` — model helper / implementation used by `model.py`.
- `emotion_model.joblib` — pretrained model (joblib format) used by the app.
- `templates/` — HTML templates (contains `index.html`).
- `requirements.txt` — Python dependencies.
- `Procfile` — Heroku-compatible process file (`web: gunicorn app:app`).

## Requirements

- Python 3.8+ (recommended)
- Install project dependencies:

```bash
pip install -r requirements.txt
```

If you use a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run locally

The application is a Flask app and can be started directly for development or served with Gunicorn for production-like behavior.

- Development (quick run):

```bash
python app.py
```

By default the app listens on port 8000. Open http://localhost:8000 in your browser.

- Production / Heroku-like (uses `Procfile`):

```bash
# start with gunicorn (Procfile uses: web: gunicorn app:app)
gunicorn app:app
```

If deploying to Heroku, the included `Procfile` already contains the startup command.

## How to use

1. Open the app in your browser (`/` route).
2. Fill name (required) and optionally email.
3. Upload an image file (jpg, jpeg, png, bmp, gif).
4. Click the analyze button — the app will return the predicted emotion and confidence, plus a breakdown of emotion percentages.

Notes:
- `app.py` validates uploads (allowed extensions and a 16 MB max file size configured).
- If the trained model isn't available or initialization fails, the app falls back to a random prediction for demo purposes.

## Model

The pretrained model is provided as `emotion_model.joblib` in the repository root. `model.py` contains the code that loads this model (via joblib) and exposes a `predict_emotion` helper used by `app.py`.

If you retrain or replace the model, keep the same file path or update `model.py` accordingly.

## Database

The app creates a lightweight SQLite database file named `emotion_detection.db` in the project root. It stores simple user records and prediction results. The DB is created automatically on first run.

## Deployment

- Heroku: The `Procfile` includes `web: gunicorn app:app` so deployment to Heroku is straightforward. Make sure to set any required environment variables (e.g. `SECRET_KEY`).

- Docker: You can containerize the app yourself (not included here). Use `gunicorn app:app` as the container entrypoint.

## Troubleshooting

- If the app prints an import/initialization error for the detector, ensure `emotion_model.joblib` exists and `requirements.txt` packages are installed.
- File upload errors: ensure the image is of an allowed type and under 16MB.
- Port in use: change the `PORT` environment variable or edit the port in `app.py` for local testing.

## Contributing

Small improvements and fixes are welcome. If you change model behavior or public APIs, please update this README accordingly.

## License

Educational / demo project. No explicit license set in this repository — update if you want to apply one.

---

If you'd like, I can add a short example `curl` command and a minimal `.env.example` next.
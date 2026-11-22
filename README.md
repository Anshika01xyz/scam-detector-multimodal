# Multimodal Scam Detector

Detects scams from pasted text, screenshots (via OCR) and URLs, and
combines the results into one risk score.

## Team
- Person 1: NLP, screenshot analysis + OCR, frontend
- Person 2: URL detection, risk scoring, backend

## Modules
- `nlp/`: text scam classifier
- `ocr/`: screenshot preprocessing and text extraction
- `url_model/`: phishing URL classifier
- `backend/`: APIs and risk engine
- `frontend/`: Streamlit app

## Setup
```
pip install -r requirements.txt
```

## Data and models
Large datasets and trained models are not stored in Git.

**URL dataset (urldata.csv):** [Download from Google Drive](https://drive.google.com/file/d/1wbHGoB-TF2iio9jozxd7eJtDwJmiGj-P/view?usp=sharing)

Steps:
1. Download `urldata.csv` from the link above.
2. Create a folder named `data` in the project root and put the file inside it, so the path is `data/urldata.csv`.
3. Train the URL model (this creates `url_model/url_model.joblib`):
```
python url_model/train.py
```

## Run the backend
```
cd backend
uvicorn main:app --reload
```
Then open http://127.0.0.1:8000/docs to test the API.

Endpoints:
- `POST /predict-url` with `{"url": "..."}` returns label, score and reasons.
- `POST /analyze` with `{"url": "...", "text_score": 0-1, "screenshot_score": 0-1}` returns `risk_score` and `verdict`. Any field can be left out.

## Docs
See `docs/` for the interface spec, decisions and workflow.
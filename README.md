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
Download links: _(add Drive link here)_

## Docs
See `docs/` for the interface spec, decisions and workflow.

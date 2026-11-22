# Running the demo on your own computer (text model + OCR + frontend)

## 1. Get the code
On GitHub choose the branch `person1-nlp-ocr`, then **Code > Download ZIP** and unzip it.
(Or `git clone -b person1-nlp-ocr <repo url>` if you use git.)

## 2. Install Python and the packages
- Python 3.10 - 3.12 from python.org (on Windows tick **Add Python to PATH**).
- Open a terminal **inside the project folder**, then:
```
python -m venv venv
venv\Scripts\activate          (Windows)      source venv/bin/activate   (Mac/Linux)
pip install -r requirements-person1.txt
```
`torch` is large (about 1-2 GB). That is normal.

## 3. Install Tesseract (reads text from screenshots)
- **Windows:** download the installer from the "UB Mannheim" Tesseract page and install it with the default
  folder (`C:\Program Files\Tesseract-OCR`). The code finds it there, no PATH change needed.
- **Mac:** `brew install tesseract`
- **Linux:** `sudo apt install tesseract-ocr`

## 4. Put the trained models in `models/`
Download them from the shared Google Drive folder (`scam_detector_models`) so you have:
```
models/tfidf_lr.pkl
models/distilbert_scam/        (a folder with config.json, model.safetensors, tokenizer files)
```
Models are not stored in GitHub because they are too large.

## 5. Start the app
```
streamlit run frontend/app.py
```
A browser tab opens. The first analysis takes a few seconds while DistilBERT loads.

## Troubleshooting
| Problem | Fix |
|---|---|
| "No trained model found" | The `models/` folder is empty or in the wrong place (it must be next to `nlp/` and `frontend/`). |
| Error about `InconsistentVersionWarning` or the .pkl will not load | scikit-learn version differs from Colab. In Colab run `!pip show scikit-learn`, then `pip install scikit-learn==<that version>` on your computer. Or just retrain with `python nlp/train_tfidf.py`. |
| "Tesseract not found" | Install it (step 3). Without it screenshots do not work; pasted text still works. |
| Screenshot analysis is slow | About 3-5 seconds per image on a normal laptop CPU is expected. |
| Very slow / out of memory | Pick **TF-IDF** in the sidebar, it needs far less memory. |

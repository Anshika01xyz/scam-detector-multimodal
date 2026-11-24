# Person 1 guide: text model, OCR, frontend, deployment

## What is in the project
| Folder | Contents |
|---|---|
| `nlp/` | text cleaning, training (TF-IDF + DistilBERT), `predict.py` (main function), scam type, reasons, evaluation scripts |
| `ocr/` | image clean-up, text extraction (tesseract / EasyOCR), screenshot generator, OCR evaluation |
| `frontend/app.py` | Streamlit web app |
| `data/raw/` | SMS Spam Collection + team-written Indian-style examples (`my_scams*.csv`) |
| `data/test/hard_test.csv` | 160-message hard test set (never train on it) |
| `deploy/` | builds the Hugging Face Space bundle |
| `notebooks/run_all.ipynb` | one notebook that trains, tests, saves and builds the bundle |

## The 3 things to do
1. **Run** `notebooks/run_all.ipynb` in Google Colab (GPU). Upload `scam-detector-person1-complete.zip` when asked.
2. **Deploy:** unzip `space_bundle.zip` (made by step 6 of the notebook), create a Hugging Face Space
   (New Space > SDK: **Streamlit** > Public), then **Files > Add file > Upload files** and drag in everything from the
   unzipped bundle. The build takes 5-15 minutes. You get a public link.
3. **Submit:** upload the project files to GitHub (branch `person1-nlp-ocr`) and add the Space link to the README.

## Results to put in the report (from the notebook output)
- Main test set (SMS + Indian-style): precision / recall / F1 for TF-IDF and DistilBERT.
- Hard test set (160 messages): accuracy with 95% confidence interval, per-category table, mistakes.
- Link-shortcut probe: does adding a link change the score?
- OCR: text recovered, links read correctly, by engine and preprocessing mode (EasyOCR vs tesseract).
- Full pipeline: screenshot -> OCR -> model -> verdict accuracy.

## Honest limitations (say these in the report)
- Indian-style training and test messages are mostly **synthetic**, written by the team. Real-world accuracy will be lower.
- The hard test set was used for error analysis, so it is no longer a clean test. Build a final test set of real
  messages (`text,label,category`) and run `python nlp/evaluate_hard.py --csv <file>`.
- The TF-IDF model flags some real bank alerts; DistilBERT handles them better but misses some subtle scams.
- OCR was tested on generated screenshots, which are cleaner than real photos.
- Loan-app scams fall under "Other / unknown" (7 agreed scam types).

## Integration with Person 2 (still to do)
`frontend/app.py` has a marked INTEGRATION POINT. Agree on the score scale (0 to 1, `legit`/`suspicious`/`scam`) and send
`urls_found` to their URL model. `text_score` can be `null` when OCR reads nothing.

## Credits
UCI SMS Spam Collection (Almeida & Hidalgo, CC BY 4.0), DistilBERT (Hugging Face), Tesseract OCR.

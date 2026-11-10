# Decisions

1. **Model task:** binary text classification (scam / legit). Output: scam
   probability + verdict (legit < 0.40 / suspicious 0.40 to 0.70 / scam > 0.70)
   + rule-based scam type + reasons.
2. **Screenshots:** converted to text by OCR and classified by the same text
   model. URLs from either input go to the URL module. The risk engine
   produces the final verdict.
3. **Stack:** Python, scikit-learn (TF-IDF baseline), DistilBERT (comparison),
   EasyOCR + OpenCV, regex URL extraction, rule-based scam typing,
   Streamlit frontend.
4. **Scam types (rule-based):** Phishing / account verification, Bank / KYC
   fraud, UPI / payment scam, Fake lottery / prize, Fake job / task offer,
   Impersonation / urgency, Other / unknown.

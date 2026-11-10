# Interface Specification

> Changes to this file need agreement from both team members.

## Rules
- Scores are always floats from 0 to 1.
- Verdicts are always the lowercase strings `legit`, `suspicious` or `scam`
  (legit < 0.40, suspicious 0.40 to 0.70, scam > 0.70).
- Missing values are `null`. Never omit keys.
- Both members use these field names exactly.

## 1. Text/screenshot module output (Person 1 -> Person 2)

```json
{
  "input_type": "text",
  "extracted_text": "Your KYC has expired. Update now at http://sbi-kyc.xyz",
  "text_score": 0.91,
  "text_verdict": "scam",
  "scam_type": "Bank / KYC fraud",
  "reasons": ["urgency language", "KYC keyword", "contains link"],
  "urls_found": ["http://sbi-kyc.xyz"],
  "ocr_confidence": null
}
```

- `input_type`: `"text"` or `"screenshot"`
- `extracted_text`: pasted text, or OCR output for a screenshot
- `ocr_confidence`: float for screenshots, `null` for pasted text
- `urls_found`: list of URLs, may be empty

## 2. URL module output (Person 2, one object per URL)

```json
{
  "url": "http://sbi-kyc.xyz",
  "url_score": 0.88,
  "url_verdict": "scam",
  "reasons": ["suspicious TLD", "brand name in domain"]
}
```

If `urls_found` is empty, the URL step is skipped and the final score uses
the text score alone.

## 3. Final output (risk engine -> frontend)

```json
{
  "final_score": 0.90,
  "final_verdict": "scam",
  "scam_type": "Bank / KYC fraud",
  "reasons": ["urgency language", "KYC keyword", "suspicious TLD"],
  "details": { "text": {}, "urls": [] }
}
```

`details` contains the full outputs of messages 1 and 2 for the breakdown view.

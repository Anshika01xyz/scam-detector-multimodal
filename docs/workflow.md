# Workflow

```mermaid
flowchart TD
    A[Pasted text] --> C[Text cleaning]
    B[Screenshot upload] --> P[Preprocess image]
    P --> O[OCR extraction]
    O --> C
    O -->|URLs found| U[URL detection - Person 2]
    C --> M[Text classifier: TF-IDF, then DistilBERT]
    M --> J[Result JSON]
    J --> R[Risk engine - Person 2]
    U --> R
    R --> F[Final verdict]
```

## Text path
Pasted text is cleaned (lowercase, strip noise), classified, and returned
as a result JSON with score, verdict, scam type and reasons.

## Screenshot path
The image is preprocessed (resize, denoise) and run through OCR. The
extracted text joins the text path at the cleaning step. Any URLs found are
passed to Person 2's URL module.

## Handoffs to Person 2
- URLs found in either input go to URL detection.
- The result JSON goes to the risk engine, which combines it with the URL
  score into the final verdict.

# URL Model Results

## Setup
- Dataset: data/clean_urls.csv (cleaned version of urldata.csv)
- Sample: 100,000 URLs, stratified 60/40 split
- Training set: 60,000 URLs
- Test set: 40,000 URLs

## Best model
XGBoost (saved as url_model/url_model.joblib)

## Metrics (test set, 40,000 URLs)
| Metric | Value |
|---|---|
| Accuracy | 0.9939 (calculated from the confusion matrix) |
| Precision | 0.9947 |
| Recall | 0.9790 |
| F1-score | 0.9868 |
| False positive rate | 0.0016 |

## Confusion matrix
|  | Predicted safe | Predicted phishing |
|---|---|---|
| Actual safe | 30569 (TN) | 49 (FP) |
| Actual phishing | 197 (FN) | 9185 (TP) |

## Earlier run (20% test split, 20,000 URLs)
Precision 0.9944, Recall 0.9776, F1 0.9859, FPR 0.0017, Accuracy about 0.993.

## Notes
- 49 safe URLs were wrongly flagged (false alarms).
- 197 phishing URLs were missed.
- Test set was increased from 20% to 40% and results stayed almost the same, so the model looks stable.
- Tested on one dataset only, so real-world results may be lower.
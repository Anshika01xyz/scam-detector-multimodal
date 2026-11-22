# URL Model Results

## Best model
XGBoost (saved as url_model/url_model.joblib)

## Metrics (test set, 20,000 URLs)
| Metric | Value |
|---|---|
| Precision | 0.9944 |
| Recall | 0.9776 |
| F1-score | 0.9859 |
| False positive rate | 0.0017 |
| Accuracy | about 0.993 |

## Confusion matrix
|  | Predicted safe | Predicted phishing |
|---|---|---|
| Actual safe | 15283 (TN) | 26 (FP) |
| Actual phishing | 105 (FN) | 4586 (TP) |

## Notes
- 26 safe URLs were wrongly flagged (false alarms).
- 105 phishing URLs were missed.
- Tested on one dataset only, so real-world results may be lower.
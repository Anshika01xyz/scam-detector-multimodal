# URL Model Results

## Setup
- Dataset: data/clean_urls.csv (cleaned version of urldata.csv)
- Sample: 100,000 URLs, stratified 60/40 split
- Training set: 60,000 URLs
- Test set: 40,000 URLs
- Features: 18 URL features (features.py)

## Model comparison (test set, 40,000 URLs)
| Model | Accuracy | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|
| Logistic Regression | 0.9874 | 0.9972 | 0.9491 | 0.9725 | 0.0008 |
| Decision Tree | 0.9903 | 0.9972 | 0.9612 | 0.9789 | 0.0008 |
| KNN | 0.9908 | 0.9906 | 0.9699 | 0.9802 | 0.0028 |
| Random Forest | 0.9935 | 0.9957 | 0.9764 | 0.9860 | 0.0013 |
| **XGBoost** | **0.9939** | 0.9947 | **0.9790** | **0.9868** | 0.0016 |

## Best model
XGBoost (highest F1 = 0.9868 and highest recall = 0.9790), saved as url_model/url_model.joblib.

## Notes
- Best model is chosen by F1-score.
- Logistic Regression and Decision Tree have the highest precision and lowest false positive rate, but they miss more phishing URLs (lower recall).
- KNN has the highest false positive rate (0.0028).
- Missing a phishing URL is the worse error for scam detection, so recall is preferred.
- Tested on one dataset only, so real-world results may be lower.
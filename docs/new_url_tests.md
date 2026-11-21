# Tests on New URLs

These URLs were not in the training data. They were tested with the trained XGBoost model through the `/predict-url` API.

| URL | Expected | Model said | Score | Correct? |
|---|---|---|---|---|
| https://www.wikipedia.org | legitimate | legitimate | 0.002 | Yes |
| https://www.amazon.in | legitimate | legitimate | 0.001 | Yes |
| https://github.com/Anshika01xyz | legitimate | phishing | 0.999 | **No** |
| https://www.sbi.co.in | legitimate | legitimate | 0.002 | Yes |
| https://www.hdfcbank.com | legitimate | legitimate | 0.047 | Yes |
| http://192.168.4.5/bank/login.php | phishing | phishing | 1.000 | Yes |
| http://sbi-account-verify.top/login | phishing | phishing | 1.000 | Yes |
| http://bit.ly/free-bonus-claim | phishing | phishing | 1.000 | Yes |
| http://paytm-wallet-update.xyz/confirm | phishing | phishing | 1.000 | Yes |
| http://secure-amazon.com.login-verify.tk/signin | phishing | phishing | 1.000 | Yes |

**Result: 9 of 10 correct (90%).**

## Observations
- All 5 phishing URLs were caught with a score near 1.0, and each came with clear reasons (suspicious ending, no HTTPS, keywords, fake brand name).
- 4 of 5 legitimate URLs were correctly marked safe. HDFC scored 0.047, which is still safe, but the rule-based reasons wrongly said "Looks like a fake hdfc website".
- **One mistake:** `https://github.com/Anshika01xyz` was flagged as phishing (score 0.999), a false positive. The reasons list was empty, so the model made this decision from the features alone. A likely cause is the URL's digits and mixed-case username path, which look unusual compared to the training data, and the training set probably had few GitHub-style URLs.
- Only 10 URLs were tested, so this is a small sample, and it shows the model can fail on URLs that look different from its training data.

## Possible improvements
- Add more legitimate URLs with user-profile paths (GitHub, LinkedIn, etc.) to the training data.
- Add a whitelist of well-known domains in the backend.
- Fix the brand check in `get_reasons` so that an official domain such as hdfcbank.com is not called fake.
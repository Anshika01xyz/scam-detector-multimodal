import math
import re
from urllib.parse import urlparse

import tldextract

# offline mode: internet ki zaroorat nahi
_extract = tldextract.TLDExtract(suffix_list_urls=())

KEYWORDS = ["login", "verify", "secure", "account", "update", "bank",
            "signin", "confirm", "password", "wallet", "paypal", "free", "bonus"]
SUSPICIOUS_TLDS = {"xyz", "top", "tk", "ml", "ga", "cf", "gq", "click",
                   "work", "support", "zip", "loan", "icu", "buzz"}
SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
              "is.gd", "buff.ly", "cutt.ly", "rb.gy"}
BRANDS = ["paypal", "google", "amazon", "apple", "microsoft", "facebook",
          "netflix", "sbi", "hdfc", "icici", "paytm", "instagram"]
IP_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")

# Well-known official domains (registered domain = name + suffix)
TRUSTED_DOMAINS = {
    "google.com", "wikipedia.org", "amazon.in", "amazon.com", "github.com",
    "sbi.co.in", "hdfcbank.com", "icicibank.com", "paytm.com", "paypal.com",
    "microsoft.com", "apple.com", "facebook.com", "instagram.com",
    "netflix.com", "linkedin.com",
}


def _parse(url):
    u = str(url).strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", u):
        u = "http://" + u
    try:
        p = urlparse(u)
        host = (p.hostname or "").lower()
    except ValueError:
        # kharab URL: brackets hata ke dobara try karo
        u = u.replace("[", "").replace("]", "")
        p = urlparse(u)
        host = (p.hostname or "").lower()
    ext = _extract(host)
    return u, p, host, ext


def _entropy(s):
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def is_trusted(url):
    """True if the URL's registered domain is in the trusted list."""
    u, p, host, ext = _parse(url)
    return f"{ext.domain}.{ext.suffix}" in TRUSTED_DOMAINS


def extract_features(url):
    u, p, host, ext = _parse(url)
    low = u.lower()
    sub = ext.subdomain
    return {
        "url_length": len(u),
        "host_length": len(host),
        "path_length": len(p.path),
        "num_dots": u.count("."),
        "num_hyphens": host.count("-"),
        "num_digits": sum(c.isdigit() for c in u),
        "digit_ratio": sum(c.isdigit() for c in u) / max(len(u), 1),
        "num_special": sum(not c.isalnum() for c in u),
        "num_subdomains": len(sub.split(".")) if sub else 0,
        "num_params": p.query.count("=") if p.query else 0,
        "has_at": int("@" in u),
        "has_ip": int(bool(IP_RE.match(host))),
        "is_https": int(p.scheme == "https"),
        "keyword_count": sum(k in low for k in KEYWORDS),
        "suspicious_tld": int(ext.suffix.split(".")[-1] in SUSPICIOUS_TLDS),
        "is_shortener": int(host in SHORTENERS),
        "has_port": int(bool(re.search(r":\d+(/|$)", p.netloc))),
        "entropy": _entropy(host),
    }


def get_reasons(url):
    """Insaan ko samajhne wali reasons ki list (explainability)."""
    if is_trusted(url):
        return []

    u, p, host, ext = _parse(url)
    f = extract_features(url)
    reasons = []

    if f["has_ip"]:
        reasons.append("Uses an IP address instead of a domain name")
    if f["has_at"]:
        reasons.append("Contains '@' which can hide the real destination")
    if f["suspicious_tld"]:
        reasons.append(f"Suspicious domain ending (.{ext.suffix})")
    if f["is_shortener"]:
        reasons.append("Shortened link hides the real destination")
    if not f["is_https"]:
        reasons.append("Does not use HTTPS")
    found = [k for k in KEYWORDS if k in u.lower()]
    if found:
        reasons.append("Suspicious keywords: " + ", ".join(found[:4]))
    if f["num_subdomains"] >= 3:
        reasons.append("Too many subdomains")
    if f["url_length"] > 75:
        reasons.append("Unusually long URL")
    if f["num_hyphens"] >= 2:
        reasons.append("Many hyphens in the domain name")
    for b in BRANDS:
        if b in host and ext.domain != b:
            reasons.append(f"Looks like a fake '{b}' website")
            break
    return reasons


if __name__ == "__main__":
    tests = [
        "https://www.google.com",
        "http://paypal-secure-login.xyz/verify/account",
        "http://192.168.4.5/bank/login.php",
        "https://github.com/Anshika01xyz",
        "https://www.hdfcbank.com",
    ]
    for t in tests:
        print(t)
        print("  trusted:", is_trusted(t))
        print("  reasons:", get_reasons(t))
        print("  features:", extract_features(t))
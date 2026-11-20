"""Text cleaning shared by training and prediction.

Keeps URLs (they are needed for the URL module) but normalises everything else.
"""
import re

URL_RE = re.compile(r"(https?://[^\s]+|www\.[^\s]+)", re.IGNORECASE)


def extract_urls(text: str) -> list:
    """Return all URLs found in the text (trailing punctuation removed)."""
    if not isinstance(text, str):
        return []
    return [u.rstrip(".,;:!?)\"'") for u in URL_RE.findall(text)]


def clean_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip odd characters. URLs are kept."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s:/.\-@?=&%#₹$]", " ", text)  # keep URL-safe chars and currency
    text = re.sub(r"\s+", " ", text).strip()
    return text

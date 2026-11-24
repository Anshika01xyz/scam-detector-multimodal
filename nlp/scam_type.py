"""Rule-based scam type (the 7 types agreed in docs/decisions.md).

Each type has keyword patterns with weights. The type with the highest total wins;
ties are broken by the order of TYPE_ORDER. Returns "Other / unknown" if nothing matches well.
"""
import re

BANK = "Bank / KYC fraud"
UPI = "UPI / payment scam"
LOTTERY = "Fake lottery / prize"
JOB = "Fake job / task offer"
IMPERSON = "Impersonation / urgency"
PHISH = "Phishing / account verification"
OTHER = "Other / unknown"

TYPE_ORDER = [BANK, UPI, LOTTERY, JOB, IMPERSON, PHISH]

RULES = {
    BANK: [
        (r"\bkyc\b", 3), (r"\bpan\b", 3), (r"aadhaar|aadhar", 3), (r"block (?:your )?accounts?", 3),
        (r"account (?:will be |has been |is |been )?(?:blocked|suspended|frozen|deactivated|on hold|put on hold|closed)", 3),
        (r"account .{0,20}(?:band ho|block ho)", 3),
        (r"net ?banking|internet banking", 2), (r"credit limit|credit card|debit card", 1),
        (r"\b(?:sbi|hdfc|icici|axis|kotak|pnb|yono|union bank|bank of baroda|canara)\b", 1),
        (r"\bbank\b", 1),
    ],
    UPI: [
        (r"\bupi\b", 3), (r"phone ?pe|google pay|gpay|paytm|\bbhim\b", 3), (r"cashback", 2), (r"by mistake", 3), (r"debited", 1),
        (r"refund", 2), (r"collect request", 3), (r"\bqr\b", 2), (r"\bpin\b", 2),
        (r"scratch card", 2), (r"payment", 1),
    ],
    LOTTERY: [
        (r"lottery|lucky draw|\bkbc\b", 3), (r"\bwon\b|\bwinner\b|jeet", 2), (r"prize", 2),
        (r"congratulations|congrats|badhai", 1),
        (r"selected for|free (?:iphone|laptop|gift|tv|recharge|data)|free \d+ ?gb", 2), (r"reward", 1), (r"claim", 1),
        (r"redeem", 2), (r"\bfree\b|\bgift\b|recharge", 1), (r"\bwin\b|spin", 2), (r"lucky", 2),
    ],
    JOB: [
        (r"work from home|money from home|part[- ]time|ghar baithe", 3), (r"data entry", 3),
        (r"\bearn\b|kamaye", 2), (r"\bhiring\b|\bjob\b|\bresume\b|shortlisted", 2), (r"\btasks?\b", 2),
        (r"registration fee|security deposit|joining fee", 3), (r"telegram|t\.me", 2),
        (r"\binvest|guaranteed returns|\breturns\b|\bstocks?\b|trading", 2),
    ],
    IMPERSON: [
        (r"electricity|power bill|disconnect|bijli", 3),
        (r"courier|parcel|package|shipment|customs|delivery (?:failed|attempt)|redeliver|"
         r"\bdhl\b|fedex|blue ?dart|india post|delhivery", 3),
        (r"\bcbi\b|police|\bcourt\b|income tax", 3), (r"arrest|case (?:is |has been )?registered", 3),
        (r"\bofficer\b", 2), (r"sim (?:will be )?blocked", 3),
        (r"immediately|within \d+ ?(?:hours|hrs|hour)|last warning|final notice|turant", 1),
    ],
    PHISH: [
        (r"verify|verification", 2), (r"billing|subscription", 2), (r"click|\blink\b", 1), (r"log ?in|password", 2),
        (r"update (?:your|now)|confirm your", 1), (r"https?://|www\.", 1),
        (r"suspended|deactivated|expire", 1),
    ],
}
MIN_SCORE = 3


def scam_type_scores(text):
    t = (text or "").lower()
    return {typ: sum(w for pat, w in rules if re.search(pat, t)) for typ, rules in RULES.items()}


def detect_scam_type(text):
    scores = scam_type_scores(text)
    best = max(TYPE_ORDER, key=lambda k: (scores[k], -TYPE_ORDER.index(k)))
    return best if scores[best] >= MIN_SCORE else OTHER

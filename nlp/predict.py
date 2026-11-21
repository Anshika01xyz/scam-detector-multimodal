"""Plain-language reasons why a message looks suspicious (rule-based).

Reason names follow the style in docs/interface.md. URL-specific reasons
(suspicious TLD, brand in domain...) belong to Person 2's URL module, so here
we only say that a link is present.
"""
import re

NEGATED = re.compile(r"(?:do not|don't|dont|never|not|won't)\s+\w*\s*$")

RULES = [
    ("urgency language",
     r"urgent|immediately|right now|asap|within \d+ ?(?:hours?|hrs?|minutes?|mins?)|\btoday\b|tonight|"
     r"last warning|final notice|expires?|expiring|expired|hurry|turant|abhi|aaj raat|limited seats"),
    ("KYC keyword",
     r"\bkyc\b|verify (?:your )?(?:pan|account|aadhaar|details)|update (?:your )?(?:pan|aadhaar)"),
    ("threat of account block or service cut",
     r"will be (?:blocked|suspended|frozen|deactivated|disconnected|terminated|cut|closed)|"
     r"account (?:is |has been )?(?:blocked|suspended|frozen|deactivated|on hold)|"
     r"connection (?:will be )?(?:cut|disconnected)|(?:band|block) ho jayega|kat jayega|"
     r"will be (?:cut|stopped)|put on hold|disconnection|(?:is|are|been) (?:deactivated|blocked|suspended|frozen)"),
    ("prize or reward claim",
     r"\bwon\b|winner|lucky draw|lottery|\bprize\b|selected for|free (?:iphone|laptop|gift|tv|recharge|data)|"
     r"cashback|\breward|jeete hain"),
    ("unrealistic earnings offer",
     r"(?:earn|income)\s+(?:rs\.?|₹)?\s?\d[\d,\-]*.{0,40}(?:daily|per day|per task|per month|a day)|"
     r"guaranteed returns|work from home|part[- ]time (?:job|offer)|ghar baithe|\bearn\b.{0,30}\btask"),
    ("asks for payment or fee",
     r"(?:processing|registration|delivery|clearance|security|redelivery|joining|shipping)\s+(?:fee|charge|deposit)|"
     r"\bfee\b.{0,15}\bonly\b|(?:pay|send|deposit)\s+(?:rs\.?|₹|inr)\s?\d|"
     r"\bsend rs\b|pay .{0,20}(?:fee|charge)|bhej kar"),
    ("impersonates authority",
     r"\bcbi\b|police|customs|income tax|\bcourt\b|arrest|case (?:is |has been )?registered|\bofficer\b"),
    ("asks to call or message a number",
     r"\bcall\b.{0,25}(?:\+?\d[\d ]{6,}|officer|now|immediately|customer care)|whatsapp (?:now|\+?\d)|"
     r"contact .{0,25}(?:whatsapp|\+?\d[\d ]{6,})|\bwhatsapp\b.{0,10}kare"),
    ("moves chat to Telegram or WhatsApp", r"telegram|t\.me/|join .{0,20}group"),
    ("refund or payment request",
     r"refund|collect request|\bqr\b|by mistake|return it to my"),
    ("parcel or delivery problem",
     r"parcel|courier|package|shipment|delivery (?:failed|attempt)|redeliver|customs|\breschedule\b"),
    ("bill overdue or disconnection threat", r"electricity|power bill|bijli|bill .{0,15}overdue"),
    ("subscription or billing problem", r"subscription|billing"),
    ("suspicious loan offer or threat",
     r"pre-approved|instant loan|loan (?:is )?approved|no documents|loan repayment|inform all your contacts|aapka loan"),
    ("asks for card or bank details",
     r"(?:confirm|enter|share|update).{0,15}(?:bank|card|billing|account) details"),
    ("pushes you to click a link", r"\bclick\b|tap (?:here|link)|link par"),
]
OTP_ASK = re.compile(r"(share|sharing|send|enter|give|tell|provide)\s+(?:your\s+|the\s+|an?\s+|this\s+)?"
                     r"(?:otp|pin|upi pin|password|cvv)\b")


def asks_for_secret(text):
    if "otp" in text and re.search(r"(?<!not )(?<!never )\bshare (?:it|this|the code)\b", text):
        return True
    for m in OTP_ASK.finditer(text):
        before = text[max(0, m.start() - 20):m.start()]
        if not NEGATED.search(before):
            return True
    return False


def get_reasons(text, urls=(), max_reasons=5):
    t = (text or "").lower()
    found = []
    for name, pat in RULES:
        if re.search(pat, t):
            found.append(name)
    if asks_for_secret(t):
        found.insert(0, "asks for OTP or PIN")
    if urls:
        found.append("contains link")
    return found[:max_reasons]"""Plain-language reasons why a message looks suspicious (rule-based).

Reason names follow the style in docs/interface.md. URL-specific reasons
(suspicious TLD, brand in domain...) belong to Person 2's URL module, so here
we only say that a link is present.
"""
import re

NEGATED = re.compile(r"(?:do not|don't|dont|never|not|won't)\s+\w*\s*$")

RULES = [
    ("urgency language",
     r"urgent|immediately|right now|asap|within \d+ ?(?:hours?|hrs?|minutes?|mins?)|\btoday\b|tonight|"
     r"last warning|final notice|expires?|expiring|expired|hurry|turant|abhi|aaj raat|limited seats"),
    ("KYC keyword",
     r"\bkyc\b|verify (?:your )?(?:pan|account|aadhaar|details)|update (?:your )?(?:pan|aadhaar)"),
    ("threat of account block or service cut",
     r"will be (?:blocked|suspended|frozen|deactivated|disconnected|terminated|cut|closed)|"
     r"account (?:is |has been )?(?:blocked|suspended|frozen|deactivated|on hold)|"
     r"connection (?:will be )?(?:cut|disconnected)|(?:band|block) ho jayega|kat jayega|"
     r"will be (?:cut|stopped)|put on hold|disconnection|(?:is|are|been) (?:deactivated|blocked|suspended|frozen)"),
    ("prize or reward claim",
     r"\bwon\b|winner|lucky draw|lottery|\bprize\b|selected for|free (?:iphone|laptop|gift|tv|recharge|data)|"
     r"cashback|\breward|jeete hain"),
    ("unrealistic earnings offer",
     r"(?:earn|income)\s+(?:rs\.?|₹)?\s?\d[\d,\-]*.{0,40}(?:daily|per day|per task|per month|a day)|"
     r"guaranteed returns|work from home|part[- ]time (?:job|offer)|ghar baithe|\bearn\b.{0,30}\btask"),
    ("asks for payment or fee",
     r"(?:processing|registration|delivery|clearance|security|redelivery|joining|shipping)\s+(?:fee|charge|deposit)|"
     r"\bfee\b.{0,15}\bonly\b|(?:pay|send|deposit)\s+(?:rs\.?|₹|inr)\s?\d|"
     r"\bsend rs\b|pay .{0,20}(?:fee|charge)|bhej kar"),
    ("impersonates authority",
     r"\bcbi\b|police|customs|income tax|\bcourt\b|arrest|case (?:is |has been )?registered|\bofficer\b"),
    ("asks to call or message a number",
     r"\bcall\b.{0,25}(?:\+?\d[\d ]{6,}|officer|now|immediately|customer care)|whatsapp (?:now|\+?\d)|"
     r"contact .{0,25}(?:whatsapp|\+?\d[\d ]{6,})|\bwhatsapp\b.{0,10}kare"),
    ("moves chat to Telegram or WhatsApp", r"telegram|t\.me/|join .{0,20}group"),
    ("refund or payment request",
     r"refund|collect request|\bqr\b|by mistake|return it to my"),
    ("parcel or delivery problem",
     r"parcel|courier|package|shipment|delivery (?:failed|attempt)|redeliver|customs|\breschedule\b"),
    ("bill overdue or disconnection threat", r"electricity|power bill|bijli|bill .{0,15}overdue"),
    ("subscription or billing problem", r"subscription|billing"),
    ("suspicious loan offer or threat",
     r"pre-approved|instant loan|loan (?:is )?approved|no documents|loan repayment|inform all your contacts|aapka loan"),
    ("asks for card or bank details",
     r"(?:confirm|enter|share|update).{0,15}(?:bank|card|billing|account) details"),
    ("pushes you to click a link", r"\bclick\b|tap (?:here|link)|link par"),
]
OTP_ASK = re.compile(r"(share|sharing|send|enter|give|tell|provide)\s+(?:your\s+|the\s+|an?\s+|this\s+)?"
                     r"(?:otp|pin|upi pin|password|cvv)\b")


def asks_for_secret(text):
    if "otp" in text and re.search(r"(?<!not )(?<!never )\bshare (?:it|this|the code)\b", text):
        return True
    for m in OTP_ASK.finditer(text):
        before = text[max(0, m.start() - 20):m.start()]
        if not NEGATED.search(before):
            return True
    return False


def get_reasons(text, urls=(), max_reasons=5):
    t = (text or "").lower()
    found = []
    for name, pat in RULES:
        if re.search(pat, t):
            found.append(name)
    if asks_for_secret(t):
        found.insert(0, "asks for OTP or PIN")
    if urls:
        found.append("contains link")
    return found[:max_reasons]

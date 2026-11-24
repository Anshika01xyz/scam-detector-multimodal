"""Generate extra TRAINING examples that fix weaknesses found on the hard test set.

    python nlp/make_extra_training_data.py      ->  data/raw/my_scams_v2.csv

Why: the first training data had (a) links almost only in scams, (b) no legit promotions,
official notices or scam warnings, and (c) few scams without links or urgency. Models learned
the shortcut "link = scam". These templates cover those gaps (legit messages WITH links,
scams WITHOUT links, etc.).

Messages are built from templates + random details, so wording differs from the hand-written
test sets. The script also drops any message that is too similar to a test message
(data/test/hard_test.csv and data/screenshots/labels.csv) so the test sets stay clean.
"""
import csv
import random
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nlp"))
from clean import clean_text  # noqa: E402

rng = random.Random(2026)
P = rng.choice

BANKS = ["SBI", "HDFC Bank", "ICICI Bank", "Axis Bank", "Kotak Bank", "PNB", "Bank of Baroda",
         "Canara Bank", "Union Bank", "IDFC First Bank", "Yes Bank", "IndusInd Bank"]
SHOPS = ["Amazon", "Flipkart", "Myntra", "Swiggy", "Zomato", "Nykaa", "Meesho", "BigBasket", "Ajio",
         "Croma", "Blinkit", "Zepto", "Tata CLiQ", "Lenskart"]
TELCOS = ["Jio", "Airtel", "Vi", "BSNL"]
COURIERS = ["Delhivery", "Ekart", "BlueDart", "DTDC", "Xpressbees", "India Post", "Ecom Express"]
NAMES = ["Rohit", "Neha", "Amit", "Sneha", "Vikram", "Pooja", "Arjun", "Kavita", "Suresh", "Meena",
         "Rahul", "Anjali", "Deepak", "Ritu", "Manish", "Divya"]
FAM = ["Mummy", "Papa", "Bhaiya", "Didi", "Uncle", "Beta"]
MERCH = ["Swiggy", "Zomato", "Big Bazaar", "Reliance Fresh", "Uber", "Ola", "Dmart", "IRCTC", "BookMyShow", "Apollo Pharmacy"]
UTIL = ["electricity", "power", "gas", "water", "broadband", "postpaid"]
PRIZES = ["iPhone 15", "Samsung smart TV", "Rs 25 lakh cash prize", "gold coin set", "Royal Enfield bike", "laptop", "Rs 5 lakh lottery"]
TLDS = [".top", ".xyz", ".click", ".shop", ".in.net", ".online", ".site", ".live", ".icu"]
KW = ["kyc", "verify", "update", "refund", "reward", "secure", "login", "claim", "support", "bonus", "offer", "pay", "help"]
LEGIT_MEET = ["https://meet.google.com/qwe-rtyu-iop", "https://zoom.us/j/84512345678", "https://teams.microsoft.com/l/meetup-join/19%3ameeting",
              "https://meet.jit.si/ProjectSync", "https://us02web.zoom.us/j/91234567890"]
LEGIT_DOC = ["https://docs.google.com/document/d/1AbCdEf", "https://drive.google.com/file/d/1XyZ/view", "https://forms.gle/Ab12CdEf",
             "https://github.com/team/project", "https://www.youtube.com/watch?v=a1B2c3", "https://notion.so/team/Plan-12345"]
LEGIT_SHOP = ["https://www.amazon.in/gp/your-account/order-details", "https://www.flipkart.com/account/orders",
              "https://www.myntra.com/my/orders", "https://www.swiggy.com/my-account/orders", "https://www.nykaa.com/orders",
              "https://www.bigbasket.com/order/track", "https://www.zomato.com/orders"]
LEGIT_OFFICIAL = ["https://www.incometax.gov.in", "https://uidai.gov.in", "https://www.irctc.co.in", "https://digilocker.gov.in",
                  "https://www.sbi.co.in", "https://www.hdfcbank.com", "https://www.icicibank.com", "https://www.epfindia.gov.in",
                  "https://www.passportindia.gov.in", "https://parivahan.gov.in"]


def amt(lo=50, hi=60000):
    return f"Rs {rng.randint(lo // 10, hi // 10) * 10:,}"


def date():
    return f"{rng.randint(1, 28)} {P(['Nov', 'Dec', 'Jan', 'Feb'])}"


def date2():
    return f"{rng.randint(1, 28):02d}-{P(['Nov', 'Dec'])}-26"


def acct():
    return f"XX{rng.randint(1000, 9999)}"


def otp():
    return str(rng.randint(1000, 999999))


def ref():
    return str(rng.randint(10**8, 10**11))


def phone():
    return f"{P(['+91 ', ''])}9{rng.randint(1000, 9999)} {rng.randint(10000, 99999)}"


def tm():
    return f"{rng.randint(1, 12)} {P(['AM', 'PM'])}"


def fake_url(brand=None):
    b = (brand or P(BANKS + SHOPS + TELCOS)).lower().replace(" bank", "").replace(" ", "")
    return f"{P(['http://', 'https://', ''])}{b}-{P(KW)}{P(TLDS)}"


# ---------------------------------------------------------------- LEGIT
def legit_links():
    return P([
        lambda: f"{P(['Meeting', 'Zoom', 'Teams', 'Google Meet'])} link for {P(['today', 'tomorrow', 'Monday'])}'s {P(['standup', 'class', 'review call', 'project discussion'])}: {P(LEGIT_MEET)}",
        lambda: f"Here is the {P(['notes', 'assignment', 'slides', 'report', 'project doc'])} for {P(['Friday', 'the review', 'next week', 'the exam'])}: {P(LEGIT_DOC)}. Please check when free.",
        lambda: f"Your {P(SHOPS)} order #{ref()[:8]} has shipped. Track it here: {P(LEGIT_SHOP)}",
        lambda: f"{P(BANKS)}: View your e-statement anytime at {P(LEGIT_OFFICIAL)}. Never share your password or OTP with anyone.",
        lambda: f"Reminder: your {P(['insurance', 'licence', 'passport appointment', 'ITR'])} is due on {date()}. Details at {P(LEGIT_OFFICIAL)}",
        lambda: f"Please fill the {P(['attendance', 'feedback', 'registration', 'lunch'])} form before {tm()}: {P(LEGIT_DOC)}",
        lambda: f"Download your {P(['e-Aadhaar', 'ITR acknowledgement', 'Form 16', 'ticket'])} from the official site: {P(LEGIT_OFFICIAL)}",
        lambda: f"{P(NAMES)} shared a {P(['video', 'playlist', 'photo album'])} with you: {P(LEGIT_DOC)}",
    ])()


def legit_promo():
    return P([
        lambda: f"{P(SHOPS)}: Flat {rng.randint(10, 70)}% off on {P(['winter wear', 'electronics', 'shoes', 'kitchenware', 'books'])}. Use code {P(['SAVE', 'FEST', 'NEW', 'WOW'])}{rng.randint(10, 99)}. Valid till {date()}. T&C apply.",
        lambda: f"{P(SHOPS)} Sale is live! Up to {rng.randint(30, 80)}% off. Shop now in the app.",
        lambda: f"Congratulations! You earned {rng.randint(50, 900)} reward points on your recent {P(SHOPS)} order. Redeem them in the app.",
        lambda: f"{P(TELCOS)}: Recharge with {amt(149, 999)} and get {P(['1.5GB extra data', 'a free OTT subscription', '5 days extra validity'])}. Offer valid till {date()}.",
        lambda: f"Your {P(SHOPS)} cashback of {amt(10, 300)} has been credited to your wallet.",
        lambda: f"Hurry! {P(SHOPS)} mega sale ends tonight. Extra {rng.randint(5, 15)}% off with {P(BANKS)} cards.",
        lambda: f"You have won a {amt(50, 500)} discount coupon on your next {P(SHOPS)} order. Use code {P(['LUCKY', 'WIN', 'GIFT'])}{rng.randint(10, 99)} at checkout.",
        lambda: f"{P(SHOPS)}: Free delivery on orders above {amt(299, 999)}. Order now in the app.",
        lambda: f"Free gift on your birthday from {P(SHOPS)}! Open the app and claim your coupon. T&C apply.",
        lambda: f"{P(SHOPS)}: Buy 1 Get 1 free today only on selected items. Download the app for more offers.",
        lambda: f"Win big this festive season with {P(SHOPS)}! Spin the wheel in the app for a chance to get discounts up to {rng.randint(20, 70)}%.",
    ])()


def legit_official():
    return P([
        lambda: f"{P(['Passport Seva', 'RTO', 'Municipal Corp', 'Electricity Board'])}: Your {P(['application', 'request', 'appointment'])} {ref()[:6]} has been received. Status updates will be sent by SMS.",
        lambda: f"{P(TELCOS)}: Your KYC re-verification is due. Please visit the nearest store with ID proof. No link will be sent.",
        lambda: f"{P(BANKS)}: Your KYC is due by {date()}. Visit your branch with ID proof or use the official mobile app. Do not click on links in unknown messages.",
        lambda: f"Income Tax Dept: Your return for AY 2026-27 has been verified. Acknowledgement No. {ref()}.",
        lambda: f"Electricity Board: Planned shutdown on {date()} from {tm()} to {tm()} for maintenance in your area.",
        lambda: f"Property tax of {amt(1000, 9000)} is due by {date()}. Pay at the ward office or the official portal.",
        lambda: f"UIDAI: Your Aadhaar update request URN {ref()} is approved. Download your updated e-Aadhaar from the official portal.",
        lambda: f"EPFO: Your PF contribution of {amt(1000, 5000)} for October has been credited to your account.",
        lambda: f"Your driving licence renewal application {ref()[:7]} is under process. You will get an SMS once it is approved.",
        lambda: f"Gas agency: Cylinder booking confirmed. Delivery within {rng.randint(1, 3)} days. Never pay extra to the delivery person.",
        lambda: f"{P(TELCOS)}: Your postpaid bill of {amt(300, 1500)} is generated. Due date {date()}. Pay on the official app to avoid late fee.",
    ])()


def legit_warning():
    return P([
        lambda: f"Beware of fake {P(['KYC', 'refund', 'electricity bill', 'lottery', 'job offer'])} messages. {P(BANKS)} will never ask for your {P(['OTP', 'PIN', 'CVV', 'password'])} on call, SMS or WhatsApp.",
        lambda: f"Cyber alert: Fraudsters are posing as {P(['bank', 'police', 'courier', 'telecom'])} officials. Do not share OTP or click unknown links. Report at 1930.",
        lambda: f"Warning: messages saying you have won a {P(['lottery', 'prize', 'gift'])} are fraud. Delete them and do not reply.",
        lambda: f"PSA: Got a message asking you to update your {P(['PAN', 'Aadhaar', 'KYC', 'bank details'])} using a link? It is fake. Do not click.",
        lambda: f"{P(BANKS)} Customer Advisory: Never share your card number, CVV, PIN or OTP with anyone, including bank staff.",
        lambda: f"My cousin lost {amt(2000, 50000)} to a fake {P(['KYC', 'courier', 'electricity bill'])} call yesterday. Be careful and never share your OTP.",
        lambda: "Reminder: always check the sender and the website before entering your details. If unsure, call the bank's official number.",
        lambda: "Ignore messages that ask you to pay a fee to claim a prize. They are scams. Report them on cybercrime.gov.in.",
        lambda: f"{P(TELCOS)} will never call you to ask for OTP or to complete KYC over the phone. Report suspicious calls at 1930.",
    ])()


def legit_bank():
    return P([
        lambda: f"{amt()} debited from A/c {acct()} on {date2()} at {P(MERCH)}. Avl Bal {amt(500, 90000)}. Not you? Call your bank immediately.",
        lambda: f"{P(BANKS)}: {amt()} credited to your A/c {acct()} on {date2()} via {P(['NEFT', 'IMPS', 'UPI'])} from {P(NAMES).upper()}.",
        lambda: f"Your {P(BANKS)} credit card {acct()} statement of {amt(500, 40000)} is due on {date()}. Pay via the official app.",
        lambda: f"{P(BANKS)}: EMI of {amt(1000, 20000)} for loan A/c {acct()} has been auto-debited on {date2()}. Thank you.",
        lambda: f"Your FD of {amt(10000, 200000)} with {P(BANKS)} will mature on {date()}. Visit the branch or use net banking to renew.",
        lambda: f"UPI: {amt(20, 5000)} paid to {P(MERCH)} on {date2()}. Ref No {ref()}.",
        lambda: f"{P(BANKS)}: Your A/c {acct()} balance is {amt(500, 90000)} as on {date2()}.",
        lambda: f"{P(BANKS)}: Your debit card {acct()} will expire on {rng.randint(1, 12):02d}/27. A new card has been dispatched to your registered address.",
        lambda: f"Your SIP of {amt(500, 10000)} in the mutual fund was processed on {date2()}.",
        lambda: f"{P(BANKS)}: Standing instruction of {amt(500, 8000)} to {P(MERCH)} executed successfully.",
        lambda: f"Rs {rng.randint(1, 900)}.00 interest credited to your savings account {acct()} on {date2()}.",
    ])()


def legit_otp():
    return P([
        lambda: f"{otp()} is your OTP for {P(SHOPS + BANKS + ['Paytm', 'PhonePe'])}. Do not share it with anyone. Valid for {rng.randint(3, 10)} minutes.",
        lambda: f"Use {otp()} as your {P(['Uber', 'Google', 'Amazon', 'Swiggy'])} verification code. Never share this code.",
        lambda: f"{P(['Zomato', 'Ola', 'Flipkart'])} login code: {otp()}. If you did not request this, ignore this message.",
        lambda: f"OTP for transaction of {amt()} at {P(MERCH)} is {otp()}. Do not share with anyone. - {P(BANKS)}",
    ])()


def legit_delivery():
    return P([
        lambda: f"{P(SHOPS)}: Your order #{ref()[:8]} is out for delivery today. Keep {amt(200, 3000)} ready for cash on delivery.",
        lambda: f"{P(COURIERS)}: Package {ref()} delivered on {date()} at {tm()}.",
        lambda: f"Your {P(SHOPS)} order is on the way. Delivery partner {P(NAMES)} will arrive in {rng.randint(5, 40)} minutes.",
        lambda: f"{P(COURIERS)}: We tried to deliver your parcel today but you were unavailable. Next attempt tomorrow.",
        lambda: f"Your return for {P(SHOPS)} order #{ref()[:8]} is approved. Pickup on {date()}.",
        lambda: f"{P(COURIERS)}: Your shipment {ref()} is delayed due to weather. New delivery date: {date()}.",
    ])()


def legit_plain():
    return P([
        lambda: f"Are we still on for {P(['lunch', 'the movie', 'cricket', 'the call'])} at {tm()}?",
        lambda: f"Can you send me the {P(['notes', 'report', 'photos', 'invoice', 'address'])}?",
        lambda: f"I will be late by {rng.randint(5, 40)} minutes. Please start without me.",
        lambda: f"Send me {amt(100, 2000)} on UPI, I will return it tomorrow. Forgot my wallet.",
        lambda: f"{P(FAM)}: Bring {P(['milk', 'vegetables', 'medicines', 'bread'])} on your way home.",
        lambda: f"Hi {P(NAMES)}, we would like to schedule an interview for the {P(['analyst', 'developer', 'designer', 'support'])} role on {date()}. Please confirm by replying YES.",
        lambda: f"Urgent: the {P(['server', 'wifi', 'printer', 'VPN'])} is down. Please join the call now.",
        lambda: f"Congratulations {P(NAMES)} on your {P(['promotion', 'new job', 'results', 'wedding'])}!",
        lambda: f"Rent for {P(['November', 'December'])} is due. Please pay by the 5th via UPI.",
        lambda: f"Kal {P(['class', 'match', 'meeting'])} ke liye kitne baje milna hai?",
        lambda: f"Bhai {amt(100, 1500)} UPI pe bhej de, kal wapas kar dunga.",
        lambda: f"{P(FAM)} ne kaha ghar jaldi aana, {P(['dinner', 'pooja', 'guests'])} hai.",
        lambda: f"Your {P(['electricity', 'water', 'phone'])} bill is {amt(200, 2500)} this month. I have paid it already.",
        lambda: f"Doctor appointment confirmed for {P(['tomorrow', 'Friday', 'Monday'])} at {tm()}. Please carry old reports.",
    ])()


# ---------------------------------------------------------------- SCAM
def scam_subtle():
    return P([
        lambda: f"Hello {P(NAMES)}, we reviewed your profile and have a {P(['remote', 'part-time', 'data entry'])} opening paying {amt(20000, 60000)} per month. Reply YES to know more.",
        lambda: f"Greetings from {P(['Bajaj Finserv', 'HDB Finance', 'Fullerton', 'Tata Capital'])}. You are pre-approved for a personal loan of {amt(100000, 500000)}. Interested? Reply 1 and our agent will call you.",
        lambda: f"Dear customer, your {P(['KYC', 'PAN', 'account'])} needs verification as per RBI rules. Kindly send your account number and date of birth to our executive.",
        lambda: f"This is {P(NAMES)} from {P(BANKS)} customer care. We are processing your refund of {amt(500, 9000)}. Please confirm your card number to proceed.",
        lambda: f"Hi, a package in your name is held at customs. A small duty of {amt(1000, 9000)} must be paid to release it. Message us for details.",
        lambda: f"Congratulations! You were chosen for the {P(SHOPS)} loyalty reward. Our manager will call you to share the details.",
        lambda: f"Dear user, a small fee of {amt(20, 300)} is pending against your account. Kindly clear it with our agent to avoid inconvenience.",
        lambda: f"Limited investment scheme with {rng.randint(10, 40)}% monthly returns. Many members are already earning. Message me for details.",
        lambda: f"Sir, your insurance policy bonus of {amt(50000, 300000)} is ready for release. Please share your bank details with our agent.",
        lambda: f"Hi, I am {P(NAMES)} from {P(SHOPS)} HR. We need part-time reviewers. Earn {amt(500, 4000)} a day from your phone. Ping me on WhatsApp.",
        lambda: f"We noticed an unusual login on your account. For safety, please reply with the 6 digit code sent to you.",
        lambda: f"You qualify for a government subsidy of {amt(5000, 50000)}. Send your Aadhaar and bank account number to claim it.",
        lambda: f"Hello, your {P(UTIL)} connection details need an update. Please call our officer for assistance today.",
    ])()


def scam_conv():
    return P([
        lambda: f"Hi {P(FAM)}, I lost my phone, this is my new number. Please send {amt(5000, 30000)} on GPay urgently, I will explain later.",
        lambda: f"Sir, your relative is in hospital after an accident. Please transfer {amt(10000, 50000)} to this UPI id right away: help{rng.randint(10, 99)}@upi",
        lambda: "Hello, I think I have the wrong number but you seem nice. How are you? I do forex trading and earn well, want to learn?",
        lambda: f"This is {P(NAMES)} from the {P(BANKS)} security team. To stop fraud on your card, tell me the code you just received.",
        lambda: f"Boss here. I am in a meeting. Buy {rng.randint(3, 10)} gift cards of Rs 1000 and send me the codes. Will reimburse today.",
        lambda: "Hello, I am calling from Microsoft support. Your computer has a virus. Please install AnyDesk so we can fix it.",
        lambda: "Dear user, our system shows suspicious activity. Share the OTP you receive to secure your account.",
        lambda: f"Hi, I am {P(NAMES)}, army officer posted far away. Selling my bike cheap. Pay advance {amt(2000, 9000)} to book, delivery by army courier.",
        lambda: f"You have a pending traffic fine. Pay {amt(500, 3000)} now to the officer on this number to avoid court summons: {phone()}",
        lambda: f"I accidentally sent {amt(1000, 20000)} to your account. Please return it to this UPI id: {P(NAMES).lower()}{rng.randint(10, 99)}@upi",
        lambda: f"Madam, I am from the electricity office. Your meter is faulty and we need {amt(500, 2500)} for replacement today. Please pay to this number.",
        lambda: f"Hi {P(NAMES)}, remember me? I got a new number. Can you help me with {amt(3000, 15000)} for an emergency? I will return it Monday.",
    ])()


def scam_std():
    return P([
        lambda: f"{P(BANKS)}: Your account will be {P(['blocked', 'suspended', 'frozen'])} today. Complete KYC now: {fake_url()}",
        lambda: f"Dear {P(SHOPS)} customer, you have won {P(PRIZES)}. Claim here: {fake_url()} before {date()}",
        lambda: f"{P(TELCOS)}: Your SIM will be disconnected. Verify at {fake_url()}",
        lambda: f"Pay {amt(99, 2500)} {P(['delivery charge', 'processing fee', 'registration fee'])} to receive your {P(PRIZES)}. Call {phone()}",
        lambda: f"{P(SHOPS)}: An order of {P(['iPhone 15', 'laptop', 'gold coin'])} worth {amt(20000, 120000)} was placed from your account. Not you? Call {phone()} to cancel",
        lambda: f"Your {P(UTIL)} bill of {amt(500, 6000)} is overdue. Pay now at {fake_url()} else connection will be cut tonight. Call {phone()}",
        lambda: f"A case is registered against you. Call {phone()} immediately to avoid arrest.",
        lambda: f"Earn {amt(1000, 6000)} daily from home. No investment. Join Telegram t.me/{P(['earn', 'cash', 'task'])}{rng.randint(10, 99)}_india",
        lambda: f"Your {P(BANKS)} reward points of {amt(2000, 9000)} expire today. Redeem: {fake_url()}",
        lambda: f"Income tax refund of {amt(5000, 40000)} approved. Submit bank details: {fake_url('incometax')}",
        lambda: f"Instant loan up to {amt(50000, 500000)} without documents. Pay {amt(500, 3000)} processing fee first.",
        lambda: f"Your {P(SHOPS)} account is on hold. Update payment method at {fake_url()}",
        lambda: f"{P(COURIERS)}: Parcel held. Pay {amt(20, 150)} redelivery fee here: {fake_url(P(COURIERS))}",
        lambda: f"{P(BANKS)} alert: {amt(5000, 50000)} debit attempted on your card. If not you, call {phone()} immediately.",
    ])()


def scam_hinglish():
    return P([
        lambda: f"{P(BANKS)} ka aapka account band ho jayega. Abhi KYC update karein: {fake_url()}",
        lambda: f"Badhai ho! Aapne {P(PRIZES)} jeeta hai. Paane ke liye {amt(500, 5000)} processing fees bhejein.",
        lambda: f"Aapka {P(UTIL)} bill pending hai, aaj raat connection kat jayega. Turant call karein {phone()}",
        lambda: f"Ghar baithe {amt(1000, 5000)} roz kamaye. Abhi WhatsApp karein {phone()}",
        lambda: "Sir aapka card block ho gaya hai. Unblock ke liye card number aur OTP bataiye.",
        lambda: f"Aapke naam par ek {P(['parcel', 'refund', 'inaam'])} aaya hai. Release karne ke liye {amt(100, 3000)} jama karein.",
        lambda: f"Aapka {P(['PAN', 'Aadhaar', 'SIM'])} band hone wala hai. Abhi update karein: {fake_url()}",
        lambda: f"Loan chahiye? Bina document ke {amt(50000, 300000)} tak turant. Pehle {amt(500, 2500)} fees bhejein.",
    ])()


LEET = {"a": "@", "o": "0", "i": "1", "e": "3", "s": "$"}


def obfuscate(t):
    out = []
    for w in t.split(" "):
        r = rng.random()
        if r < 0.35 and not w.lower().startswith(("http", "www", "t.me")):
            w = "".join(LEET[c.lower()] if c.lower() in LEET and rng.random() < 0.5 else c for c in w)
        elif r < 0.43 and len(w) > 3 and "." not in w:
            w = " ".join(w)
        elif r < 0.5:
            w = w.upper()
        out.append(w)
    t = " ".join(out)
    if rng.random() < 0.6:
        t = re.sub(r"\.(top|xyz|click|shop|online|site|live|icu)\b", lambda m: P(["[.]", " . "]) + m.group(1), t)
    return t


def scam_obf():
    return obfuscate(P([scam_std, scam_hinglish, scam_subtle])())


PLAN = [(legit_links, 0, "legit_links", 30), (legit_promo, 0, "legit_promo", 40), (legit_official, 0, "legit_official", 28),
        (legit_warning, 0, "legit_warning", 26), (legit_bank, 0, "legit_bank", 32), (legit_otp, 0, "legit_otp", 10),
        (legit_delivery, 0, "legit_delivery", 16), (legit_plain, 0, "legit_plain", 26),
        (scam_subtle, 1, "scam_subtle", 42), (scam_conv, 1, "scam_conv", 32), (scam_obf, 1, "scam_obf", 26),
        (scam_std, 1, "scam_std", 48), (scam_hinglish, 1, "scam_hinglish", 30)]


def main():
    seen, rows = set(), []
    for fn, label, cat, n in PLAN:
        tries = 0
        got = 0
        while got < n and tries < n * 50:
            tries += 1
            t = fn().strip()
            key = clean_text(t)
            if key in seen:
                continue
            seen.add(key)
            rows.append((t, label, cat))
            got += 1

    # drop anything too similar to the test sets, so the tests stay clean
    refs = []
    for f in ("data/test/hard_test.csv", "data/screenshots/labels.csv"):
        if (ROOT / f).exists():
            refs += pd.read_csv(ROOT / f)["text"].tolist()
    if refs:
        mine = [clean_text(r[0]) for r in rows]
        ref_c = [clean_text(t) for t in refs]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(mine + ref_c)
        sims = (vec.transform(mine) @ vec.transform(ref_c).T).max(axis=1).toarray().ravel()
        kept = [r for r, s in zip(rows, sims) if s < 0.80]
        print(f"dropped {len(rows) - len(kept)} messages too similar to test messages (similarity >= 0.80)")
        rows = kept
    out = ROOT / "data" / "raw" / "my_scams_v2.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        w.writerow(["text", "label", "category"])
        w.writerows(rows)
    df = pd.DataFrame(rows, columns=["text", "label", "category"])
    print(f"wrote {len(df)} rows to {out}: {int(df.label.sum())} scam / {int((1 - df.label).sum())} legit")
    print(df.category.value_counts().to_string())


if __name__ == "__main__":
    main()

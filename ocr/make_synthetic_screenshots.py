"""Generate labelled phone-style screenshots of scam / legit messages for OCR testing.

Run from the project root:   python ocr/make_synthetic_screenshots.py
Creates  data/screenshots/*.jpg|png  and  data/screenshots/labels.csv
(labels.csv has the true message text, so OCR accuracy can be measured).

These messages are NOT in data/raw/my_scams.csv, so they are safe to use as a test set.
All sender names, numbers and links are fictional.
"""
import csv
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "screenshots"
SEED = 7

# (text, label, scam_type, sender)
MESSAGES = [
    # ---- scams ----
    ("Dear customer your Kotak account is suspended. Verify PAN immediately at http://kotak-verify.top", 1, "kyc", "VK-KOTAKB"),
    ("Your Union Bank KYC expires today. Update now to avoid account freeze: http://unionbank-kyc.in.net/up", 1, "kyc", "+91 90000 11122"),
    ("Aapka bank account block ho jayega. Abhi KYC update kare is link par: http://bank-kyc.click", 1, "kyc", "AD-BANKKY"),
    ("Your PAN is inactive. Income tax dept will block your accounts. Click: incometax-pan.xyz/verify", 1, "kyc", "+91 90000 33344"),
    ("Your UPI refund of Rs 3000 is waiting. Enter your PIN on this link to receive: http://upi-refund.shop", 1, "upi", "+91 90000 55566"),
    ("Dear customer, your PhonePe reward of Rs 1500 expires today. Claim here http://phonepe-win.shop", 1, "upi", "VM-PHONEP"),
    ("Rs 8000 cashback received from Google Pay. Accept the collect request to get the money in your account", 1, "upi", "+91 90000 77788"),
    ("You won Rs 15 lakh lucky draw. Pay Rs 3000 fee on WhatsApp +91 90000 99900 to claim your prize", 1, "lottery", "+91 90000 99900"),
    ("Congratulations you have been selected for a free laptop. Pay Rs 799 shipping to receive", 1, "lottery", "AX-OFFERZ"),
    ("Work from home and earn Rs 4000 daily just by rating hotels. Join telegram today t.me/rate_earn_in", 1, "job", "+91 90000 12121"),
    ("Hiring part time. Earn Rs 800 per task, withdraw daily. Registration fee Rs 500 only. WhatsApp now", 1, "job", "+91 90000 23232"),
    ("Your electricity will be cut tonight due to unpaid bill. Call officer 90000 11111 now", 1, "electricity", "+91 90000 11111"),
    ("Your courier is stuck. Pay Rs 30 redelivery charge at http://courier-redeliver.xyz", 1, "parcel", "VM-COURIR"),
    ("Delivery failed: wrong pincode. Confirm address and pay Rs 25 here: http://parcel-fix.in.net/pay", 1, "parcel", "+91 90000 45454"),
    ("Get instant loan Rs 1 lakh without documents. Pay Rs 2000 processing fee first. Apply: quickloan-now.top", 1, "loan", "AD-LOANZZ"),
    ("A case is registered against your Aadhaar. Call CBI officer on +91 90000 67676 within 1 hour to avoid arrest", 1, "other", "+91 90000 67676"),
    # ---- legit ----
    ("Rs 1,200 debited from your account XX4455 at BigBazaar on 20-Nov. Avl bal Rs 5,300.", 0, "none", "VM-HDFCBK"),
    ("Your OTP is 556677 for logging in to your Kotak app. Do not share with anyone.", 0, "none", "VK-KOTAKB"),
    ("Your Ola ride has ended. Total fare Rs 245. Thanks for riding with us.", 0, "none", "OLACAB"),
    ("Your order from Myntra will be delivered tomorrow between 10 AM and 2 PM.", 0, "none", "VM-MYNTRA"),
    ("Dinner at 8? I'll book the table at the Italian place.", 0, "none", "Riya"),
    ("Your Jio recharge of Rs 239 is successful. Validity till 20 Dec.", 0, "none", "JIOPAY"),
    ("Your electricity bill for November is Rs 980. Last date to pay is 30 Nov.", 0, "none", "VM-DISCOM"),
    ("Meeting moved to Monday 11 AM. Please confirm your availability.", 0, "none", "Rahul Office"),
    ("Salary of Rs 52,000 credited to your A/c XX9012 on 30-Nov-26.", 0, "none", "AX-ICICIB"),
    ("Aaj ka match dekhne aa raha hai kya? Sab log mere ghar pe hain.", 0, "none", "Karan"),
    ("Your IRCTC ticket PNR 2345678901 is confirmed. Train 12951 departs 17:00 on 25 Nov.", 0, "none", "IRCTC"),
    ("Hi, the plumber will come at 4 pm tomorrow. Please keep the kitchen free.", 0, "none", "Mom"),
    ("Your package from Amazon has been delivered. Thank you for shopping with us.", 0, "none", "AMAZON"),
    ("Reminder: your EMI of Rs 3,150 is due on 5 Dec. Please maintain sufficient balance.", 0, "none", "VM-BAJAJF"),
    ("Your semester fee receipt has been generated. Download it from the college portal.", 0, "none", "COLLEGE"),
    ("HDFC Bank: Never share your OTP, PIN or password with anyone, even bank staff.", 0, "none", "VM-HDFCBK"),
]

STYLES = {
    #            page bg      header bg    header fg    bubble bg    text fg      meta fg
    "sms_light": ("#FFFFFF", "#F6F6F6", "#111111", "#E9E9EB", "#111111", "#8A8A8E"),
    "sms_dark":  ("#000000", "#1C1C1E", "#FFFFFF", "#26262A", "#FFFFFF", "#8E8E93"),
    "wa_light":  ("#ECE5DD", "#075E54", "#FFFFFF", "#FFFFFF", "#111111", "#7A7A7A"),
    "wa_dark":   ("#0B141A", "#1F2C34", "#E9EDEF", "#202C33", "#E9EDEF", "#8696A0"),
}
URL_RE = re.compile(r"(https?://\S+|www\.\S+|\S+\.(?:top|xyz|click|shop|in\.net|co\.in)\S*|t\.me/\S+|bit\.ly/\S+)", re.I)
FONT_PATHS = ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]
BOLD_PATHS = ["DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]


def load_font(paths, size):
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def wrap_words(words, font, max_w):
    lines, cur = [], []
    for w in words:
        trial = " ".join(cur + [w])
        if cur and font.getlength(trial) > max_w:
            lines.append(cur)
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(cur)
    return lines


def render(text, sender, style, rng):
    page_bg, head_bg, head_fg, bub_bg, txt_fg, meta_fg = STYLES[style]
    W = rng.choice([720, 828, 1080])
    s = W / 720
    font = load_font(FONT_PATHS, int(rng.choice([27, 29, 31]) * s))
    small = load_font(FONT_PATHS, int(20 * s))
    bold = load_font(BOLD_PATHS, int(30 * s))
    link_col = "#6CB6FF" if style.endswith("dark") else "#0B57D0"

    pad, bub_pad = int(26 * s), int(22 * s)
    max_text_w = int(W * 0.72) - 2 * bub_pad
    words = text.split()
    lines = wrap_words(words, font, max_text_w)
    line_h = int(font.size * 1.35)
    bub_w = int(max(max(font.getlength(" ".join(l)) for l in lines), 130 * s)) + 2 * bub_pad
    bub_h = len(lines) * line_h + 2 * bub_pad + int(30 * s)

    status_h, head_h = int(56 * s), int(96 * s)
    body_top = status_h + head_h + int(50 * s)
    content_h = body_top + bub_h + int(80 * s)
    H = max(content_h, int(W * rng.uniform(1.55, 2.05)))

    img = Image.new("RGB", (W, H), page_bg)
    d = ImageDraw.Draw(img)
    # status bar + header
    d.rectangle([0, 0, W, status_h + head_h], fill=head_bg)
    d.text((pad, int(14 * s)), f"{rng.randint(1, 12)}:{rng.randint(0, 59):02d}", font=small, fill=head_fg)
    d.rounded_rectangle([W - pad - int(46 * s), int(18 * s), W - pad, int(36 * s)], radius=4, outline=head_fg, width=2)
    d.text((pad, status_h + int(22 * s)), "<", font=bold, fill=head_fg)
    d.text((pad + int(60 * s), status_h + int(24 * s)), sender, font=bold, fill=head_fg)
    # bubble
    x0 = pad if rng.random() < 0.85 else W - pad - bub_w
    d.rounded_rectangle([x0, body_top, x0 + bub_w, body_top + bub_h], radius=int(24 * s), fill=bub_bg)
    y = body_top + bub_pad
    for line in lines:
        x = x0 + bub_pad
        for w in line:
            is_url = bool(URL_RE.fullmatch(w.strip(".,")))
            col = link_col if is_url else txt_fg
            d.text((x, y), w, font=font, fill=col)
            wl = font.getlength(w)
            if is_url:
                d.line([x, y + font.size + 3, x + wl, y + font.size + 3], fill=col, width=2)
            x += wl + font.getlength(" ")
        y += line_h
    d.text((x0 + bub_pad, body_top + bub_h - int(40 * s)),
           f"{rng.randint(1, 12)}:{rng.randint(0, 59):02d} {rng.choice(['AM', 'PM'])}", font=small, fill=meta_fg)
    return img


def degrade(img, rng):
    """Make the screenshot look like a real, slightly messy photo/screenshot."""
    if rng.random() < 0.5:                       # re-saved / downscaled screenshot
        f = rng.uniform(0.55, 0.9)
        img = img.resize((int(img.width * f), int(img.height * f)), Image.BILINEAR)
    if rng.random() < 0.4:
        img = img.rotate(rng.uniform(-1.5, 1.5), resample=Image.BICUBIC, fillcolor=img.getpixel((2, img.height - 3)))
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.1)))
    if rng.random() < 0.5:
        arr = np.asarray(img).astype(np.float32)
        arr += np.random.default_rng(rng.randint(0, 10**6)).normal(0, rng.uniform(2, 7), arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img


def main():
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (text, label, stype, sender) in enumerate(MESSAGES, 1):
        style = rng.choice(list(STYLES))
        img = degrade(render(text, sender, style, rng), rng)
        ext = "jpg" if rng.random() < 0.7 else "png"
        name = f"shot_{i:02d}_{'scam' if label else 'legit'}.{ext}"
        if ext == "jpg":
            img.save(OUT / name, quality=rng.randint(55, 92))
        else:
            img.save(OUT / name)
        rows.append([name, label, stype, style, text])
    with open(OUT / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        w.writerow(["filename", "label", "scam_type", "style", "text"])
        w.writerows(rows)
    print(f"Created {len(rows)} screenshots in {OUT}")


if __name__ == "__main__":
    main()

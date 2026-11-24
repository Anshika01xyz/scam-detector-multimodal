"""Scam Detector - Streamlit frontend (Person 1).

Run from the project root:
    streamlit run frontend/app.py

Needs the trained models in models/ (tfidf_lr.pkl and/or distilbert_scam/).
Text analysis comes from nlp/predict.py. The URL / final-risk part from Person 2 plugs in
at the spot marked "INTEGRATION POINT" below.
"""
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nlp"))
from predict import BERT_DIR, TFIDF, analyze, resolve_model  # noqa: E402

st.set_page_config(page_title="Scam Detector", page_icon="🛡️", layout="centered")

VERDICTS = {
    "scam": ("🚨 Likely a SCAM", "#d93025", "#fce8e6"),
    "suspicious": ("⚠️ Suspicious", "#b06000", "#fef7e0"),
    "legit": ("✅ Looks safe", "#188038", "#e6f4ea"),
}
EXAMPLES = {
    "— pick an example —": "",
    "KYC scam": "Dear customer your SBI account will be blocked today. Complete KYC now: http://sbi-kyc-verify.top",
    "Lottery scam": "Congratulations! You won Rs 25 lakh in KBC lucky draw. Pay Rs 5000 processing fee to claim your prize.",
    "Fake job": "Earn Rs 4000 daily working from home. No experience needed. Join our Telegram group now.",
    "Real bank alert": "Rs 1,200 debited from A/c XX4455 at BigBazaar on 20-Nov. Avl bal Rs 5,300. Not you? Call your bank.",
    "Real OTP": "Your OTP for login is 482913. Do not share it with anyone. Valid for 10 minutes.",
    "Friend chat": "Hey, are we still meeting at 5 today?",
}


def available_models():
    opts = {}
    if BERT_DIR.exists() and TFIDF.exists():
        opts["Both models averaged"] = "ensemble"
    if BERT_DIR.exists():
        opts["DistilBERT (best on our tests)"] = "distilbert"
    if TFIDF.exists():
        opts["TF-IDF (fast, simpler)"] = "tfidf"
    return opts


def show_result(res):
    if res["text_score"] is None:
        st.warning("No readable text was found in the image. Try a clearer screenshot.")
        return
    title, colour, bg = VERDICTS[res["text_verdict"]]
    st.markdown(
        f"<div style='background:{bg};border-left:8px solid {colour};padding:16px 20px;border-radius:8px;'>"
        f"<div style='font-size:1.5rem;font-weight:700;color:{colour};'>{title}</div>"
        f"<div style='color:#333;'>Scam likelihood: <b>{res['text_score'] * 100:.0f}%</b></div></div>",
        unsafe_allow_html=True)
    st.progress(min(max(res["text_score"], 0.0), 1.0))

    c1, c2 = st.columns(2)
    c1.metric("Scam type", res["scam_type"] or "—")
    if res["ocr_confidence"] is not None:
        c2.metric("Text-reading confidence", f"{res['ocr_confidence'] * 100:.0f}%")

    if res["reasons"]:
        st.subheader("Why it looks suspicious")
        for r in res["reasons"]:
            st.markdown(f"- {r}")
    elif res["text_verdict"] == "legit":
        st.caption("No scam patterns were found in the wording.")

    if res["urls_found"]:
        st.subheader("Links found")
        for u in res["urls_found"]:
            st.code(u, language=None)
        # ---- INTEGRATION POINT (Person 2) -------------------------------------------
        # Send res["urls_found"] to the URL model / risk engine here and show the combined
        # score. Keep the text result above as the "text analysis" part of the page.
        st.info("Link safety check will appear here once the URL model is connected.")
        # -----------------------------------------------------------------------------

    with st.expander("Technical details"):
        shown = res["extracted_text"] if res["input_type"] == "screenshot" else None
        if shown is not None:
            st.text_area("Text read from the screenshot", shown, height=120, disabled=True)
        st.json({k: v for k, v in res.items() if k != "extracted_text"})


def main():
    st.title("🛡️ Scam Message Detector")
    st.write("Paste a message or upload a screenshot. We check the wording and tell you how likely it is to be a scam.")

    models = available_models()
    if not models:
        st.error("No trained model found in the models/ folder. Run nlp/train_tfidf.py (and train_distilbert.py) first.")
        st.stop()
    with st.sidebar:
        st.header("Settings")
        choice = st.selectbox("Model", list(models), help="DistilBERT is used when available.")
        st.caption(f"Using: {resolve_model(models[choice])}")
        st.divider()
        st.caption("Study project. Results are estimates, not proof. When in doubt, never share OTPs, PINs or "
                   "passwords, and report fraud at cybercrime.gov.in or 1930.")
    model = models[choice]

    tab_text, tab_img = st.tabs(["✍️ Message text", "🖼️ Screenshot"])
    with tab_text:
        ex = st.selectbox("Try an example", list(EXAMPLES))
        text = st.text_area("Message", value=EXAMPLES[ex], height=150, key=f"text_{ex}",
                            placeholder="Paste the SMS / WhatsApp message here...")
        if st.button("Analyze message", type="primary", key="go_text"):
            if not text.strip():
                st.warning("Please type or paste a message first.")
            else:
                with st.spinner("Analyzing..."):
                    st.session_state["result"] = analyze(text=text, model=model)
    with tab_img:
        up = st.file_uploader("Upload a screenshot (PNG or JPG)", type=["png", "jpg", "jpeg"])
        if up is not None:
            st.image(up, caption="Your screenshot", width=280)
        if st.button("Analyze screenshot", type="primary", key="go_img"):
            if up is None:
                st.warning("Please upload a screenshot first.")
            else:
                with st.spinner("Reading the screenshot and analyzing..."):
                    with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as f:
                        f.write(up.getvalue())
                    try:
                        st.session_state["result"] = analyze(image=f.name, model=model)
                    except Exception as e:  # missing OCR engine etc.
                        st.session_state.pop("result", None)
                        st.error(f"Could not read the screenshot: {e}")
                    finally:
                        Path(f.name).unlink(missing_ok=True)

    if "result" in st.session_state:
        st.divider()
        show_result(st.session_state["result"])


main()

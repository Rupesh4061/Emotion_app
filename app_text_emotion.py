# app_text_emotion.py
import streamlit as st
from transformers import pipeline
import pandas as pd
import datetime
import json
import os

st.set_page_config(page_title="Text → Emotion Analyzer", layout="centered")

st.title("Text → Emotion Analyzer 🎯")
st.write("Type a sentence and get predicted emotion(s), probabilities, emoji and save results to CSV.")

# -----------------------
# Models we'll use
# Small/fast English model:
EN_MODEL = "bhadresh-savani/distilbert-base-uncased-emotion"
# Multilingual model that works reasonably for many languages (including Hindi)
MULTI_MODEL = "cardiffnlp/twitter-xlm-roberta-base-emotion"
# -----------------------

# Emoji map (label keys must match model labels used)
EMOJI_MAP = {
    "joy": "😄",
    "happy": "😄",
    "sadness": "😢",
    "sad": "😢",
    "anger": "😡",
    "anger/annoyance": "😡",
    "disgust": "🤢",
    "fear": "😨",
    "surprise": "😲",
    "neutral": "😐",
    "love": "❤️",
    "enthusiasm": "🤩"
}

# Utility: map label to emoji (case-insensitive)
def label_to_emoji(label: str):
    l = label.lower()
    return EMOJI_MAP.get(l, "🟦")  # default color block if not found

# Caching loader that can load different models based on selection
@st.cache_resource
def load_pipeline(model_name: str):
    # Use CPU by default (device=-1)
    return pipeline(task="text-classification", model=model_name, return_all_scores=True, device=-1)

# Sidebar options
with st.sidebar:
    st.header("Settings & Examples")
    lang = st.selectbox("Language / Model", ("English (fast)", "Hindi / Multilingual"))
    st.markdown("**Quick examples:**")
    st.write("- I got a promotion today!")
    st.write("- I am so worried about the results.")
    st.write("- You are disgusting.")
    st.write("- मुझे आज बहुत खुश महसूस हो रहा है। (Hindi example)")

    st.markdown("---")
    st.write("Notes:")
    st.write("- English uses a small fast model. Hindi uses a multilingual model.")
    st.write("- First run downloads model weights (may take a minute).")
    st.write("")

# Choose model name
model_name = EN_MODEL if lang.startswith("English") else MULTI_MODEL

# Load model (cached)
with st.spinner(f"Loading model ({'English' if model_name==EN_MODEL else 'Multilingual'}) ..."):
    pipe = load_pipeline(model_name)

# Input area
text = st.text_area("Enter a sentence:", height=140, placeholder="e.g. I'm excited about the new job.")
col1, col2 = st.columns([3,1])

# CSV log path
LOG_CSV = "predictions_log.csv"

def save_prediction(text, lang, top_label, top_score, all_scores):
    row = {
        "timestamp": datetime.datetime.now().isoformat(),
        "text": text,
        "language": lang,
        "top_label": top_label,
        "top_score": float(top_score),
        "all_scores": json.dumps(all_scores, ensure_ascii=False)
    }
    # Append to CSV
    if not os.path.exists(LOG_CSV):
        df = pd.DataFrame([row])
        df.to_csv(LOG_CSV, index=False, encoding="utf-8")
    else:
        df = pd.DataFrame([row])
        df.to_csv(LOG_CSV, mode="a", header=False, index=False, encoding="utf-8")

# Predict button
with col1:
    if st.button("Predict emotion"):
        if not text.strip():
            st.warning("Please enter a sentence to analyze.")
        else:
            with st.spinner("Analyzing..."):
                preds_all = pipe(text)  # returns list of list-of-dicts; for single input index 0
            preds = preds_all[0]
            # Each item: {'label': 'joy', 'score': 0.92}
            # Standardize label names (some models may produce uppercase or different tokens)
            # Sort by score desc
            preds_sorted = sorted(preds, key=lambda x: x["score"], reverse=True)
            top3 = preds_sorted[:3]
            top = top3[0]

            # UI output: top emotion with emoji
            emoji = label_to_emoji(top["label"])
            st.markdown(f"### {emoji}  Predicted emotion: **{top['label']}**")
            st.write(f"Confidence: **{top['score']:.3f}**")

            # Progress bar
            st.progress(int(top["score"] * 100))

            # Show top-3 clearly
            st.markdown("#### Top 3 predictions")
            for i, d in enumerate(top3, start=1):
                em = label_to_emoji(d["label"])
                st.write(f"**{i}. {d['label']}** {em} — {d['score']:.3f}")

            # Prepare dataframe for table and chart
            df = pd.DataFrame(preds_sorted)
            df_display = df.copy()
            df_display["score"] = df_display["score"].map(lambda v: round(v, 3))
            df_display = df_display.rename(columns={"label": "Emotion", "score": "Confidence"})
            st.table(df_display)

            # Bar chart (use numeric floats)
            chart_df = df.set_index("label")["score"]
            st.bar_chart(chart_df)

            # Save prediction to CSV log
            save_prediction(text, lang, top["label"], top["score"], preds_sorted)
            st.success("Saved to log ✅")

with col2:
    st.write("Actions")
    if os.path.exists(LOG_CSV):
        try:
            df_log = pd.read_csv(LOG_CSV)
            st.write(f"Total saved predictions: **{len(df_log):,}**")
            st.download_button(
                label="⬇ Download CSV log",
                data=open(LOG_CSV, "rb").read(),
                file_name="predictions_log.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.write("Could not load CSV log:", e)
    else:
        st.write("No predictions saved yet.")
    st.write("---")
    st.write("Advanced:")
    if st.button("Clear saved CSV"):
        if os.path.exists(LOG_CSV):
            os.remove(LOG_CSV)
            st.success("Log cleared.")
        else:
            st.info("No log file found.")

st.markdown("---")
st.write("Made with Transformers • Model downloads occur only once and are cached locally.")

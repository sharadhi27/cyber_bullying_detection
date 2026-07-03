# ==========================================
# FINAL CYBERBULLYING DETECTION SYSTEM
# ML + Streamlit + All Features
# ==========================================

import streamlit as st
import joblib
import re
import pandas as pd

# ---------------------------
# Load Model
# ---------------------------
model = joblib.load("model/svm_model.pkl")
vectorizer = joblib.load("model/tfidf.pkl")

# ---------------------------
# Functions
# ---------------------------
def clean(text):
    return re.sub(r'[^a-zA-Z ]', '', text.lower())

def get_toxicity(text):
    toxic_words = ["hate","stupid","idiot","loser","dumb","ugly"]
    count = sum(w in text for w in toxic_words)

    if count == 0:
        return "Low"
    elif count <= 2:
        return "Medium"
    else:
        return "High"

def highlight_text(text):
    toxic_words = ["hate","stupid","idiot","loser","dumb","ugly"]
    words = text.split()

    highlighted = []
    for w in words:
        if w.lower() in toxic_words:
            highlighted.append(f"**{w}**")
        else:
            highlighted.append(w)

    return " ".join(highlighted)

# ---------------------------
# UI Config
# ---------------------------
st.set_page_config(page_title="Cyberbullying Detector", layout="centered")

st.title("🧠 Cyberbullying Detection System")
st.write("Analyze text → Detect bullying + toxicity level")

# ---------------------------
# Session History
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------
# Text Input
# ---------------------------
user_input = st.text_area("✍️ Enter Text")

if st.button("Analyze"):

    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean(user_input)

        # Vectorize & Predict
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]

        # Confidence score
        try:
            decision = model.decision_function(vec)[0]
            confidence = round(abs(decision) * 10, 2)
        except:
            confidence = "N/A"

        # Toxicity
        toxicity = get_toxicity(cleaned)

        # Result
        st.subheader("🔍 Result")

        if pred == 1:
            st.error("🚨 Cyberbullying Detected")
        else:
            st.success("😊 Safe Content")

        st.write(f"**Toxicity Level:** {toxicity}")
        st.write(f"**Confidence Score:** {confidence}")

        # Highlight text
        st.subheader("📝 Highlighted Text")
        st.markdown(highlight_text(user_input))

        # Suggestions
        st.subheader("💡 Suggestions")
        if toxicity == "High":
            st.write("- Avoid harmful language")
            st.write("- Content should be moderated")
        elif toxicity == "Medium":
            st.write("- Be mindful of tone")
        else:
            st.write("- Content looks good 👍")

        # Save history
        st.session_state.history.append({
            "text": user_input,
            "prediction": "Bullying" if pred == 1 else "Safe",
            "toxicity": toxicity
        })

# ---------------------------
# Show History
# ---------------------------
st.subheader("📜 History")

if st.session_state.history:
    for item in st.session_state.history[::-1]:
        st.write(item)

# ---------------------------
# CSV Upload (Batch Prediction)
# ---------------------------
st.subheader("📂 Batch Prediction (CSV Upload)")

file = st.file_uploader("Upload CSV with 'text' column", type=["csv"])

if file:
    df = pd.read_csv(file)

    if "text" in df.columns:
        cleaned = df['text'].apply(clean)
        vec = vectorizer.transform(cleaned)

        df['prediction'] = model.predict(vec)
        df['toxicity'] = cleaned.apply(get_toxicity)

        st.write(df)
    else:
        st.error("CSV must contain a 'text' column")       app.py code

import streamlit as st
import joblib
nb = joblib.load("nb.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📢 Fake News / Malicious Message Detector")

st.write("Type a message below and click **Check**. The app will decide if it's Safe or Malicious.")

# User input
user_input = st.text_area("Enter your message:")

if st.button("Check"):
    if user_input.strip():
        X = vectorizer.transform([user_input])
        pred = nb.predict(X)[0]
        proba = nb.predict_proba(X)[0].max()

        if pred == 0:
            st.success(f"✅ Safe Message — Confidence: {proba*100:.2f}%")
        else:
            st.error(f"⚠️ Spam/Malicious — Confidence: {proba*100:.2f}%")
    else:
        st.warning("Please enter some text.")
import streamlit as st
import joblib
log_reg = joblib.load("log_reg.pkl")
nb = joblib.load("nb.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📢 Fake News / Malicious Message Detector")

user_input = st.text_area("Enter your message:")

model_choice = st.selectbox("Choose a model:", ["Logistic Regression", "Naive Bayes"])

if st.button("Check"):
    if user_input.strip():
        X = vectorizer.transform([user_input])

        if model_choice == "Logistic Regression":
            pred = log_reg.predict(X)[0]
            proba = log_reg.predict_proba(X)[0].max()
        else:
            pred = nb.predict(X)[0]
            proba = nb.predict_proba(X)[0].max()

        if pred == 0:
            st.success(f"✅ Safe Message — Confidence: {proba*100:.2f}%")
        else:
            st.error(f"⚠️ Spam/Malicious — Confidence: {proba*100:.2f}%")
    else:
        st.warning("Please enter some text.")

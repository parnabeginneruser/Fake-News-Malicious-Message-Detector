import streamlit as st
import joblib
log_reg = joblib.load("log_reg.pkl")
nb = joblib.load("nb.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📢 Fake News / Malicious Message Detector")

st.write("Type a message below and click **Check**. The app will decide if it's Safe or Malicious using both models.")

user_input = st.text_area("Enter your message:")

if st.button("Check"):
    if user_input.strip():
        X = vectorizer.transform([user_input])
        pred_lr = log_reg.predict(X)[0]
        proba_lr = log_reg.predict_proba(X)[0].max()

        pred_nb = nb.predict(X)[0]
        proba_nb = nb.predict_proba(X)[0].max()

        if pred_lr == pred_nb:
            final_pred = pred_lr
            final_proba = (proba_lr + proba_nb) / 2
        else:
            final_pred = pred_lr
            final_proba = proba_lr
        if final_pred == 0:
            st.success(f"✅ Safe Message — Confidence: {final_proba*100:.2f}% (combined models)")
        else:
            st.error(f"⚠️ Spam/Malicious — Confidence: {final_proba*100:.2f}% (combined models)")
    else:
        st.warning("Please enter some text.")

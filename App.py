import streamlit as st
import pickle
with open("nb.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
st.title("📢 Fake News / Malicious Message Detector")
st.write("Just type a message below and click **Check**. The app will automatically decide if it's Safe or Malicious.")
user_input = st.text_area("Enter your message:", "")
if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        confidence = model.predict_proba(input_tfidf)[0].max()
        if prediction == 0:
            st.success(f"✅ Safe Message (HAM) — Confidence: {confidence*100:.2f}%")
        else:
            st.error(f"⚠️ Malicious/Spam Message — Confidence: {confidence*100:.2f}%")

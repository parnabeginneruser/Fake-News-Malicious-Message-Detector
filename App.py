import streamlit as st    # web UI library
import joblib             # to load saved model

# 1) Load the trained vectorizer and model
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("model.pkl")

st.title("Fake News & Malicious Message Detector")

user_text = st.text_area("Enter news / message:")

if st.button("Check"):
    if user_text.strip():
        # 2) Convert user text to TF-IDF vector
        X = vectorizer.transform([user_text])

        # 3) Predict label and probability
        pred = model.predict(X)[0]                         # 'ham' or 'spam'
        proba = model.predict_proba(X)[0].max()            # highest class probability

        # 4) Show result
        st.write(f"Prediction: **{pred}**")
        st.write(f"Confidence: {proba*100:.2f}%")
    else:
        st.write("Please enter some text.")

import streamlit as st    
import joblib           
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("model.pkl")

st.title("Fake News & Malicious Message Detector")

user_text = st.text_area("Enter news / message:")

if st.button("Check"):
    if user_text.strip():
        X = vectorizer.transform([user_text])
        pred = model.predict(X)[0]                         
        proba = model.predict_proba(X)[0].max()            
        st.write(f"Prediction: **{pred}**")
        st.write(f"Confidence: {proba*100:.2f}%")
    else:
        st.write("Please enter some text.")

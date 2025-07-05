import streamlit as st
import pandas as pd
import joblib
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
import nltk

nltk.download('vader_lexicon')

# Load model and vectorizer
model = joblib.load('hybrid_model_with_topics.pkl')
vectorizer = joblib.load('tfidf_vectorizer_with_topics.pkl')

# Streamlit UI
st.title("📩 Customer Satisfaction Predictor with Topic & Sentiment")

# User Inputs
ticket_text = st.text_area("📝 Ticket Description", height=150)
age = st.slider("🎂 Customer Age", 18, 70, 35)
gender = st.selectbox("🚻 Gender", ['Male', 'Female', 'Other'])
product = st.selectbox("📦 Product Purchased", ['Product A', 'Product B', 'Product C'])
priority = st.selectbox("⚠️ Ticket Priority", ['High', 'Medium', 'Low'])
channel = st.selectbox("🌐 Ticket Channel", ['Email', 'Chat', 'Phone'])

# Analyze sentiment
if st.button("🔍 Analyze Sentiment"):
    sid = SentimentIntensityAnalyzer()
    score = sid.polarity_scores(ticket_text)['compound']
    sentiment = 'Positive' if score >= 0.05 else 'Negative' if score <= -0.05 else 'Neutral'
    st.write(f"🧠 **Sentiment:** `{sentiment}` ({score})")

# Predict satisfaction
if st.button("⭐ Predict Satisfaction"):
    # Encode structured fields
    gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
    product_map = {'Product A': 0, 'Product B': 1, 'Product C': 2}
    priority_map = {'High': 0, 'Medium': 1, 'Low': 2}
    channel_map = {'Email': 0, 'Chat': 1, 'Phone': 2}

    # Sample logic to infer topic (optional random or sentiment-based mock)
    # You can replace this with actual LDA model later
    dominant_topic = 1 if "refund" in ticket_text.lower() else 3 if "delay" in ticket_text.lower() else 0

    # Create structured input
    structured_input = np.array([[age,
                                  gender_map[gender],
                                  product_map[product],
                                  priority_map[priority],
                                  channel_map[channel],
                                  dominant_topic]])

    # Vectorize ticket text
    X_text = vectorizer.transform([ticket_text])

    # Combine structured + text
    X_combined = hstack([structured_input, X_text])

    # Predict
    predicted_rating = model.predict(X_combined)[0]

    st.markdown(f"🌟 **Predicted Satisfaction Rating:** `{int(predicted_rating)} / 5`")
    st.markdown(f"🧩 **Detected Topic ID:** `{dominant_topic}`")

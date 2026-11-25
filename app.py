import streamlit as st
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)
import pickle
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords if not available
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

# Load stopwords
stop_words = set(stopwords.words("english"))

# Porter Stemmer
ps = PorterStemmer()


# -----------------------------
# Text Preprocessing
# -----------------------------
def stemming(content):

    content = str(content)

    content = re.sub(r'[^a-zA-Z\s]', ' ', content)

    content = content.lower()

    content = content.split()

    content = [
        ps.stem(word)
        for word in content
        if word not in stop_words
    ]

    return " ".join(content)


# -----------------------------
# Load Model & Vectorizer
# -----------------------------
@st.cache_resource
def load_model():

    model = pickle.load(open("model.pkl", "rb"))

    vector = pickle.load(open("vectorizer.pkl", "rb"))

    return model, vector


# Load once
with st.spinner("Loading Model..."):

    model, vector = load_model()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection using Machine Learning")

st.write(
    "Paste a news article below and click **Predict** "
    "to determine whether it is Fake or Real."
)

st.success("Model Accuracy : 94.49%")


# User Input
input_text = st.text_area(
    "Paste News Article Here",
    height=250
)


# -----------------------------
# Prediction Function
# -----------------------------
def predict_news(news):

    news = stemming(news)

    news_vector = vector.transform([news])

    prediction = model.predict(news_vector)[0]

    confidence = np.max(
        model.predict_proba(news_vector)
    ) * 100

    return prediction, confidence


# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict"):

    if input_text.strip() == "":

        st.warning("Please enter a news article.")

    else:

        prediction, confidence = predict_news(input_text)

        st.subheader("Prediction")

        st.write(f"**Confidence:** {confidence:.2f}%")

        if prediction == 1:

            st.error("🚨 Fake News")

        else:

            st.success("✅ Real News")

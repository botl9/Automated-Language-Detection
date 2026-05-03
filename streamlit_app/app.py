from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "dataset" / "language-detection.csv"


st.set_page_config(
    page_title="Language Detection Demo",
    page_icon="🌐",
    layout="centered",
)


@st.cache_data
def load_dataset():
    data = pd.read_csv(DATASET_PATH)
    data = data.dropna(subset=["Text", "Language"])
    data["Text"] = data["Text"].astype(str).str.strip()
    data = data[data["Text"] != ""]
    return data


@st.cache_resource
def train_model():
    data = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        data["Text"],
        data["Language"],
        test_size=0.2,
        random_state=42,
    )

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 3),
                    max_features=5000,
                ),
            ),
            ("classifier", MultinomialNB()),
        ]
    )

    model.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(x_test))
    return model, accuracy, sorted(data["Language"].unique())


model, accuracy, languages = train_model()

st.title("Automated Language Detection")
st.caption("ML Surprise Test - 3")

st.write(
    "Enter any text below. The model uses character-level TF-IDF n-grams "
    "with a Multinomial Naive Bayes classifier."
)

sample_texts = {
    "English": "Language detection is useful for multilingual applications.",
    "French": "Bonjour, comment allez-vous aujourd'hui?",
    "Spanish": "La inteligencia artificial puede identificar idiomas.",
    "Hindi": "भाषा पहचान प्राकृतिक भाषा प्रसंस्करण का एक महत्वपूर्ण कार्य है।",
}

sample = st.selectbox("Try a sample", ["Custom text", *sample_texts.keys()])
default_text = "" if sample == "Custom text" else sample_texts[sample]

text = st.text_area(
    "Input text",
    value=default_text,
    height=150,
    placeholder="Type or paste text here...",
)

if st.button("Detect Language", type="primary", use_container_width=True):
    cleaned = text.strip()

    if not cleaned:
        st.warning("Please enter some text first.")
    else:
        prediction = model.predict([cleaned])[0]
        probabilities = model.predict_proba([cleaned])[0]
        confidence = probabilities.max()

        st.subheader(prediction)
        st.progress(float(confidence))
        st.write(f"Confidence: **{confidence:.2%}**")

        top_predictions = (
            pd.DataFrame(
                {
                    "Language": model.classes_,
                    "Confidence": probabilities,
                }
            )
            .sort_values("Confidence", ascending=False)
            .head(5)
        )
        top_predictions["Confidence"] = top_predictions["Confidence"].map("{:.2%}".format)

        st.write("Top predictions")
        st.dataframe(top_predictions, hide_index=True, use_container_width=True)

st.divider()

col1, col2, col3 = st.columns(3)
col1.metric("Test Accuracy", f"{accuracy:.2%}")
col2.metric("Languages", len(languages))
col3.metric("Features", "5000")

with st.expander("Model pipeline"):
    st.code(
        "Text → TF-IDF Vectorizer (char n-grams 2-3) → Multinomial Naive Bayes → Language",
        language="text",
    )

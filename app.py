import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch

st.set_page_config(
    page_title="Nizam-e-AI",
    layout="wide"
)

st.title("🧠 Nizam-e-AI")
st.caption("Roman Urdu Poetry Analysis, Search & Recommendation System")

@st.cache_resource
def load_models():
    tfidf_model = joblib.load("nizam_e_ai_author_model.pkl")
    tfidf_vectorizer = joblib.load("nizam_e_ai_tfidf.pkl")

    sbert_clf = joblib.load("nizam_sbert_author_clf.pkl")
    label_encoder = joblib.load("nizam_author_encoder.pkl")

    embeddings = np.load("nizam_poem_embeddings.npy")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sbert = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2",
        device=device
    )

    return tfidf_model, tfidf_vectorizer, sbert_clf, label_encoder, embeddings, sbert

@st.cache_data
def load_data():
    df = pd.read_csv("collected_dataset.csv")
    return df

model, tfidf, sbert_clf, le, embeddings, sbert_model = load_models()
df = load_data()

def normalize_roman(text):
    text = str(text).lower()
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def keyword_search(keyword, top_k=10):
    keyword = keyword.lower().strip()
    pattern = rf'\b{re.escape(keyword)}\b'

    matches = df[df['clean_Aashar'].str.contains(
        pattern, regex=True, case=False, na=False
    )]

    return matches[['Shayer', 'Aashar']].head(top_k)

def semantic_recommend(query, top_k=5):
    query = normalize_roman(query)
    q_emb = sbert_model.encode([query])
    sims = cosine_similarity(q_emb, embeddings)[0]
    top_idx = sims.argsort()[-top_k:][::-1]

    res = df.iloc[top_idx][['Shayer', 'Aashar']].copy()
    res['similarity'] = sims[top_idx]
    return res

st.sidebar.header("🔎 Search Mode")

mode = st.sidebar.radio(
    "Choose search type:",
    ["Author Identification", "Keyword Search", "Semantic Search", "Hybrid Search"]
)

top_k = st.sidebar.slider("Results", 3, 10, 5)

user_input = st.text_area(
    "Enter Roman Urdu couplet / poem:",
    height=120
)


if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text.")
        st.stop()

    st.divider()


    if mode == "Author Identification":
        pred = model.predict([user_input])[0]
        probs = model.predict_proba([user_input])[0]

        top_idx = probs.argsort()[-3:][::-1]
        st.subheader("✍️ Predicted Author")
        for i in top_idx:
            st.write(f"**{model.classes_[i]}** — {probs[i]*100:.2f}%")

    elif mode == "Keyword Search":
        results = keyword_search(user_input, top_k)
        if results.empty:
            st.info("No Aashar were found with this exact keyword.")
        else:
            st.subheader("🔍 Keyword Matches")
            for _, row in results.iterrows():
                st.markdown(f"**{row['Shayer']}**")
                st.write(row['Aashar'])
                st.divider()

    elif mode == "Semantic Search":
        results = semantic_recommend(user_input, top_k)
        st.subheader("🧠 Semantic Recommendations")
        for _, row in results.iterrows():
            st.markdown(f"**{row['Shayer']}** (score: {row['similarity']:.3f})")
            st.write(row['Aashar'])
            st.divider()

    elif mode == "Hybrid Search":
        kw = keyword_search(user_input, top_k)
        if not kw.empty:
            st.subheader("🔍 Keyword Matches")
            for _, row in kw.iterrows():
                st.markdown(f"**{row['Shayer']}**")
                st.write(row['Aashar'])
                st.divider()
        else:
            st.subheader("🧠 Semantic Fallback")
            results = semantic_recommend(user_input, top_k)
            for _, row in results.iterrows():
                st.markdown(f"**{row['Shayer']}** (score: {row['similarity']:.3f})")
                st.write(row['Aashar'])
                st.divider()

                
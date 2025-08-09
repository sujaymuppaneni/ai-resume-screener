import streamlit as st
import pdfplumber
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

# Load models
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Functions
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def calculate_match_score(resume_text, jd_text):
    documents = [resume_text, jd_text]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(score * 100, 2)

def calculate_semantic_score(resume_text, jd_text):
    embeddings = model.encode([resume_text, jd_text])
    cosine_sim = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    return round(cosine_sim * 100, 2)

def calculate_final_score(resume_text, jd_text, weight_tfidf=0.4, weight_semantic=0.6):
    tfidf_score = calculate_match_score(resume_text, jd_text)
    semantic_score = calculate_semantic_score(resume_text, jd_text)
    final_score = (tfidf_score * weight_tfidf) + (semantic_score * weight_semantic)
    return round(final_score, 2), tfidf_score, semantic_score

# Streamlit App
st.title("📄 AI Resume Screener")
st.write("Upload one or more resumes and a job description to get match scores.")

jd_input = st.text_area("Paste Job Description here:")
uploaded_resumes = st.file_uploader("Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)

if st.button("Run Matching"):
    if jd_input and uploaded_resumes:
        cleaned_jd = clean_text(jd_input)
        results = []

        for resume_file in uploaded_resumes:
            extracted_text = extract_text_from_pdf(resume_file)
            cleaned_resume = clean_text(extracted_text)
            final_score, tfidf_score, semantic_score = calculate_final_score(cleaned_resume, cleaned_jd)
            results.append({
                "Resume": resume_file.name,
                "Final Score (%)": final_score,
                "TF-IDF Score (%)": tfidf_score,
                "Semantic Score (%)": semantic_score
            })

        df = pd.DataFrame(results).sort_values(by="Final Score (%)", ascending=False).reset_index(drop=True)
        st.dataframe(df)
    else:
        st.warning("Please enter a job description and upload at least one resume.")

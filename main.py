import streamlit as st
import fitz
import spacy
import subprocess
import importlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    importlib.invalidate_caches()
    nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")

st.title("📄 AI Resume Screener & Matcher")

st.markdown("""
Upload your **resume** and paste a **job description** to see:
- ✅ Your **match score**
- 🔍 **Missing important keywords**
- 📄 Extracted resume text
""")

uploaded_file = st.file_uploader("Upload your resume (PDF format only)", type=["pdf"])
job_description = st.text_area("Paste the Job Description here", height=200)

if uploaded_file is not None:
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        resume_text = ""
        for page in doc:
            resume_text += page.get_text()

    st.subheader("📄 Extracted Resume Text")
    st.write(resume_text if resume_text.strip() else "⚠ No text found in resume. Check if the PDF is scanned.")

    if job_description.strip():
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume_text, job_description])
        similarity_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100

        st.subheader("📊 Match Score")
        st.write(f"✅ Your resume matches **{similarity_score:.2f}%** with the job description")

        resume_doc = nlp(resume_text.lower())
        jd_doc = nlp(job_description.lower())

        resume_words = set([token.text for token in resume_doc if token.is_alpha and not token.is_stop])
        jd_words = set([token.text for token in jd_doc if token.is_alpha and not token.is_stop])

        missing_keywords = jd_words - resume_words

        st.subheader("🔍 Missing Keywords")
        if missing_keywords:
            st.write(", ".join(sorted(missing_keywords)))
        else:
            st.write("🎉 No major keywords missing — great job!")

else:
    st.info("📌 Please upload your resume to get started.")

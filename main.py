import streamlit as st
import pandas as pd
import pdfplumber
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import re

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Preprocess text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

# Function to calculate similarity
def get_similarity(resume_text, job_desc):
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    job_embedding = model.encode(job_desc, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(resume_embedding, job_embedding)
    return similarity_score.item() * 100  # percentage

# Streamlit UI
st.title("📄 AI Resume Screener (No spaCy)")
st.write("Upload your resume and enter a job description to see how well it matches!")

# File upload
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description")

if uploaded_file and job_description:
    with st.spinner("Processing..."):
        resume_text = extract_text_from_pdf(uploaded_file)
        resume_text = clean_text(resume_text)
        score = get_similarity(resume_text, job_description)

    st.subheader("✅ Match Score")
    st.write(f"Your resume matches **{score:.2f}%** with the job description.")

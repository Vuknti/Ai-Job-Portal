# AI Job Portal MVP (Streamlit)
# Features:
# - Job posting form (Recruiters)
# - Resume upload (Job Seekers)
# - AI resume-job match score
# - AI-generated cover letter (simple)

import streamlit as st
import base64
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uuid

st.set_page_config(page_title="AI Job Portal MVP", layout="wide")

# In-memory storage (MVP only)
jobs_db = []
candidates_db = []

st.title("🚀 AI-Powered Job Portal MVP")

# Sidebar - User Type
user_type = st.sidebar.radio("I am a:", ["Job Seeker", "Recruiter"])

# ------------------ Recruiter Section ------------------
if user_type == "Recruiter":
    st.header("Post a Job")
    with st.form("job_post_form"):
        job_title = st.text_input("Job Title")
        company = st.text_input("Company Name")
        job_desc = st.text_area("Job Description")
        skills_required = st.text_input("Required Skills (comma-separated)")
        submit_job = st.form_submit_button("Post Job")

    if submit_job:
        job_id = str(uuid.uuid4())
        jobs_db.append({
            "id": job_id,
            "title": job_title,
            "company": company,
            "description": job_desc,
            "skills": skills_required.lower().split(",")
        })
        st.success("✅ Job posted successfully!")

    if jobs_db:
        st.subheader("📌 All Posted Jobs")
        for job in jobs_db:
            st.markdown(f"**{job['title']}** at *{job['company']}*\n\n{job['description']}")
            st.markdown("---")

# ------------------ Job Seeker Section ------------------
else:
    st.header("Apply for Jobs with AI Assistance")
    name = st.text_input("Your Name")
    email = st.text_input("Email")
    resume_file = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"])

    if resume_file:
        doc = fitz.open(stream=resume_file.read(), filetype="pdf")
        resume_text = "".join([page.get_text() for page in doc]).lower()
        st.success("✅ Resume uploaded and processed.")

        # Show jobs and match
        for job in jobs_db:
            st.markdown(f"### {job['title']} at {job['company']}")
            jd = job['description'].lower()
            tfidf = TfidfVectorizer()
            matrix = tfidf.fit_transform([resume_text, jd])
            score = round(cosine_similarity(matrix[0:1], matrix[1:2])[0][0] * 100, 2)
            st.markdown(f"**Match Score:** {score}%")

            # Cover letter
            if score > 60:
                cover_letter = f"""
                Dear {job['company']} Team,

                I am excited to apply for the {job['title']} role. With my experience and skills aligned with your job description, I am confident I would be a valuable addition to your team.

                I would welcome the opportunity to contribute and grow with {job['company']}.

                Sincerely,
                {name}
                """
                st.markdown("**📄 Suggested Cover Letter:**")
                st.text(cover_letter)

                # Download button
                b64 = base64.b64encode(cover_letter.encode()).decode()
                href = f'<a href="data:file/txt;base64,{b64}" download="cover_letter.txt">📩 Download Cover Letter</a>'
                st.markdown(href, unsafe_allow_html=True)

            st.markdown("---")

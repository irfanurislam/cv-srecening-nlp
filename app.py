from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = os.urandom(24).hex()

# Load the skills dataset
skills_df = pd.read_csv('skills_dataset.csv')

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Define job description for a frontend developer
job_description = """
We are looking for a talented Frontend Developer with strong skills in JavaScript, Node.js, Express, React, and Firebase.
The ideal candidate should have hands-on experience in building modern and responsive web applications.
Key Responsibilities:
- Collaborate with the design team to implement user-friendly interfaces
- Develop and maintain web applications using React.js
- Work with backend developers to integrate frontend components
- Optimize applications for maximum speed and scalability
- Stay updated with the latest frontend technologies and best practices
Skills and Requirements:
- Proficient in JavaScript, HTML, and CSS
- Experience with Node.js, Express, React, and Firebase
- Knowledge of responsive design and cross-browser compatibility
- Strong problem-solving and communication skills
If you are passionate about creating exceptional user experiences and have a solid background in frontend development, we encourage you to apply.
"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'resume' in request.files:
        resume = request.files['resume']
        if resume.filename != '' and resume.filename.endswith('.pdf'):
            # Securely save the file using secure_filename
            filename = secure_filename(resume.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            resume.save(filepath)

            # Process the uploaded PDF file
            name, email, skills_match, recommended_skills = extract_info_and_calculate_match(filepath)

            # Provide feedback to the user
            flash(f"Resume uploaded successfully! Skills Match: {skills_match:.2%}", 'success')

            return render_template('upload.html', filename=filename, name=name, email=email,
                                   skills_match=skills_match, recommended_skills=recommended_skills)
        else:
            flash("Invalid file format. Please upload a PDF file.", 'danger')
    else:
        flash("No file uploaded.", 'danger')

    return redirect(url_for('index'))

def extract_info_and_calculate_match(filepath):
    # Read CV content using PyMuPDF
    cv_text, name, email = extract_text_and_info_from_pdf(filepath)

    # Combine job description and CV content
    corpus = [job_description.lower(), cv_text.lower()]

    # Calculate TF-IDF features
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)[0, 1]

    # Extract recommended skills based on the cosine similarity
    recommended_skills = recommend_skills(cosine_sim)

    return name, email, cosine_sim, recommended_skills

def extract_text_and_info_from_pdf(filepath):
    text = ""
    name = ""
    email = ""
    with fitz.open(filepath) as pdf_document:
        num_pages = pdf_document.page_count
        for page_num in range(num_pages):
            page = pdf_document[page_num]
            text += page.get_text()
            # Extract name and email (you may need to enhance this based on your specific CV format)
            if "Name:" in text and "Email:" in text:
                start_name = text.find("Name:")
                end_name = text.find("Email:")
                name = text[start_name + 5:end_name].strip()
                start_email = text.find("Email:") + 6
                email = text[start_email:].strip()
    return text, name, email

def recommend_skills(cosine_sim):
    # Use a simple threshold for illustration
    threshold = 0.2
    if cosine_sim > threshold:
        # Select top recommended skills
        top_skills = skills_df['skill'].tolist()[:3]
        return top_skills
    else:
        return []

if __name__ == '__main__':
    app.run(debug=True)

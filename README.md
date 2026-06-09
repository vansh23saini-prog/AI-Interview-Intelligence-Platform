# AI Interview Intelligence Platform

AI Interview Intelligence Platform is a full-stack Flask web application that helps users practice interviews by uploading their resume, selecting a job role, answering generated questions, receiving AI-style scoring, tracking performance history, and downloading PDF reports.

## Features

- User signup and login
- Secure password hashing
- Dashboard with interview analytics
- Resume PDF upload
- Resume skill extraction
- Role-based interview question generation
- Easy, medium, and hard difficulty levels
- Technical and HR interview questions
- NLP-based answer scoring using TF-IDF and cosine similarity
- Question-wise feedback
- Weak area detection
- Interview history
- Performance chart using Chart.js
- Downloadable PDF report
- SQLite database
- GitHub-ready project structure

## Tech Stack

- Python
- Flask
- Flask-SQLAlchemy
- SQLite
- HTML
- CSS
- JavaScript
- Chart.js
- PyPDF2
- Scikit-learn
- TF-IDF
- Cosine Similarity
- ReportLab
- Gunicorn

## How to Run

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Resume Description

Developed a full-stack AI Interview Intelligence Platform using Flask, SQLite, NLP, TF-IDF, cosine similarity, Chart.js, and ReportLab to generate interview questions, evaluate answers, track performance, and export PDF reports.

## Future Improvements

- Add OpenAI/Gemini API based question generation
- Add voice-based interview answers
- Add webcam proctoring mode
- Add admin dashboard
- Add company-wise interview preparation
- Deploy using Render

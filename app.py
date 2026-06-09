import os
import re
import string
from datetime import datetime
from functools import wraps

from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

app = Flask(__name__)
app.config["SECRET_KEY"] = "change-this-secret-key-later"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///interview_platform.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["REPORT_FOLDER"] = "reports"

db = SQLAlchemy(app)
ALLOWED_EXTENSIONS = {"pdf"}

SKILLS_DB = [
    "python", "java", "c++", "c", "html", "css", "javascript", "react", "node",
    "flask", "django", "sql", "mysql", "mongodb", "sqlite", "machine learning",
    "deep learning", "data science", "pandas", "numpy", "matplotlib",
    "scikit-learn", "tensorflow", "nlp", "computer vision", "git", "github",
    "docker", "aws", "azure", "linux", "api", "rest api", "oop", "dbms",
    "operating system", "computer networks", "communication", "leadership",
    "problem solving", "teamwork", "data analysis", "bootstrap", "chart.js"
]

QUESTION_BANK = {
    "python developer": [
        {"question": "Explain the difference between list, tuple, set, and dictionary in Python.", "keywords": "list tuple set dictionary mutable immutable key value unique"},
        {"question": "What is OOP and explain inheritance with an example.", "keywords": "object oriented programming class object inheritance polymorphism encapsulation abstraction"},
        {"question": "What are decorators in Python?", "keywords": "decorator function wrapper higher order function modify behavior"},
        {"question": "Explain exception handling in Python.", "keywords": "try except finally error handling exception"},
        {"question": "What is the difference between shallow copy and deep copy?", "keywords": "shallow copy deep copy reference nested object"}
    ],
    "web developer": [
        {"question": "Explain how a web request works from browser to server.", "keywords": "browser request server response http dns client backend frontend"},
        {"question": "What is the difference between GET and POST methods?", "keywords": "get post http request data url body secure"},
        {"question": "Explain responsive web design.", "keywords": "responsive design media query mobile desktop css flexible layout"},
        {"question": "What is REST API?", "keywords": "rest api endpoint http methods json client server"},
        {"question": "What is the difference between authentication and authorization?", "keywords": "authentication identity login authorization permission access"}
    ],
    "data analyst": [
        {"question": "What is data cleaning and why is it important?", "keywords": "missing values duplicates outliers formatting accuracy quality"},
        {"question": "Explain mean, median, and mode.", "keywords": "mean average median middle mode frequent statistics"},
        {"question": "What is correlation?", "keywords": "correlation relationship variables positive negative strength"},
        {"question": "What is data visualization?", "keywords": "charts graphs visualization insights dashboard matplotlib"},
        {"question": "Explain the difference between classification and regression.", "keywords": "classification category regression continuous prediction machine learning"}
    ],
    "machine learning engineer": [
        {"question": "Explain supervised and unsupervised learning.", "keywords": "supervised labeled data unsupervised unlabeled clustering classification regression"},
        {"question": "What is overfitting and how can you reduce it?", "keywords": "overfitting training testing regularization cross validation pruning"},
        {"question": "Explain train-test split.", "keywords": "train test split model evaluation unseen data accuracy"},
        {"question": "What is Random Forest?", "keywords": "random forest ensemble decision trees bagging classification regression"},
        {"question": "What are precision, recall, and accuracy?", "keywords": "precision recall accuracy classification metrics true false positive negative"}
    ]
}

HR_QUESTIONS = [
    {"question": "Tell me about yourself.", "keywords": "education skills projects experience goal strength"},
    {"question": "Why should we hire you?", "keywords": "skills hardworking projects learning problem solving team"},
    {"question": "What are your strengths and weaknesses?", "keywords": "strength weakness improvement learning honest"},
    {"question": "Where do you see yourself in five years?", "keywords": "career growth learning responsibility company goals"},
    {"question": "Describe a challenging project you worked on.", "keywords": "project challenge problem solution teamwork result"}
]

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Interview(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    job_role = db.Column(db.String(120), nullable=False)
    difficulty = db.Column(db.String(50), nullable=False)
    resume_skills = db.Column(db.Text)
    weak_skills = db.Column(db.Text)
    score = db.Column(db.Float, default=0)
    total_questions = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Answer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    interview_id = db.Column(db.Integer, db.ForeignKey("interview.id"), nullable=False)
    question = db.Column(db.Text, nullable=False)
    user_answer = db.Column(db.Text, nullable=False)
    expected_keywords = db.Column(db.Text, nullable=False)
    score = db.Column(db.Float, default=0)
    feedback = db.Column(db.Text)

def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login first.", "error")
            return redirect(url_for("login"))
        return func(*args, **kwargs)
    return wrapper

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    except Exception:
        return ""
    return text

def extract_skills(text):
    cleaned = clean_text(text)
    found = []
    for skill in SKILLS_DB:
        if skill in cleaned:
            found.append(skill)
    return sorted(set(found))

def score_answer(user_answer, expected_keywords):
    if not user_answer.strip():
        return 0, "No answer provided. Try to explain the concept with examples."
    documents = [clean_text(user_answer), clean_text(expected_keywords)]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(documents)
    similarity = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    score = round(similarity * 100, 2)
    answer_words = len(user_answer.split())
    keyword_list = clean_text(expected_keywords).split()
    matched_keywords = [kw for kw in keyword_list if kw in clean_text(user_answer)]
    if answer_words < 15:
        feedback = "Answer is too short. Add definition, example, and real-world use."
    elif score >= 70:
        feedback = "Strong answer. You covered most important points."
    elif score >= 40:
        feedback = "Average answer. Add more keywords and explain with an example."
    else:
        feedback = "Weak answer. Revise the concept and include important keywords."
    if matched_keywords:
        feedback += " Matched keywords: " + ", ".join(sorted(set(matched_keywords[:8])))
    return score, feedback

def generate_questions(job_role, difficulty):
    role_key = job_role.lower().strip()
    questions = QUESTION_BANK.get(role_key, QUESTION_BANK["web developer"])
    if difficulty == "easy":
        return questions[:3] + HR_QUESTIONS[:2]
    if difficulty == "medium":
        return questions[:4] + HR_QUESTIONS[:3]
    return questions[:5] + HR_QUESTIONS[:5]

def wrap_text(text, limit):
    words = str(text).split()
    lines, current = [], []
    for word in words:
        current.append(word)
        if len(" ".join(current)) > limit:
            lines.append(" ".join(current[:-1]))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines

def generate_pdf_report(interview, answers, user):
    os.makedirs(app.config["REPORT_FOLDER"], exist_ok=True)
    path = os.path.join(app.config["REPORT_FOLDER"], f"interview_report_{interview.id}.pdf")
    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4
    y = height - 50
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, "AI Interview Intelligence Report")
    y -= 35
    c.setFont("Helvetica", 11)
    for item in [f"Candidate: {user.name}", f"Email: {user.email}", f"Job Role: {interview.job_role}", f"Difficulty: {interview.difficulty}", f"Overall Score: {interview.score}%"]:
        c.drawString(50, y, item); y -= 20
    y -= 10
    c.setFont("Helvetica-Bold", 13); c.drawString(50, y, "Resume Skills Found:"); y -= 18
    c.setFont("Helvetica", 10)
    for line in wrap_text(interview.resume_skills or "No skills detected.", 85):
        c.drawString(50, y, line); y -= 14
    y -= 15
    c.setFont("Helvetica-Bold", 13); c.drawString(50, y, "Question-wise Feedback:"); y -= 25
    for idx, ans in enumerate(answers, start=1):
        if y < 130:
            c.showPage(); y = height - 50
        c.setFont("Helvetica-Bold", 10)
        for line in wrap_text(f"Q{idx}. {ans.question}", 90):
            c.drawString(50, y, line); y -= 13
        c.setFont("Helvetica", 10)
        c.drawString(50, y, f"Score: {ans.score}%"); y -= 13
        for line in wrap_text(f"Feedback: {ans.feedback}", 90):
            c.drawString(50, y, line); y -= 13
        y -= 12
    c.save()
    return path

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form["name"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        if User.query.filter_by(email=email).first():
            flash("Email already registered. Please login.", "error")
            return redirect(url_for("login"))
        user = User(name=name, email=email, password=generate_password_hash(password))
        db.session.add(user); db.session.commit()
        flash("Account created successfully. Please login.", "success")
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            session["user_name"] = user.name
            flash("Login successful.", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.", "error")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("index"))

@app.route("/dashboard")
@login_required
def dashboard():
    interviews = Interview.query.filter_by(user_id=session["user_id"]).order_by(Interview.created_at.desc()).limit(5).all()
    total = Interview.query.filter_by(user_id=session["user_id"]).count()
    avg_score = db.session.query(db.func.avg(Interview.score)).filter_by(user_id=session["user_id"]).scalar()
    avg_score = round(avg_score, 2) if avg_score else 0
    labels = [i.created_at.strftime("%d %b") for i in reversed(interviews)]
    scores = [i.score for i in reversed(interviews)]
    return render_template("dashboard.html", interviews=interviews, total=total, avg_score=avg_score, labels=labels, scores=scores)

@app.route("/start", methods=["GET", "POST"])
@login_required
def start_interview():
    if request.method == "POST":
        job_role = request.form["job_role"]
        difficulty = request.form["difficulty"]
        file = request.files.get("resume")
        resume_skills = []
        if file and file.filename:
            if allowed_file(file.filename):
                filename = secure_filename(file.filename)
                os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
                path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(path)
                resume_skills = extract_skills(extract_text_from_pdf(path))
            else:
                flash("Only PDF resume files are allowed.", "error")
                return redirect(url_for("start_interview"))
        questions = generate_questions(job_role, difficulty)
        interview_obj = Interview(user_id=session["user_id"], job_role=job_role, difficulty=difficulty, resume_skills=", ".join(resume_skills), weak_skills="", total_questions=len(questions))
        db.session.add(interview_obj); db.session.commit()
        session["current_questions"] = questions
        session["current_interview_id"] = interview_obj.id
        return redirect(url_for("interview"))
    return render_template("start_interview.html")

@app.route("/interview", methods=["GET", "POST"])
@login_required
def interview():
    questions = session.get("current_questions")
    interview_id = session.get("current_interview_id")
    if not questions or not interview_id:
        flash("Please start a new interview first.", "error")
        return redirect(url_for("start_interview"))
    if request.method == "POST":
        total_score = 0
        weak_keywords = []
        for idx, q in enumerate(questions):
            user_answer = request.form.get(f"answer_{idx}", "")
            score, feedback = score_answer(user_answer, q["keywords"])
            total_score += score
            if score < 50:
                weak_keywords.extend(q["keywords"].split()[:5])
            db.session.add(Answer(interview_id=interview_id, question=q["question"], user_answer=user_answer, expected_keywords=q["keywords"], score=score, feedback=feedback))
        final_score = round(total_score / len(questions), 2)
        interview_obj = Interview.query.get(interview_id)
        interview_obj.score = final_score
        interview_obj.weak_skills = ", ".join(sorted(set(weak_keywords)))
        db.session.commit()
        session.pop("current_questions", None)
        session.pop("current_interview_id", None)
        return redirect(url_for("result", interview_id=interview_id))
    return render_template("interview.html", questions=questions)

@app.route("/result/<int:interview_id>")
@login_required
def result(interview_id):
    interview_obj = Interview.query.get_or_404(interview_id)
    if interview_obj.user_id != session["user_id"]:
        flash("Access denied.", "error")
        return redirect(url_for("dashboard"))
    answers = Answer.query.filter_by(interview_id=interview_id).all()
    return render_template("result.html", interview=interview_obj, answers=answers)

@app.route("/history")
@login_required
def history():
    interviews = Interview.query.filter_by(user_id=session["user_id"]).order_by(Interview.created_at.desc()).all()
    return render_template("history.html", interviews=interviews)

@app.route("/download_report/<int:interview_id>")
@login_required
def download_report(interview_id):
    interview_obj = Interview.query.get_or_404(interview_id)
    if interview_obj.user_id != session["user_id"]:
        flash("Access denied.", "error")
        return redirect(url_for("dashboard"))
    answers = Answer.query.filter_by(interview_id=interview_id).all()
    user = User.query.get(session["user_id"])
    path = generate_pdf_report(interview_obj, answers, user)
    return send_file(path, as_attachment=True)

@app.cli.command("init-db")
def init_db():
    db.create_all()
    print("Database created successfully.")

if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["REPORT_FOLDER"], exist_ok=True)
    with app.app_context():
        db.create_all()
    app.run(debug=True)

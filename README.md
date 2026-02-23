# 🎯 Recruiter OS - Agentic Recruitment Platform

Recruiter OS is an AI-powered recruitment automation platform designed to streamline end-to-end hiring workflows. From job description extraction to automated interview scheduling and workload monitoring, it leverages agentic AI to assist recruiters in managing their daily tasks efficiently.

## 🚀 Key Features

- **🤖 AI Requirement Intelligence**: Automatically extracts job titles, priority, urgency, and required skills from uploaded JDs (PDF, DOCX, TXT) or pasted text.
- **📋 Task Management**: Smart task creation and assignment system to prevent duplicates and ensure one-at-a-time task focus.
- **📅 Interview Scheduling Agent**: Automated interview booking with Google Calendar synchronization and candidate email invites.
- **📊 Workload & EOD Summary**:
    - **Workload Monitor**: Real-time analysis of recruiter capacity and risk levels using AI.
    - **Comprehensive EOD**: Automated generation of daily activity reports, emailed directly to recruiters.
- **💬 AI Chatbot Assistant**: Embedded support agent to help recruiters navigate the system and manage candidates.
- **🛡️ Admin Dashboard**: Centralized control for managing teams, assigning tasks, and monitoring team-wide workload metrics.

## 🛠️ Technology Stack

- **Backend**: Python, FastAPI
- **Database**: MongoDB (Motor / Pymongo)
- **Frontend**: Vanilla JS, HTML5, CSS3 (Modern Glassmorphism UI)
- **AI/ML**: 
    - HuggingFace Transformers (BART for summarization)
    - Google Gemini / OpenAI (for intelligent extraction and chat)
- **Scheduler**: APScheduler for automated EOD triggers

## 📦 Installation & Setup

### 1. Prerequisites
- Python 3.9+
- MongoDB instance (Local or Atlas)
- Google Cloud Project (for Calendar API integration)

### 2. Clone the Repository
```bash
git clone https://github.com/Maniteja-Iglobus/Recruiter_OS.git
cd Recruiter_OS
```

### 3. Environment Configuration
Create a `.env` file in the root directory and add:
```env
MONGODB_URL=your_mongodb_connection_string
MONGODB_DB=recruiter_os
GEMINI_API_KEY=your_gemini_api_key
SMTP_USER=your_email@gmail.com
SMTP_PASS=your_app_password
GOOGLE_API_CREDENTIALS=path/to/credentials.json
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Application
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

## 📂 Project Structure

- `app.py`: Main FastAPI application logic and API endpoints.
- `static/`: Frontend assets (Admin dashboard, CSS, and Chat widget).
- `llm_extraction.py`: Helper scripts for AI-driven data extraction.
- `requirements.txt`: Project dependencies.

## 🤝 Contributing
Feel free to fork this project and submit pull requests for any improvements or bug fixes.

---
*Built with ❤️ for Modern Recruitment Teams*

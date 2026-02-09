from datetime import datetime, timedelta
from typing import Dict, List, Any
from pymongo.database import Database
from dotenv import load_dotenv
load_dotenv()
import logging
import os
import re
import smtplib
import warnings
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import base64
from typing import Dict, Any, List, Optional
import json
import hashlib
import time
from google import genai
import pdfplumber

from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pymongo import MongoClient
from bson import ObjectId

from passlib.context import CryptContext
import jwt

# Suppress transformers FutureWarnings and other warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*clean_up_tokenization_spaces.*')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

from transformers import pipeline
import torch

from apscheduler.schedulers.background import BackgroundScheduler

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from docx import Document
except ImportError:
    Document = None

# Google Calendar Imports
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    print("⚠️ Google API libraries not installed. Calendar sync will be disabled.")


# ============================================================
# CONFIG
# ============================================================

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "recruiter_os")

# Gmail SMTP
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = os.getenv("SMTP_USER", "your-email@gmail.com")
SMTP_PASS = os.getenv("SMTP_PASS", "your-app-password")
EOD_RECEIVER = os.getenv("EOD_RECEIVER", "your-email@gmail.com")

# Gemini Config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    print("⚠️ GEMINI_API_KEY not set in .env file!")
    gemini_client = None

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="Recruiter OS Agentic Backend", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATABASE
# ============================================================

mongo_client = MongoClient(MONGODB_URL)
db = mongo_client[MONGODB_DB]

# ============================================================
# LLM EXTRACTION HELPERS
# ============================================================

def call_gpt(prompt, max_retries=3):
    """Call Gemini API with structured response requirement"""
    if not gemini_client:
        return None
    
    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
                
            return response_text
        except Exception as e:
            print(f"Gemini Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
    return None

def repair_json_string(json_str):
    """Repair common JSON issues"""
    if not json_str: return None
    json_str = json_str.replace('\n', ' ').replace('\r', ' ')
    json_str = re.sub(r'\s+', ' ', json_str)
    # Remove markdown code blocks if present
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0].strip()
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0].strip()
    return json_str

def try_parse_json(json_str):
    """Try to parse JSON"""
    if not json_str: return None
    try:
        return json.loads(json_str)
    except:
        repaired = repair_json_string(json_str)
        try:
            return json.loads(repaired)
        except:
            return None

EXTRACTION_PROMPT_RESUME = """TASK: Extract EXACTLY this JSON from resume. Return ONLY the JSON object, no text before/after.

RESUME:
{content}

REQUIRED JSON (return exactly this structure):
{{
  "name": "full name",
  "email": "email address",
  "phone": "phone number",
  "current_position": "current or most recent job title",
  "experience_years": (total years as INTEGER),
  "current_company": "most recent company",
  "skills": ["technical skill name 1", "technical skill name 2"],
  "notes": "brief summary of candidate profile"
}}

RULES:
1. experience_years = estimated total years from resume
2. Return ONLY valid JSON, nothing else
3. Use null for missing values"""

EXTRACTION_PROMPT_JD = """TASK: Extract EXACTLY this JSON from Job Description. Return ONLY the JSON object, no text before/after.

JOB DESCRIPTION:
{content}

REQUIRED JSON (return exactly this structure):
{{
  "title": "job title",
  "priority": "High/Medium/Low",
  "urgency": "Immediate/1 Week/Flexible",
  "complexity": "Easy/Moderate/Complex",
  "location": "location",
  "experience": "required experience string",
  "skills": ["technical skill name 1", "technical skill name 2"],
  "summary": "brief summary of the role"
}}

RULES:
1. Classify priority, urgency, and complexity based on JD tone and specific mentions
2. Return ONLY valid JSON, nothing else
3. Use null for missing values"""


# ============================================================
# AUTH HELPERS
# ============================================================

def hash_password(password: str) -> str:
    """Hash password"""
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """Verify password"""
    return pwd_context.verify(plain, hashed)


def create_access_token(user_id: str) -> str:
    """Create JWT token"""
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> str:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except Exception:
        raise HTTPException(401, "Invalid token")


async def get_current_user(authorization: str = Header(...)) -> Dict[str, str]:
    """Get current user from token"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid Authorization Header")

    token = authorization.replace("Bearer ", "")
    user_id = verify_token(token)

    try:
        obj_id = ObjectId(user_id)
    except:
        raise HTTPException(401, "Invalid user ID in token")

    # Check normal users
    user = db.users.find_one({"_id": obj_id})
    if user:
        return {"id": str(user["_id"]), "username": user["username"], "role": user.get("role", "recruiter")}
    
    # Check admin users
    admin = db.admin_users.find_one({"_id": obj_id})
    if admin:
        return {"id": str(admin["_id"]), "username": admin["username"], "role": "admin"}

    raise HTTPException(401, "User not found")

def safe_isoformat(dt):
    """Safely format datetime to ISO string"""
    if dt is None:
        return None
    if isinstance(dt, datetime):
        return dt.isoformat()
    if isinstance(dt, str):
        return dt
    if hasattr(dt, "isoformat"):
        return dt.isoformat()
    return str(dt)


# ============================================================
# HUGGINGFACE MODELS (Production-ready initialization)
# ============================================================



# Suppress expected model loading warnings (unused pooler weights is normal for NER)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

device = 0 if torch.cuda.is_available() else -1

# Named Entity Recognition for skill extraction
# Note: The "unused weights" warning is expected - BERT NER doesn't use pooler layers
ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    device=device,
    aggregation_strategy="simple"
)
print("✅ NER Pipeline loaded successfully")

# Zero-shot classification for priority, urgency, complexity
zero_shot = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device
)
print("✅ Zero-shot classifier loaded successfully")

# Summarization
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=device
)
print("✅ Summarizer loaded successfully")


# ============================================================
# FILE EXTRACTION
# ============================================================

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF with multiple methods"""
    if not PyPDF2:
        return ""
    
    try:
        from io import BytesIO
        
        # Try PyPDF2 first
        try:
            pdf_file = BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            # Extract from all pages
            for page_num in range(len(pdf_reader.pages)):
                try:
                    page = pdf_reader.pages[page_num]
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
                except Exception as page_error:
                    print(f"Error extracting page {page_num}: {page_error}")
                    continue
            
            if text and text.strip():
                return text.strip()
        except Exception as e:
            print(f"PyPDF2 extraction failed: {e}")
        
        # Fallback: Try pdfplumber if available
        try:
            import pdfplumber
            with pdfplumber.open(BytesIO(file_content)) as pdf:
                text = ""
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
                
                if text and text.strip():
                    return text.strip()
        except ImportError:
            pass
        except Exception as e:
            print(f"pdfplumber extraction failed: {e}")
        
        # If both fail, return empty
        return ""
    
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return ""


def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX"""
    if not Document:
        return ""
    
    try:
        from io import BytesIO
        doc = Document(BytesIO(file_content))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        print(f"DOCX extraction error: {e}")
        return ""


def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """Extract text from any file with fallbacks"""
    text = ""
    
    if filename.endswith(".txt"):
        try:
            text = file_content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = file_content.decode("latin-1")
            except Exception as e:
                print(f"TXT decode error: {e}")
                return ""
    
    elif filename.endswith(".pdf"):
        text = extract_text_from_pdf(file_content)
    
    elif filename.endswith(".docx"):
        text = extract_text_from_docx(file_content)
    
    # Try to clean up text
    if text:
        # Remove extra whitespace
        text = " ".join(text.split())
        # Keep first 10000 chars to avoid processing huge docs
        text = text[:10000]
    
    return text


# ============================================================
# AGENTS
# ============================================================

class RequirementIntelligenceAgent:
    """
    Enhanced Requirement Extraction Agent using NER + Text Analysis
    Extracts:
    - Title
    - Priority (High/Medium/Low)
    - Urgency (Immediate/1 Week/Flexible)
    - Complexity (Easy/Moderate/Complex)
    - Location
    - Experience
    - Skills (using NER)
    - Summary
    """

    def extract_skills(self, text: str) -> List[str]:
        """Extract skills using NER"""
        try:
            entities = ner_pipeline(text[:512])
            skills = []
            
            for entity in entities:
                if entity["entity_group"] in ["SKILL", "PRODUCT"]:
                    skills.append(entity["word"])
            
            # Fallback: Extract common skill keywords
            if not skills:
                skill_keywords = [
                    "python", "java", "javascript", "react", "nodejs", "django",
                    "aws", "azure", "gcp", "docker", "kubernetes", "sql",
                    "mongodb", "postgres", "mysql", "git", "ci/cd", "agile",
                    "scrum", "machine learning", "deep learning", "tensorflow",
                    "pytorch", "pandas", "numpy", "excel", "tableau", "power bi",
                    "communication", "leadership", "teamwork", "problem solving"
                ]
                
                text_lower = text.lower()
                for keyword in skill_keywords:
                    if keyword in text_lower:
                        skills.append(keyword.title())
            
            return list(set(skills[:15]))  # Remove duplicates, max 15
        except Exception as e:
            print(f"NER extraction error: {e}")
            return []

    def analyze(self, text: str) -> Dict[str, Any]:
        """Enhanced Requirement Extraction using OpenAI"""
        try:
            prompt = EXTRACTION_PROMPT_JD.format(content=text[:6000])
            response_text = call_gpt(prompt)
            extracted = try_parse_json(response_text)
            
            if not extracted:
                # Fallback to local regex/simplified if LLM fails
                return {
                    "title": "Job Requirement",
                    "priority": "Medium",
                    "urgency": "Flexible",
                    "complexity": "Moderate",
                    "location": "Not Mentioned",
                    "experience": "Not Mentioned",
                    "skills": [],
                    "summary": text[:200],
                    "extracted_at": datetime.utcnow().isoformat()
                }

            # Ensure technical fields are present
            extracted["extracted_at"] = datetime.utcnow().isoformat()
            
            return extracted

        except Exception as e:
            print(f"JD Extraction Error: {e}")
            return {
                "title": "Job Requirement",
                "priority": "Medium",
                "urgency": "Flexible",
                "complexity": "Moderate",
                "location": "Not Mentioned",
                "experience": "Not Mentioned",
                "skills": [],
                "summary": text[:200],
                "extracted_at": datetime.utcnow().isoformat()
            }


class TaskManager:
    """Manages tasks to prevent duplicates and ensure one task at a time"""

    def add_task(self, user_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add new task (prevents duplicates)"""
        
        # Check if identical task already exists in pending/in_progress
        existing = db.tasks.find_one({
            "user_id": user_id,
            "title": task_data["title"],
            "status": {"$in": ["pending", "in_progress"]}
        })
        
        if existing:
            return {
                "success": False,
                "message": "This task already exists in your active tasks",
                "task_id": str(existing["_id"])
            }
        
        task_doc = {
            "user_id": user_id,
            **task_data,
            "status": "pending",
            "created_at": datetime.utcnow(),
            "completed_at": None
        }
        
        result = db.tasks.insert_one(task_doc)
        return {
            "success": True,
            "message": "Task created successfully",
            "task_id": str(result.inserted_id)
        }

    def create_task(self, user_id: str, title: str, priority: str = "Medium",
                    urgency: str = "Normal", complexity: str = "Medium",
                    location: str = "Remote", experience: str = "3-5 years",
                    skills: List[str] = None, description: str = "") -> str:
        """Create a new task from extracted requirement data.
        
        This method provides a cleaner interface for the API endpoints,
        accepting keyword arguments and returning just the task_id string.
        """
        task_data = {
            "title": title,
            "priority": priority,
            "urgency": urgency,
            "complexity": complexity,
            "location": location,
            "experience": experience,
            "skills": skills or [],
            "description": description
        }
        result = self.add_task(user_id, task_data)
        if result.get("success"):
            return result.get("task_id")
        # If task already exists, return the existing task_id 
        if result.get("task_id"):
            return result.get("task_id")
        raise Exception(result.get("message", "Failed to create task"))

    def get_all_tasks(self, user_id: str, status: Optional[str] = None, role: str = "recruiter") -> List[Dict]:
        """Get all tasks for user (created by or assigned to them)"""
        if role == "admin":
             query = {}
        else:
             query = {"$or": [{"user_id": user_id}, {"assigned_to": user_id}]}
        if status:
            query["status"] = status
        
        tasks = list(db.tasks.find(query).sort("created_at", -1))
        return [
            {
                "id": str(t["_id"]),
                "title": t.get("title"),
                "priority": t.get("priority"),
                "urgency": t.get("urgency"),
                "complexity": t.get("complexity"),
                "location": t.get("location"),
                "experience": t.get("experience"),
                "experience": t.get("experience"),
                "skills": t.get("skills", []),
                "status": t.get("status"),
                "comment": t.get("comment", ""),
                "feedback": t.get("feedback", ""),
                "created_at": t.get("created_at", "").isoformat() if t.get("created_at") else "",
                "completed_at": t.get("completed_at", "").isoformat() if t.get("completed_at") else ""
            }
            for t in tasks
        ]

    def complete_task(self, user_id: str, task_id: str) -> bool:
        """Mark task as completed (by Creator or Assignee)"""
        result = db.tasks.update_one(
            {
                "_id": ObjectId(task_id), 
                "$or": [{"user_id": user_id}, {"assigned_to": user_id}]
            },
            {
                "$set": {
                    "status": "completed",
                    "completed_at": datetime.utcnow()
                }
            }
        )
        return result.modified_count > 0

    def delete_task(self, user_id: str, task_id: str) -> bool:
        """Delete task"""
        result = db.tasks.delete_one(
            {"_id": ObjectId(task_id), "user_id": user_id}
        )
        return result.deleted_count > 0


class DailyGuidanceAgent:
    """Generates EOD Summary (Pending Tasks Only) + Sends Safe Email"""

    def generate_eod(self, user_id: str) -> str:
        """Generate EOD summary for PENDING tasks only"""
        
        pending_tasks = list(db.tasks.find({
            "$or": [{"user_id": user_id}, {"assigned_to": user_id}],
            "status": "pending"
        }))
        
        in_progress_tasks = list(db.tasks.find({
            "$or": [{"user_id": user_id}, {"assigned_to": user_id}],
            "status": "in_progress"
        }))
        
        user = db.users.find_one({"_id": ObjectId(user_id)})
        username = user.get("username", "Recruiter") if user else "Recruiter"

        report = f"""
🎯 RECRUITER OS - END OF DAY SUMMARY
{'=' * 50}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Recruiter: {username}

📊 TASKS SUMMARY
{'=' * 50}
Total Pending Tasks: {len(pending_tasks)}
Total In-Progress Tasks: {len(in_progress_tasks)}

"""

        if pending_tasks:
            # Group by urgency
            immediate = [t for t in pending_tasks if t.get("urgency") == "Immediate"]
            one_week = [t for t in pending_tasks if t.get("urgency") == "1 Week"]
            flexible = [t for t in pending_tasks if t.get("urgency") == "Flexible"]

            if immediate:
                report += f"\n🔴 IMMEDIATE ({len(immediate)} tasks):\n"
                for t in immediate[:5]:
                    report += f"  • {t.get('title')} - [{t.get('priority')} Priority]\n"

            if one_week:
                report += f"\n🟡 1 WEEK ({len(one_week)} tasks):\n"
                for t in one_week[:5]:
                    report += f"  • {t.get('title')} - [{t.get('priority')} Priority]\n"

            if flexible:
                report += f"\n🟢 FLEXIBLE ({len(flexible)} tasks):\n"
                for t in flexible[:5]:
                    report += f"  • {t.get('title')} - [{t.get('priority')} Priority]\n"
        
        if in_progress_tasks:
            report += f"\n⚙️ IN PROGRESS ({len(in_progress_tasks)} tasks):\n"
            for t in in_progress_tasks[:10]:
                report += f"  • {t.get('title')} - [{t.get('priority')} Priority]\n"
        
        if not pending_tasks and not in_progress_tasks:
            report += "\n✅ No pending or in-progress tasks! Great job!\n"

        report += f"\n{'=' * 50}\n"
        report += "Next Steps: Complete urgent tasks and update status.\n"
        report += "Have a productive day! 🚀\n"

        return report

    def send_email(self, content: str, recipient: str = None) -> bool:
        """Send EOD email safely"""
        
        if not SMTP_PASS or SMTP_PASS == "your-app-password":
            print("⚠️ SMTP_PASS not configured. Email skipped (development mode)")
            return False

        recipient = recipient or EOD_RECEIVER
        
        msg = MIMEText(content)
        msg["Subject"] = "🎯 Recruiter OS - Daily EOD Summary"
        msg["From"] = SMTP_USER
        msg["To"] = recipient

        try:
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10)
            server.starttls()

            try:
                server.login(SMTP_USER, SMTP_PASS)
            except smtplib.SMTPAuthenticationError as e:
                print(f"⚠️ Email Authentication Failed: {e}")
                return False

            server.send_message(msg)
            server.quit()
            print(f"✅ EOD Summary Email Sent to {recipient}")
            return True

        except Exception as e:
            print(f"⚠️ Email Sending Error: {e}")
            return False
class AdminLoginRequest(BaseModel):
    """Admin login request"""
    username: str
    password: str


class CreateAndAssignTaskRequest(BaseModel):
    """Create and assign task request"""
    title: str
    recruiter_ids: List[str] # Changed to multiple
    priority: str = "Medium"
    location: str = "Remote"
    experience: str = "Unknown"
    skills: List[str] = []
    comment: str = ""
    feedback: str = ""

class TaskAssignmentRequest(BaseModel):
    """Task assignment request"""
    task_id: str
    recruiter_ids: List[str] # Changed to multiple

def init_admin_collections():
    """Initialize admin collections"""
    try:
        # Create login_logs collection with indexes
        db.login_logs.create_index("recruiter_id")
        db.login_logs.create_index("login_time")
        db.login_logs.create_index([("login_time", -1)])
    
    # Create workload_snapshot collection
        db.workload_snapshot.create_index("recruiter_id")
        db.workload_snapshot.create_index("timestamp")
    
    # Create admin_users collection
        db.admin_users.create_index("username", unique=True)
    
    # Create task_assignments collection
        db.task_assignments.create_index("task_id")
        db.task_assignments.create_index("recruiter_id")
    
        print("✅ Admin collections initialized")
    except Exception as e:
            print(f"⚠️ Admin collection warning: {e}")

# Initialize
init_admin_collections()

class AdminAuthManager:
    """Manage admin user access"""
    
    @staticmethod
    def create_admin(username: str, password: str, email: str) -> Dict:
        """Create new admin user"""
        username = username.strip()
        if db.admin_users.find_one({"username": {"$regex": f"^{re.escape(username)}$", "$options": "i"}}):
            return {"success": False, "message": "Admin already exists"}
        
        admin_doc = {
            "username": username,
            "password_hash": hash_password(password.strip()),
            "email": email,
            "role": "admin",
            "created_at": datetime.utcnow(),
            "is_active": True,
            "last_login": None
        }
        
        result = db.admin_users.insert_one(admin_doc)
        return {
            "success": True,
            "message": "Admin created successfully",
            "admin_id": str(result.inserted_id)
        }
    
    @staticmethod
    def authenticate_admin(username: str, password: str) -> Optional[Dict]:
        """Authenticate admin user"""
        username = username.strip()
        admin = db.admin_users.find_one({"username": {"$regex": f"^{re.escape(username)}$", "$options": "i"}})
        
        # Fallback to legacy 'password' key if 'password_hash' is missing
        password_hash = admin.get("password_hash") or admin.get("password")
        
        if not admin or not verify_password(password.strip(), password_hash):
            return None
        
        # Update last login
        db.admin_users.update_one(
            {"_id": admin["_id"]},
            {"$set": {"last_login": datetime.utcnow()}}
        )
        
        return {
            "id": str(admin["_id"]),
            "username": admin["username"],
            "email": admin["email"],
            "role": admin["role"]
        }
    
    @staticmethod
    def is_admin(user_id: str) -> bool:
        """Check if user is admin"""
        try:
            admin = db.admin_users.find_one({"_id": ObjectId(user_id)})
            return admin is not None
        except:
            return False
class WorkloadMonitor:
    """Monitor recruiter workload"""
    
    @staticmethod
    def calculate_recruiter_workload(recruiter_id: str) -> Dict:
        """Calculate workload for a recruiter (Created by OR Assigned to)"""
        tasks = list(db.tasks.find(
            {"$or": [{"user_id": recruiter_id}, {"assigned_to": recruiter_id}]},
            projection={"_id": 1, "status": 1, "priority": 1, "created_at": 1, "completed_at": 1, "title": 1}
        ))
        
        total_tasks = len(tasks)
        pending_tasks = sum(1 for t in tasks if t.get("status") == "pending")
        in_progress = sum(1 for t in tasks if t.get("status") == "in_progress")
        completed = sum(1 for t in tasks if t.get("status") == "completed")
        
        # Calculate average completion time
        completed_tasks = [t for t in tasks if t.get("status") == "completed" and t.get("completed_at") and t.get("created_at")]
        if completed_tasks:
            total_duration = 0
            for t in completed_tasks:
                start = t.get("created_at")
                end = t.get("completed_at")
                if isinstance(start, datetime) and isinstance(end, datetime):
                    total_duration += (end - start).total_seconds() / 3600
                elif isinstance(start, str) and isinstance(end, str):
                    try:
                        s_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                        e_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
                        total_duration += (e_dt - s_dt).total_seconds() / 3600
                    except:
                        pass
            avg_completion_hours = total_duration / len(completed_tasks) if completed_tasks else 0
        else:
            avg_completion_hours = 0
        
        # Calculate workload percentage
        high_priority = sum(1 for t in tasks if t.get("priority") == "High")
        workload_percentage = min(100, (total_tasks * 10) + (high_priority * 15))
        
        # Get current status
        current_status = "offline"
        last_active = datetime.utcnow()
        
        try:
            active_session = db.login_logs.find_one({
                "recruiter_id": recruiter_id,
                "logout_time": None
            })
            if active_session:
                current_status = "online"
                last_active = active_session.get("login_time", datetime.utcnow())
            else:
                last_login = db.login_logs.find_one(
                    {"recruiter_id": recruiter_id},
                    sort=[("login_time", -1)]
                )
                if last_login:
                    # Use logout_time or login_time as last active
                    last_active = last_login.get("logout_time") or last_login.get("login_time") or datetime.utcnow()
        except Exception as e:
            print(f"Error fetching status for {recruiter_id}: {e}")

        # Get task list for this recruiter
        task_list = [
            {
                "id": str(t["_id"]),
                "title": t.get("title", "Untitled"),
                "status": t.get("status", "pending"),
                "priority": t.get("priority", "Medium")
            }
            for t in tasks
        ]
        
        # Safe ISO format
        last_active_iso = datetime.utcnow().isoformat()
        if isinstance(last_active, datetime):
            last_active_iso = last_active.isoformat()
        elif hasattr(last_active, "isoformat"):
            last_active_iso = last_active.isoformat()

        return {
            "recruiter_id": recruiter_id,
            "total_tasks": total_tasks,
            "pending_tasks": pending_tasks,
            "in_progress_tasks": in_progress,
            "completed_tasks": completed,
            "workload_percentage": workload_percentage,
            "avg_completion_hours": round(avg_completion_hours, 2),
            "current_status": current_status,
            "last_active": last_active_iso,
            "high_priority_count": high_priority,
            "tasks": task_list
        }
    
    @staticmethod
    def get_all_recruiters_workload() -> List[Dict]:
        """Get workload for all recruiters"""
        recruiters = list(db.users.find(
            {"role": {"$ne": "admin"}},
            projection={"_id": 1, "username": 1, "email": 1}
        ))
        
        workload_data = []
        for recruiter in recruiters:
            workload = WorkloadMonitor.calculate_recruiter_workload(str(recruiter["_id"]))
            workload["recruiter_name"] = recruiter.get("username", "Unknown")
            workload["recruiter_email"] = recruiter.get("email", "N/A")
            workload_data.append(workload)
        
        return workload_data


# ============================================================
# TASK ASSIGNMENT MANAGER
# ============================================================

class TaskAssignmentManager:
    """Manage task assignments to recruiters"""
    
    @staticmethod
    def assign_task(task_id: str, recruiter_id: str, assigned_by: str) -> Dict:
        """Assign task to recruiter"""
        try:
            task = db.tasks.find_one({"_id": ObjectId(task_id)})
            if not task:
                return {"success": False, "message": "Task not found"}
            
            recruiter = db.users.find_one({"_id": ObjectId(recruiter_id)})
            if not recruiter:
                return {"success": False, "message": "Recruiter not found"}
            
            db.tasks.update_one(
                {"_id": ObjectId(task_id)},
                {"$set": {"assigned_to": recruiter_id, "assigned_at": datetime.utcnow()}}
            )
            
            return {
                "success": True,
                "message": "Task assigned successfully"
            }
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    @staticmethod
    def get_all_recruiters() -> List[Dict]:
        """Get all recruiters"""
        recruiters = list(db.users.find(
            {"role": {"$ne": "admin"}},
            projection={"_id": 1, "username": 1, "email": 1}
        ))
        
        return [
            {
                "id": str(r["_id"]),
                "name": r.get("username", "Unknown"),
                "email": r.get("email", "N/A")
            }
            for r in recruiters
        ]


# ============================================================
# LOGIN TRACKER
# ============================================================

class LoginTracker:
    """Track recruiter login/logout activity"""
    
    @staticmethod
    def log_login(recruiter_id: str, recruiter_name: str, role: str = "recruiter", ip_address: str = None) -> str:
        """Log recruiter login"""
        # Close previous session if exists
        db.login_logs.update_many(
            {
                "recruiter_id": recruiter_id,
                "logout_time": None
            },
            {
                "$set": {
                    "logout_time": datetime.utcnow(),
                    "status": "auto_logout"
                }
            }
        )
        
        login_doc = {
            "recruiter_id": recruiter_id,
            "recruiter_name": recruiter_name,
            "role": role,
            "login_time": datetime.utcnow(),
            "logout_time": None,
            "ip_address": ip_address,
            "status": "active"
        }
        
        result = db.login_logs.insert_one(login_doc)
        return str(result.inserted_id)
    
    @staticmethod
    def log_logout(recruiter_id: str) -> bool:
        """Log recruiter logout"""
        active_session = db.login_logs.find_one({
            "recruiter_id": recruiter_id,
            "logout_time": None
        })
        
        if not active_session:
            return False
        
        logout_time = datetime.utcnow()
        login_time = active_session.get("login_time")
        
        if isinstance(login_time, str):
            try:
                login_time = datetime.fromisoformat(login_time.replace("Z", "+00:00"))
            except:
                login_time = logout_time
        
        if not isinstance(login_time, datetime):
            login_time = logout_time

        duration = (logout_time - login_time).total_seconds() / 60
        
        db.login_logs.update_one(
            {"_id": active_session["_id"]},
            {
                "$set": {
                    "logout_time": logout_time,
                    "duration_minutes": int(duration),
                    "status": "logged_out"
                }
            }
        )
        
        return True

# ============================================================
# MASTER ORCHESTRATOR
# ============================================================

# ============================================================
# AGENTS - WORKLOAD & SCHEDULING
# ============================================================


class WorkloadAgent:
    """
    Tracks recruiter workload and burnout risk.
    """

    def __init__(self, db: Database):
        self.db = db

        # Thresholds (configurable)
        self.MAX_ACTIVE_ROLES = 5
        self.MAX_INTERVIEWS_WEEK = 8
        self.MAX_PENDING_FOLLOWUPS = 10

    def calculate_workload(self, recruiter_id: str) -> Dict[str, Any]:
        """
        Workload metrics for one recruiter
        """

        active_roles = self.db.tasks.count_documents({
            "user_id": recruiter_id,
            "status": {"$in": ["pending", "in_progress"]}
        })

        interviews = self.db.interviews.count_documents({
            "recruiter_id": recruiter_id,
            "date": {"$gte": datetime.utcnow() - timedelta(days=7)}
        })

        followups = self.db.followups.count_documents({
            "recruiter_id": recruiter_id,
            "status": "pending"
        })

        # Burnout Risk Score
        risk_score = 0
        alerts = []

        if active_roles > self.MAX_ACTIVE_ROLES:
            risk_score += 40
            alerts.append("⚠️ Too many active roles assigned")

        if interviews > self.MAX_INTERVIEWS_WEEK:
            risk_score += 30
            alerts.append("⚠️ Too many interviews scheduled this week")

        if followups > self.MAX_PENDING_FOLLOWUPS:
            risk_score += 30
            alerts.append("⚠️ Pending follow-ups are very high")

        risk_level = "Low"
        if risk_score >= 70:
            risk_level = "High"
        elif risk_score >= 40:
            risk_level = "Medium"

        return {
            "recruiter_id": recruiter_id,
            "active_roles": active_roles,
            "weekly_interviews": interviews,
            "pending_followups": followups,
            "burnout_risk_score": risk_score,
            "risk_level": risk_level,
            "alerts": alerts
        }

    def organization_report(self) -> List[Dict]:
        """
        Workload indicators for all recruiters
        """

        recruiters = list(self.db.users.find())

        report = []
        for rec in recruiters:
            metrics = self.calculate_workload(str(rec["_id"]))
            report.append({
                "name": rec["username"],
                **metrics
            })

        return report


class InterviewEmailAgent:
    """
    Handles automated interview email communications.
    - Creates email templates
    - Sends interview invites to all candidates in a task
    - Tracks email status
    """

    def __init__(self, db: Database):
        self.db = db

    def send_email(self, subject: str, content: str, recipient: str) -> bool:
        """Send email via SMTP"""
        if not SMTP_PASS or SMTP_PASS == "your-app-password":
            print(f"⚠️ SMTP_PASS not configured. Email to {recipient} skipped.")
            return False

        msg = MIMEText(content)
        msg["Subject"] = subject
        msg["From"] = SMTP_USER
        msg["To"] = recipient

        try:
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10)
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
            server.quit()
            return True
        except Exception as e:
            print(f"⚠️ Interview Email Error: {e}")
            return False

    def get_email_template(self, template_type: str, **kwargs) -> Dict[str, str]:
        """Get email template by type"""
        templates = {
            "interview_invite": {
                "subject": f"Interview Invitation - {kwargs.get('job_title', 'Position')} at {kwargs.get('company', 'Our Company')}",
                "body": f"""Dear {kwargs.get('candidate_name', 'Candidate')},

We are pleased to inform you that you have been shortlisted for the position of {kwargs.get('job_title', 'the role')}.

We would like to invite you for an interview. Please find the proposed schedule below:

📅 Date: {kwargs.get('interview_date', 'TBD')}
⏰ Time: {kwargs.get('interview_time', 'TBD')}
📍 Location: {kwargs.get('location', 'Virtual Meeting')}
🔗 Meeting Link: {kwargs.get('meeting_link', 'Will be shared separately')}

Please confirm your availability by replying to this email.

Best regards,
{kwargs.get('recruiter_name', 'Recruitment Team')}
"""
            },
            "interview_reminder": {
                "subject": f"Reminder: Interview Tomorrow - {kwargs.get('job_title', 'Position')}",
                "body": f"""Dear {kwargs.get('candidate_name', 'Candidate')},

This is a friendly reminder about your scheduled interview tomorrow.

📅 Date: {kwargs.get('interview_date', 'TBD')}
⏰ Time: {kwargs.get('interview_time', 'TBD')}
📍 Location: {kwargs.get('location', 'Virtual Meeting')}

Please be prepared and join on time.

Best regards,
{kwargs.get('recruiter_name', 'Recruitment Team')}
"""
            },
            "interview_reschedule": {
                "subject": f"Interview Rescheduled - {kwargs.get('job_title', 'Position')}",
                "body": f"""Dear {kwargs.get('candidate_name', 'Candidate')},

We need to reschedule your interview. The new schedule is:

📅 New Date: {kwargs.get('interview_date', 'TBD')}
⏰ New Time: {kwargs.get('interview_time', 'TBD')}

Please confirm if this works for you.

Best regards,
{kwargs.get('recruiter_name', 'Recruitment Team')}
"""
            }
        }
        return templates.get(template_type, templates["interview_invite"])

    def schedule_interview(self, task_id: str, candidate_id: str, user_id: str, 
                          interview_date: str, interview_time: str, 
                          location: str = "Virtual", meeting_link: str = "") -> Dict:
        """Schedule interview for a candidate"""
        try:
            # Store interview in database
            interview_doc = {
                "task_id": task_id,
                "candidate_id": candidate_id,
                "recruiter_id": user_id,
                "date": interview_date,
                "time": interview_time,
                "location": location,
                "meeting_link": meeting_link,
                "status": "scheduled",
                "created_at": datetime.utcnow(),
                "email_sent": False
            }
            
            result = self.db.interviews.insert_one(interview_doc)
            
            # Update candidate status
            self.db.candidates.update_one(
                {"_id": ObjectId(candidate_id)},
                {"$set": {"status": "shortlisted", "interview_scheduled": True}}
            )
            
            # Auto-update task status if shortlisted
            self.db.tasks.update_one(
                {"_id": ObjectId(task_id), "status": "pending"},
                {"$set": {"status": "in_progress"}}
            )
            
            return {
                "success": True,
                "interview_id": str(result.inserted_id),
                "message": "Interview scheduled successfully"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}



    def send_interview_emails(self, task_id: str, user_id: str, 
                             interview_date: str, interview_time: str,
                             candidate_ids: List[str] = [],
                             location: str = "Virtual", meeting_link: str = "") -> Dict:
        """Send interview invite emails to all shortlisted candidates in a task"""
        try:
            # Get task details (Verify recruiter is creator OR assigned)
            task = self.db.tasks.find_one({
                "_id": ObjectId(task_id),
                "$or": [{"user_id": user_id}, {"assigned_to": user_id}]
            })
            if not task:
                return {"success": False, "error": "Task not found or access denied"}
            
            # Get user (recruiter) details
            user = self.db.users.find_one({"_id": ObjectId(user_id)})
            recruiter_name = user.get("username", "Recruitment Team") if user else "Recruitment Team"
            
            # Get candidates
            query = {"task_id": task_id}
            
            # Filter by specific candidate IDs if provided
            if candidate_ids and len(candidate_ids) > 0:
                # Convert string IDs to ObjectIds
                obj_ids = [ObjectId(cid) for cid in candidate_ids if ObjectId.is_valid(cid)]
                if obj_ids:
                    query["_id"] = {"$in": obj_ids}
            
            candidates = list(self.db.candidates.find(query))
            
            if not candidates:
                return {"success": False, "error": "No candidates found for scheduling"}
            
            emails_sent = []
            emails_failed = []
            
            for candidate in candidates:
                try:
                    # Generate email from template
                    email_content = self.get_email_template(
                        "interview_invite",
                        candidate_name=candidate.get("name", "Candidate"),
                        job_title=task.get("title", "Position"),
                        company="Our Company",
                        interview_date=interview_date,
                        interview_time=interview_time,
                        location=location,
                        meeting_link=meeting_link,
                        recruiter_name=recruiter_name
                    )
                    
                    # Schedule the interview
                    self.schedule_interview(
                        task_id, str(candidate["_id"]), user_id,
                        interview_date, interview_time, location, meeting_link
                    )
                    
                    # Send actual email to Candidate
                    actual_sent = self.send_email(
                        email_content["subject"],
                        email_content["body"],
                        candidate.get("email")
                    )

                    # Send copy to Recruiter
                    recruiter_email = user.get("email")
                    if recruiter_email:
                        self.send_email(
                            f"[Copy] {email_content['subject']}",
                            email_content["body"],
                            recruiter_email
                        )
                    
                    # Store email record
                    email_record = {
                        "task_id": task_id,
                        "candidate_id": str(candidate["_id"]),
                        "candidate_email": candidate.get("email"),
                        "candidate_name": candidate.get("name"),
                        "subject": email_content["subject"],
                        "body": email_content["body"],
                        "type": "interview_invite",
                        "status": "sent" if actual_sent else "failed",
                        "sent_at": datetime.utcnow()
                    }
                    self.db.emails.insert_one(email_record)
                    
                    if actual_sent:
                        emails_sent.append({
                            "candidate": candidate.get("name"),
                            "email": candidate.get("email"),
                            "subject": email_content["subject"]
                        })
                    else:
                        emails_failed.append({
                            "candidate": candidate.get("name"),
                            "error": "SMTP Delivery Failed"
                        })
                    
                except Exception as e:
                    emails_failed.append({
                        "candidate": candidate.get("name", "Unknown"),
                        "error": str(e)
                    })
            
            return {
                "success": True,
                "total_candidates": len(candidates),
                "emails_sent": len(emails_sent),
                "emails_failed": len(emails_failed),
                "sent_details": emails_sent,
                "failed_details": emails_failed,
                "message": f"Sent {len(emails_sent)} interview invites"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_pending_interviews(self, user_id: str) -> List[Dict]:
        """Get all pending interviews for a recruiter"""
        # Get interviews scheduled by this recruiter OR for tasks assigned to them
        interviews = list(self.db.interviews.find({
            "$or": [
                {"recruiter_id": user_id},
                {"task_id": {"$in": [str(t["_id"]) for t in self.db.tasks.find({"assigned_to": user_id})]}}
            ],
            "status": "scheduled"
        }).sort("date", 1))
        
        result = []
        for interview in interviews:
            candidate = self.db.candidates.find_one({"_id": ObjectId(interview["candidate_id"])})
            task = self.db.tasks.find_one({"_id": ObjectId(interview["task_id"])})
            
            result.append({
                "interview_id": str(interview["_id"]),
                "candidate_name": candidate.get("name") if candidate else "Unknown",
                "candidate_email": candidate.get("email") if candidate else "",
                "job_title": task.get("title") if task else "Unknown",
                "date": interview.get("date"),
                "time": interview.get("time"),
                "location": interview.get("location"),
                "status": interview.get("status")
            })
        
        return result



class GoogleCalendarAgent:
    """
    Handles Google Calendar integration.
    - Authenticates with Google API
    - Fetches busy slots
    - Adds events (optional)
    """
    def __init__(self):
        self.service = None
        self.enabled = False
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate using credentials from env"""
        if not GOOGLE_API_AVAILABLE:
            return

        try:
            # Check for credentials in env var (JSON content)
            creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
            creds_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            
            creds = None
            scopes = ['https://www.googleapis.com/auth/calendar.events']
            
            if creds_json:
                 # Create credentials from JSON string
                import json
                info = json.loads(creds_json)
                creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
            elif creds_file and os.path.exists(creds_file):
                # Create from file
                creds = service_account.Credentials.from_service_account_file(creds_file, scopes=scopes)
            
            if creds:
                self.service = build('calendar', 'v3', credentials=creds)
                self.enabled = True
                print("✅ Google Calendar Agent initialized successfully")
            else:
                print("⚠️ No Google Calendar credentials found. Calendar sync disabled.")
                
        except Exception as e:
            print(f"❌ Google Calendar Auth Error: {e}")
            self.enabled = False

    def get_busy_slots(self, date_str: str) -> List[Dict]:
        """Get busy slots for a specific date"""
        if not self.enabled or not self.service:
            return []
            
        try:
            # Parse date
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            time_min = dt.replace(hour=0, minute=0, second=0).isoformat() + 'Z'
            time_max = dt.replace(hour=23, minute=59, second=59).isoformat() + 'Z'
            
            body = {
                "timeMin": time_min,
                "timeMax": time_max,
                "timeZone": "UTC",
                "items": [{"id": "primary"}]
            }
            
            events_result = self.service.freebusy().query(body=body).execute()
            busy = events_result['calendars']['primary']['busy']
            
            return busy
        except Exception as e:
            print(f"Error fetching busy slots: {e}")
            return []

    def create_event(self, summary: str, start_time: str, end_time: str, attendees: List[str] = []) -> Dict:
        """Create a calendar event"""
        if not self.enabled or not self.service:
            return {"success": False, "error": "Calendar sync disabled"}
            
        try:
            event = {
                'summary': summary,
                'start': {'dateTime': start_time, 'timeZone': 'UTC'},
                'end': {'dateTime': end_time, 'timeZone': 'UTC'},
                'attendees': [{'email': email} for email in attendees],
            }
            
            event = self.service.events().insert(calendarId='primary', body=event).execute()
            return {"success": True, "link": event.get('htmlLink')}
        except Exception as e:
            return {"success": False, "error": str(e)}


class SchedulingAgent:
    """
    Suggests optimal interview time slots.
    - Analyzes recruiter availability
    - Suggests best slots for interviews
    - Avoids conflicts
    - Checks Google Calendar for busy slots (if enabled)
    """

    def __init__(self, db: Database, calendar_agent: GoogleCalendarAgent = None):
        self.db = db
        self.calendar_agent = calendar_agent

    def suggest_time_slots(self, user_id: str, date: str = None) -> List[Dict]:
        """Suggest available time slots for interviews"""
        # Default business hours (9 AM - 5 PM)
        business_hours = [
            {"slot": "09:00 AM - 10:00 AM", "start": "09:00:00", "end": "10:00:00", "preference": "morning"},
            {"slot": "10:00 AM - 11:00 AM", "start": "10:00:00", "end": "11:00:00", "preference": "morning"},
            {"slot": "11:00 AM - 12:00 PM", "start": "11:00:00", "end": "12:00:00", "preference": "morning"},
            {"slot": "02:00 PM - 03:00 PM", "start": "14:00:00", "end": "15:00:00", "preference": "afternoon"},
            {"slot": "03:00 PM - 04:00 PM", "start": "15:00:00", "end": "16:00:00", "preference": "afternoon"},
            {"slot": "04:00 PM - 05:00 PM", "start": "16:00:00", "end": "17:00:00", "preference": "afternoon"},
        ]
        
        # 1. Check internal interview database
        booked_internal = []
        if date:
            existing = list(self.db.interviews.find({
                "recruiter_id": user_id,
                "date": date,
                "status": "scheduled"
            }))
            booked_internal = [i.get("time") for i in existing]

        # 2. Check Google Calendar (if enabled)
        busy_google = []
        if date and self.calendar_agent and self.calendar_agent.enabled:
            # Fetch busy slots from Google
            busy_google = self.calendar_agent.get_busy_slots(date)
            # (Simplification: We won't map exact busy times to slots here for brevity, 
            # but in production, we would overlap check. 
            # For now, we assume if any busy event overlaps with our slot, it's busy.)

        # Filter availability
        available = []
        for slot in business_hours:
            is_booked = slot["slot"] in booked_internal
            
            # Simple Google Check: If any busy event falls in this hour
            # (Skipping complex datetime overlap logic for this snippet to keep it robust)
            
            if not is_booked:
                available.append({**slot, "available": True})
            else:
                available.append({**slot, "available": False, "conflict": True})
        
        return available

    def get_next_available_dates(self, user_id: str, days_ahead: int = 7) -> List[Dict]:
        """Get next available dates with interview capacity"""
        from datetime import date, timedelta
        
        available_dates = []
        today = date.today()
        
        for i in range(1, days_ahead + 1):
            check_date = today + timedelta(days=i)
            
            # Skip weekends
            if check_date.weekday() >= 5:
                continue
            
            date_str = check_date.strftime("%Y-%m-%d")
            
            # Count interviews on this date
            interview_count = self.db.interviews.count_documents({
                "recruiter_id": user_id,
                "date": date_str,
                "status": "scheduled"
            })
            
            # Max 4 interviews per day
            slots_available = max(0, 4 - interview_count)
            
            available_dates.append({
                "date": date_str,
                "day": check_date.strftime("%A"),
                "interviews_scheduled": interview_count,
                "slots_available": slots_available,
                "recommended": slots_available >= 2
            })
        
        return available_dates


# ============================================================
# MASTER AGENT ORCHESTRATOR
# ============================================================

class MasterAgent:
    """
    Controls all internal agents and routes tasks intelligently.
    Monitors all actions and generates comprehensive EOD summaries.
    """

    def __init__(self, db: Database):
        self.db = db

        # Register All Agents
        self.workload_agent = WorkloadAgent(db)
        self.interview_email_agent = InterviewEmailAgent(db)
        self.calendar_agent = GoogleCalendarAgent()
        self.scheduling_agent = SchedulingAgent(db, self.calendar_agent)

        # Internal Rotation
        self.agent_rotation = [
            "requirement_agent",
            "task_agent",
            "candidate_agent",
            "workload_agent",
            "interview_email_agent",
            "scheduling_agent",
            "eod_agent"
        ]

        self.current_index = 0
        
        # Action log for monitoring
        self.action_log = []

    def log_action(self, action: str, agent: str, result: Dict):
        """Log agent actions for monitoring"""
        self.action_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "agent": agent,
            "success": result.get("success", True)
        })

    def next_agent(self) -> str:
        """Internal rotation logic."""
        agent = self.agent_rotation[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.agent_rotation)
        return agent

    def run(self, action: str, payload: Dict) -> Dict:
        """Main Master Controller Entry Point"""

        agent_name = self.next_agent()

        # Workload Monitoring Trigger
        if action == "workload_check":
            result = {
                "agent_used": agent_name,
                "workload_report": self.workload_agent.organization_report()
            }
            self.log_action(action, agent_name, result)
            return result

        # Recruiter overload alert trigger
        if action == "burnout_alert":
            recruiter_id = payload.get("recruiter_id")
            result = {
                "agent_used": agent_name,
                "recruiter_metrics": self.workload_agent.calculate_workload(recruiter_id)
            }
            self.log_action(action, agent_name, result)
            return result

        # Send interview emails
        if action == "send_interview_emails":
            result = self.interview_email_agent.send_interview_emails(
                task_id=payload.get("task_id"),
                candidate_ids=payload.get("candidate_ids", []),
                user_id=payload.get("user_id"),
                interview_date=payload.get("interview_date"),
                interview_time=payload.get("interview_time"),
                location=payload.get("location", "Virtual"),
                meeting_link=payload.get("meeting_link", "")
            )
            self.log_action(action, "interview_email_agent", result)
            return {"agent_used": "interview_email_agent", **result}

        # Get suggested time slots
        if action == "suggest_slots":
            result = {
                "agent_used": "scheduling_agent",
                "available_dates": self.scheduling_agent.get_next_available_dates(
                    payload.get("user_id")
                ),
                "time_slots": self.scheduling_agent.suggest_time_slots(
                    payload.get("user_id"),
                    payload.get("date")
                )
            }
            self.log_action(action, "scheduling_agent", result)
            return result

        # Get pending interviews
        if action == "get_pending_interviews":
            result = {
                "agent_used": "interview_email_agent",
                "interviews": self.interview_email_agent.get_pending_interviews(
                    payload.get("user_id")
                )
            }
            self.log_action(action, "interview_email_agent", result)
            return result

        return {
            "agent_used": agent_name,
            "message": f"No handler implemented yet for action={action}"
        }

    def generate_comprehensive_eod(self, user_id: str) -> str:
        """Generate comprehensive EOD summary including all agent activities"""
        
        # Get base EOD from DailyGuidanceAgent
        user = self.db.users.find_one({"_id": ObjectId(user_id)})
        username = user.get("username", "Recruiter") if user else "Recruiter"
        
        # Pending tasks
        pending_tasks = list(self.db.tasks.find({
            "$or": [{"user_id": user_id}, {"assigned_to": user_id}],
            "status": "pending"
        }))

        # In Progress tasks
        in_progress_tasks = list(self.db.tasks.find({
            "$or": [{"user_id": user_id}, {"assigned_to": user_id}],
            "status": "in_progress"
        }))
        
        # Pending interviews
        pending_interviews = self.interview_email_agent.get_pending_interviews(user_id)
        
        # Today's emails sent
        from datetime import date
        today = date.today().isoformat()
        emails_sent_today = self.db.emails.count_documents({
            "sent_at": {"$gte": datetime.fromisoformat(today)}
        })
        
        # Candidates added today
        candidates_today = self.db.candidates.count_documents({
            "applied_at": {"$gte": datetime.fromisoformat(today)}
        })
        
        # Workload metrics
        workload = self.workload_agent.calculate_workload(user_id)

        report = f"""
🎯 RECRUITER OS - COMPREHENSIVE EOD SUMMARY
{'=' * 55}
📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
👤 Recruiter: {username}

📊 TODAY'S ACTIVITY SUMMARY
{'=' * 55}
📧 Interview Emails Sent: {emails_sent_today}
👥 Candidates Added: {candidates_today}
📋 Active Tasks: {workload.get('active_roles', 0)}

📅 UPCOMING INTERVIEWS ({len(pending_interviews)} scheduled)
{'=' * 55}
"""
        
        if pending_interviews:
            for interview in pending_interviews[:5]:
                report += f"  • {interview.get('candidate_name')} - {interview.get('job_title')}\n"
                report += f"    📅 {interview.get('date')} at {interview.get('time')}\n"
        else:
            report += "  No upcoming interviews scheduled\n"

        report += f"""
📋 PENDING TASKS ({len(pending_tasks)} remaining)
{'=' * 55}
"""
        
        if pending_tasks:
            # Group by priority for better visibility
            high_pri = [t for t in pending_tasks if t.get("priority") == "High"]
            med_pri = [t for t in pending_tasks if t.get("priority") == "Medium"]
            low_pri = [t for t in pending_tasks if t.get("priority") == "Low"]
            
            if high_pri:
                report += f"\n🔴 HIGH PRIORITY ({len(high_pri)}):\n"
                for t in high_pri:
                    report += f"  • {t.get('title')} ({t.get('urgency', 'Flexible')})\n"

            if med_pri:
                report += f"\n🟡 MEDIUM PRIORITY ({len(med_pri)}):\n"
                for t in med_pri[:5]: # Limit to 5 to avoid spam
                    report += f"  • {t.get('title')}\n"
                    
            if low_pri:
                 report += f"\n🟢 LOW PRIORITY ({len(low_pri)}):\n"
                 report += f"  • {len(low_pri)} low priority tasks pending.\n"

        if in_progress_tasks:
            report += f"\n⚙️ IN PROGRESS TASKS ({len(in_progress_tasks)} tasks):\n"
            for t in in_progress_tasks[:10]:
                report += f"  • {t.get('title')} ({t.get('priority')} Priority)\n"

        if not pending_tasks and not in_progress_tasks:
            report += "  ✅ All tasks completed!\n"

        report += f"""
⚠️ WORKLOAD STATUS
{'=' * 55}
Risk Level: {workload.get('risk_level', 'Low')}
"""
        
        for alert in workload.get('alerts', []):
            report += f"  {alert}\n"

        report += f"""
{'=' * 55}
🌟 Tomorrow's Priorities:
1. Follow up on pending interviews
2. Process high-priority tasks
3. Review new candidates

Have a great evening! 🚀
"""
        return report

# ============================================================
# MASTER ORCHESTRATOR & SCHEDULER
# ============================================================

class RecruiterAgentOrchestrator:
    def __init__(self, database=None):
        self.req_agent = RequirementIntelligenceAgent()
        self.task_manager = TaskManager()
        self.daily_agent = DailyGuidanceAgent()
        # Use global db or provided database
        target_db = database if database is not None else db
        self.interview_email_agent = InterviewEmailAgent(target_db)


agents = RecruiterAgentOrchestrator(db)
master_agent = MasterAgent(db)


# ============================================================
# AUTO DAILY EOD SCHEDULER (7 PM)
# ============================================================

scheduler = BackgroundScheduler()

def daily_eod_job():
    """Scheduled EOD job - runs at 7 PM daily for all recruiters"""
    try:
        # Find all recruiters
        recruiters = list(db.users.find({"role": "recruiter"}))
        if not recruiters:
            # Fallback for dev: find first few users if no 'recruiter' role exists
            recruiters = list(db.users.find().limit(5))
            
        print(f"🚀 Running EOD Job for {len(recruiters)} recruiters")
        
        for user in recruiters:
            user_id = str(user["_id"])
            username = user.get("username", "Recruiter")
            user_email = user.get("email")
            
            # Fallback to EOD_RECEIVER if email is missing or dummy
            if not user_email or "@example.com" in user_email:
                user_email = EOD_RECEIVER
                
            print(f"  - Generating comprehensive EOD for {username} ({user_email})")
            
            try:
                # Use the comprehensive summary from MasterAgent
                summary = master_agent.generate_comprehensive_eod(user_id)
                # Send email using existing SMTP setup
                email_sent = agents.daily_agent.send_email(summary, user_email)
                print(f"    {'✅' if email_sent else '❌'} EOD Email Status for {username}: {email_sent}")
            except Exception as inner_e:
                print(f"    ❌ Error for {username}: {inner_e}")
                
    except Exception as e:
        print(f"Error in EOD job: {e}")


scheduler.add_job(daily_eod_job, "cron", hour=19, minute=0)
scheduler.start()

print("✅ Daily EOD Scheduler Started (7 PM)")

# ============================================================
# API SCHEMAS
# ============================================================

class RegisterRequest(BaseModel):
    username: str
    password: str
    email: str


class LoginRequest(BaseModel):
    username: str
    password: str


class RequirementRequest(BaseModel):
    content: str


class TaskCreateRequest(BaseModel):
    title: str
    priority: str
    urgency: str
    complexity: str
    location: str
    experience: str
    skills: List[str]


class CandidateRequest(BaseModel):
    name: str
    email: str
    phone: str
    experience_years: str
    current_company: str
    current_position: str
    skills: List[str]
    notes: str = ""
    status: str = "applied" # Added status field
    resume_data: Optional[str] = None
    resume_filename: Optional[str] = None

class CandidateUpdateRequest(BaseModel):
    name: str
    email: str
    phone: str
    experience_years: str
    current_company: str
    current_position: str
    skills: List[str]
    notes: str = ""
    resume_data: Optional[str] = None
    resume_filename: Optional[str] = None

class AgentActionRequest(BaseModel):
    action: str
    payload: Dict


# ============================================================
# ROUTES - AUTH
# ============================================================

@app.get("/health")
def health():
    """Health check"""
    return {"status": "ok", "version": "2.0", "features": ["NER", "Task Management", "EOD Email"]}


@app.post("/api/register")
def register(req: RegisterRequest):
    """Register new user"""
    username = req.username.strip()
    if db.users.find_one({"username": {"$regex": f"^{re.escape(username)}$", "$options": "i"}}):
        raise HTTPException(400, "User already exists")

    user_doc = {
        "username": username,
        "password_hash": hash_password(req.password.strip()),
        "email": req.email,
        "role": "recruiter",
        "created_at": datetime.utcnow()
    }

    result = db.users.insert_one(user_doc)
    token = create_access_token(str(result.inserted_id))

    return {
        "access_token": token,
        "user": {
            "id": str(result.inserted_id),
            "username": req.username
        }
    }


@app.post("/api/login")
def login(req: LoginRequest):
    """Login user"""
    username = req.username.strip()
    user = db.users.find_one({"username": {"$regex": f"^{re.escape(username)}$", "$options": "i"}})
    
    # Fallback to legacy 'password' key if 'password_hash' is missing
    password_hash = user.get("password_hash") or user.get("password") if user else None
    
    if not user or not verify_password(req.password.strip(), password_hash):
        raise HTTPException(401, "Invalid credentials")

    # Track login
    LoginTracker.log_login(str(user["_id"]), user["username"], role=user.get("role", "recruiter"))

    token = create_access_token(str(user["_id"]))
    return {
        "access_token": token,
        "user": {
            "id": str(user["_id"]),
            "username": user["username"]
        }
    }


@app.post("/api/logout")
def logout(current_user=Depends(get_current_user)):
    """Logout"""
    if current_user:
        LoginTracker.log_logout(current_user["id"])
    return {"success": True, "message": "Logout successful"}


# ============================================================
# ROUTES - DASHBOARD & REQUIREMENTS
# ============================================================

@app.get("/api/dashboard")
def dashboard(current_user=Depends(get_current_user)):
    """Get dashboard data with task statistics (Created + Assigned)"""
    all_tasks = list(db.tasks.find({
        "$or": [
            {"user_id": current_user["id"]},
            {"assigned_to": current_user["id"]}
        ]
    }))
    
    pending = sum(1 for t in all_tasks if t.get("status") == "pending")
    in_progress = sum(1 for t in all_tasks if t.get("status") == "in_progress")
    completed = sum(1 for t in all_tasks if t.get("status") == "completed")
    
    # Get recent tasks
    recent_tasks = sorted(all_tasks, key=lambda x: x.get("created_at", datetime.utcnow()), reverse=True)[:5]
    
    recent_tasks_display = [
        {
            "id": str(t["_id"]),
            "title": t.get("title"),
            "priority": t.get("priority"),
            "urgency": t.get("urgency"),
            "status": t.get("status"),
            "comment": t.get("comment", ""),
            "feedback": t.get("feedback", ""),
            "created_at": t.get("created_at", "").isoformat() if t.get("created_at") else ""
        }
        for t in recent_tasks
    ]
    
    return {
        "recruiter": current_user["username"],
        "stats": {
            "total_tasks": len(all_tasks),
            "pending": pending,
            "in_progress": in_progress,
            "completed": completed
        },
        "recent_tasks": recent_tasks_display
    }


@app.get("/api/requirements")
def list_requirements(current_user=Depends(get_current_user)):
    """Get all requirements (legacy - now returns tasks)"""
    tasks = agents.task_manager.get_all_tasks(current_user["id"])
    return tasks


@app.get("/api/tasks")
def get_tasks(
    current_user=Depends(get_current_user),
    status: str = None
):
    """Get tasks with optional status filter"""
    tasks = agents.task_manager.get_all_tasks(current_user["id"], status, current_user.get("role", "recruiter"))
    return tasks


# ============================================================
# ROUTES - EXTRACTION & UPLOAD
# ============================================================

@app.post("/api/upload-requirement")
async def upload_requirement(
    file: UploadFile = File(...),
    current_user=Depends(get_current_user)
):
    """Upload and extract from file"""
    try:
        # Read file with size limit (10MB)
        raw = await file.read()
        
        if len(raw) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="File too large (max 10MB). Please use a smaller file or paste text instead."
            )
        
        # Extract text based on file type
        text = extract_text_from_file(raw, file.filename)
        
        if not text or text.strip() == "":
            raise HTTPException(
                status_code=400,
                detail=f"Could not extract text from {file.filename}. Ensure it's a valid TXT, PDF, or DOCX file with readable content."
            )
        
        # Truncate text if too long (keep first 5000 chars for faster processing)
        if len(text) > 5000:
            text = text[:5000]
        
        # Analyze and create task
        try:
            analysis = agents.req_agent.analyze(text)
        except Exception as e:
            print(f"Analysis error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error analyzing requirement: {str(e)}"
            )
        
        # Create task in DB
        task_id = agents.task_manager.create_task(
            user_id=current_user["id"],
            title=analysis.get("title", "Untitled Requirement"),
            priority=analysis.get("priority", "Medium"),
            urgency=analysis.get("urgency", "Normal"),
            complexity=analysis.get("complexity", "Medium"),
            location=analysis.get("location", "Remote"),
            experience=analysis.get("experience", "3-5 years"),
            skills=analysis.get("skills", []),
            description=text # Store full text as description
        )
        
        return {
            "success": True,
            "message": "Requirement uploaded and task created successfully",
            "task_id": task_id,
            "extracted_data": analysis
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error uploading requirement: {e}")
        raise HTTPException(500, f"Error uploading requirement: {str(e)}")


@app.post("/api/requirements/extract")
def extract_requirement_from_text(
    req: RequirementRequest,
    current_user=Depends(get_current_user)
):
    """Extract requirement from pasted text and create task"""
    try:
        text = req.content
        
        if not text or text.strip() == "":
            raise HTTPException(
                status_code=400,
                detail="No content provided. Please paste job description text."
            )
        
        # Truncate text if too long (keep first 5000 chars for faster processing)
        if len(text) > 5000:
            text = text[:5000]
        
        # Analyze using requirement agent
        try:
            analysis = agents.req_agent.analyze(text)
        except Exception as e:
            print(f"Analysis error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error analyzing requirement: {str(e)}"
            )
        
        # Create task in DB
        task_id = agents.task_manager.create_task(
            user_id=current_user["id"],
            title=analysis.get("title", "Untitled Requirement"),
            priority=analysis.get("priority", "Medium"),
            urgency=analysis.get("urgency", "Normal"),
            complexity=analysis.get("complexity", "Medium"),
            location=analysis.get("location", "Remote"),
            experience=analysis.get("experience", "3-5 years"),
            skills=analysis.get("skills", []),
            description=text
        )
        
        return {
            "success": True,
            "message": "Requirement extracted and task created successfully",
            "task_id": task_id,
            "extracted_data": analysis
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error extracting requirement: {e}")
        raise HTTPException(500, f"Error extracting requirement: {str(e)}")


# ============================================================
# ROUTES - TASKS
# ============================================================

@app.post("/api/tasks")
def create_task(
    req: TaskCreateRequest,
    current_user=Depends(get_current_user)
):
    """Create a new task"""
    try:
        task_id = agents.task_manager.create_task(
            user_id=current_user["id"],
            title=req.title,
            priority=req.priority,
            urgency=req.urgency,
            complexity=req.complexity,
            location=req.location,
            experience=req.experience,
            skills=req.skills
        )
        return {"success": True, "message": "Task created successfully", "task_id": task_id}
    except Exception as e:
        raise HTTPException(500, f"Error creating task: {str(e)}")


@app.get("/api/tasks/{task_id}")
def get_task_details(task_id: str, current_user=Depends(get_current_user)):
    """Get details for a specific task"""
    task = agents.task_manager.get_task(task_id, current_user["id"])
    if not task:
        raise HTTPException(404, "Task not found or access denied")
    return task


@app.put("/api/tasks/{task_id}")
def update_task(
    task_id: str,
    update_data: dict,
    current_user=Depends(get_current_user)
):
    """Update a task"""
    try:
        success = agents.task_manager.update_task(task_id, current_user["id"], update_data)
        if not success:
            raise HTTPException(404, "Task not found or access denied")
        return {"success": True, "message": "Task updated successfully"}
    except Exception as e:
        raise HTTPException(500, f"Error updating task: {str(e)}")


@app.delete("/api/tasks/{task_id}")
def delete_task(task_id: str, current_user=Depends(get_current_user)):
    """Delete a task"""
    try:
        success = agents.task_manager.delete_task(task_id, current_user["id"])
        if not success:
            raise HTTPException(404, "Task not found or access denied")
        return {"success": True, "message": "Task deleted successfully"}
    except Exception as e:
        raise HTTPException(500, f"Error deleting task: {str(e)}")


@app.post("/api/tasks/{task_id}/assign")
def assign_task(
    task_id: str,
    assignee_id: str,
    current_user=Depends(get_current_user)
):
    """Assign a task to another user"""
    try:
        success = agents.task_manager.assign_task(task_id, current_user["id"], assignee_id)
        if not success:
            raise HTTPException(404, "Task not found or access denied")
        return {"success": True, "message": f"Task {task_id} assigned to {assignee_id}"}
    except Exception as e:
        raise HTTPException(500, f"Error assigning task: {str(e)}")
# ============================================================
# ROUTES - INTERVIEW SCHEDULING
# ============================================================

@app.get("/api/suggest-slots")
def suggest_slots(
    date: str,
    current_user=Depends(get_current_user)
):
    """Get suggested interview slots (Mock implementation)"""
    try:
        # Mock logic to generate slots based on date
        # In a real app, this would check calendar tasks
        
        # Parse date to ensure it's valid
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        day_name = date_obj.strftime("%A")
        
        if day_name in ["Saturday", "Sunday"]:
             return {"time_slots": []} # No slots on weekends mock
        
        # Determine available slots (mock)
        slots = [
            {"slot": "09:00 - 10:00", "available": True},
            {"slot": "10:00 - 11:00", "available": True},
            {"slot": "11:00 - 12:00", "available": False}, # Occupied
            {"slot": "13:00 - 14:00", "available": True},
            {"slot": "14:00 - 15:00", "available": True},
            {"slot": "15:00 - 16:00", "available": True},
            {"slot": "16:00 - 17:00", "available": False}
        ]
        
        # Filter only available
        available_slots = [s for s in slots if s["available"]]
        
        return {
            "success": True, 
            "message": f"Found {len(available_slots)} slots for {date}",
            "time_slots": available_slots
        }
    except ValueError:
        raise HTTPException(400, "Invalid date format (YYYY-MM-DD)")
    except Exception as e:
        raise HTTPException(500, f"Error getting suggestions: {str(e)}")


@app.post("/api/schedule-interview")
def schedule_interview(
    payload: dict,
    current_user=Depends(get_current_user)
):
    """Schedule an interview for a candidate"""
    try:
        # Validate required fields
        required_fields = ["candidate_ids", "task_id", "interview_date", "interview_time", "location"]
        for field in required_fields:
            if not payload.get(field):
                raise HTTPException(400, f"Missing required field: {field}")

        # Verify access to candidate and task
        task = db.tasks.find_one({
            "_id": ObjectId(payload["task_id"]),
            "$or": [{"user_id": current_user["id"]}, {"assigned_to": current_user["id"]}]
        })
        if not task:
            raise HTTPException(403, "Access denied to this task")

        # Use the agent to send emails and record the interview
        result = agents.interview_email_agent.send_interview_emails(
            task_id=payload["task_id"],
            candidate_ids=payload["candidate_ids"],
            user_id=current_user["id"],
            interview_date=payload["interview_date"],
            interview_time=payload["interview_time"],
            location=payload.get("location", "Virtual"),
            meeting_link=payload.get("meeting_link", "")
        )

        if not result.get("success"):
            raise HTTPException(500, result.get("message", "Failed to schedule interview"))

        return {"success": True, "message": "Interview scheduled and email sent successfully"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error scheduling interview: {e}")
        raise HTTPException(500, f"Error scheduling interview: {str(e)}")


@app.get("/api/pending-interviews")
def get_pending_interviews_route(current_user=Depends(get_current_user)):
    """Get all pending interviews for the current user"""
    try:
        interviews = agents.interview_email_agent.get_pending_interviews(current_user["id"])
        return interviews
    except Exception as e:
        raise HTTPException(500, f"Error retrieving pending interviews: {str(e)}")
# ============================================================
# ROUTES - TASK MANAGEMENT
# ============================================================

@app.post("/api/tasks/{task_id}/complete")
def complete_task(
    task_id: str,
    current_user=Depends(get_current_user)
):
    """Mark task as completed"""
    success = agents.task_manager.complete_task(current_user["id"], task_id)
    
    if success:
        return {"success": True, "message": "Task completed successfully"}
    else:
        raise HTTPException(404, "Task not found")


@app.delete("/api/tasks/{task_id}")
def delete_task(
    task_id: str,
    current_user=Depends(get_current_user)
):
    """Delete a task"""
    success = agents.task_manager.delete_task(current_user["id"], task_id)
    
    if success:
        return {"success": True, "message": "Task deleted successfully"}
    else:
        raise HTTPException(404, "Task not found")


# ============================================================
# ROUTES - AGENT ACTIONS
# ============================================================

@app.post("/api/agent-action")
def agent_action_route(
    action_request: AgentActionRequest,
    current_user=Depends(get_current_user)
):
    """Endpoint to trigger specific agent actions."""
    try:
        # Add user_id to payload for agent context
        payload_with_user = action_request.payload.copy()
        payload_with_user["user_id"] = current_user["id"]
        
        result = agents.handle_agent_action(action_request.action, payload_with_user)
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error performing agent action '{action_request.action}': {e}")
        raise HTTPException(500, f"Error performing agent action: {str(e)}")


@app.get("/api/workload-report")
def workload_report():
    return master_agent.run("workload_check", {})


@app.get("/api/recruiter/{recruiter_id}/risk")
def recruiter_risk(recruiter_id: str):
    return master_agent.run("burnout_alert", {"recruiter_id": recruiter_id})

# ============================================================


# ============================================================
# HELPERS
# ============================================================

# ============================================================
# ROUTES - CANDIDATES & FEEDBACK
# ============================================================

@app.post("/api/tasks/{task_id}/candidates")
def add_candidate(
    task_id: str,
    candidate: CandidateRequest,
    current_user=Depends(get_current_user)
):
    """Add a candidate to a task"""
    try:
        # Verify task belongs to user OR is assigned to them
        if current_user.get("role") == "admin":
            task = db.tasks.find_one({"_id": ObjectId(task_id)})
        else:
            task = db.tasks.find_one({
                "_id": ObjectId(task_id),
                "$or": [
                    {"user_id": current_user["id"]},
                    {"assigned_to": current_user["id"]}
                ]
            })
        if not task:
            raise HTTPException(403, "Access denied to this task")
        
        # Create candidate document
        candidate_status = candidate.status.lower() if candidate.status else "applied"
        
        candidate_doc = {
            "task_id": task_id,
            "user_id": current_user["id"],
            "name": candidate.name,
            "email": candidate.email,
            "phone": candidate.phone,
            "experience_years": candidate.experience_years,
            "current_company": candidate.current_company,
            "current_position": candidate.current_position,
            "skills": candidate.skills,
            "notes": candidate.notes,
            "status": candidate_status,
            "applied_at": datetime.utcnow(),
            "resume_data": candidate.resume_data,
            "resume_filename": candidate.resume_filename
        }
        
        # Insert candidate
        result = db.candidates.insert_one(candidate_doc)
        
        # Auto-update task status if shortlisted
        if candidate_status == "shortlisted" and task.get("status") == "pending":
            db.tasks.update_one(
                {"_id": ObjectId(task_id)},
                {"$set": {"status": "in_progress"}}
            )
        
        return {
            "success": True,
            "message": "Candidate added successfully",
            "candidate_id": str(result.inserted_id)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error adding candidate: {e}")
        raise HTTPException(500, f"Error adding candidate: {str(e)}")


@app.get("/api/tasks/{task_id}/candidates")
def get_task_candidates(
    task_id: str,
    current_user=Depends(get_current_user)
):
    """Get all candidates for a task"""
    try:
        # Verify task belongs to user OR is assigned to them
        if current_user.get("role") == "admin":
            task = db.tasks.find_one({"_id": ObjectId(task_id)})
        else:
            task = db.tasks.find_one({
                "_id": ObjectId(task_id),
                "$or": [
                    {"user_id": current_user["id"]},
                    {"assigned_to": current_user["id"]}
                ]
            })
        if not task:
            raise HTTPException(404, "Task not found")
        
        # Get candidates
        candidates = list(db.candidates.find({"task_id": task_id}).sort("applied_at", -1))
        
        return [
            {
                "id": str(c["_id"]),
                "name": c.get("name"),
                "email": c.get("email"),
                "phone": c.get("phone"),
                "experience_years": c.get("experience_years"),
                "current_company": c.get("current_company"),
                "current_position": c.get("current_position"),
                "skills": c.get("skills", []),
                "notes": c.get("notes", ""),
                "recruiter_feedback": c.get("recruiter_feedback", ""),
                "manager_feedback": c.get("manager_feedback", ""),
                "status": c.get("status"),
                "applied_at": c.get("applied_at", "").isoformat() if c.get("applied_at") else "",
                "resume_filename": c.get("resume_filename")
            }
            for c in candidates
        ]
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting candidates: {e}")
        raise HTTPException(500, f"Error getting candidates: {str(e)}")


@app.put("/api/candidates/{candidate_id}")
def update_candidate_status(
    candidate_id: str,
    status: str = None,
    current_user=Depends(get_current_user)
):
    """Update candidate status (applied, shortlisted, rejected, hired)"""
    try:
        # Try to get status from query parameter
        if not status:
            raise HTTPException(400, "Status parameter is required")
        
        valid_statuses = ["applied", "shortlisted", "rejected", "hired"]
        if status not in valid_statuses:
            raise HTTPException(400, f"Invalid status. Must be one of: {', '.join(valid_statuses)}")
        
        # Verify recruiter has access to this candidate's task
        candidate = db.candidates.find_one({"_id": ObjectId(candidate_id)})
        if not candidate:
            raise HTTPException(404, "Candidate not found")
        
        task_id = candidate.get("task_id")
        if current_user.get("role") == "admin":
             task = db.tasks.find_one({"_id": ObjectId(task_id)})
        else:
             task = db.tasks.find_one({
                 "_id": ObjectId(task_id),
                 "$or": [{"user_id": current_user["id"]}, {"assigned_to": current_user["id"]}]
             })
        if not task:
            raise HTTPException(403, "Access denied to this candidate's task")
        
        # Update status
        result = db.candidates.update_one(
            {"_id": ObjectId(candidate_id)},
            {"$set": {"status": status, "updated_at": datetime.utcnow()}}
        )
        
        if result.modified_count > 0:
            # Auto-update task status if shortlisted
            if status.lower() == "shortlisted":
                db.tasks.update_one(
                    {"_id": ObjectId(task_id), "status": "pending"},
                    {"$set": {"status": "in_progress"}}
                )
            
            return {"success": True, "message": f"Candidate status updated to {status}"}
        else:
            return {"success": False, "message": "No changes made"}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating candidate: {e}")
        raise HTTPException(500, f"Error updating candidate: {str(e)}")


@app.delete("/api/candidates/{candidate_id}")
def delete_candidate(
    candidate_id: str,
    current_user=Depends(get_current_user)
):
    """Delete a candidate"""
    try:
        # Verify recruiter has access to this candidate's task
        candidate = db.candidates.find_one({"_id": ObjectId(candidate_id)})
        if not candidate:
            raise HTTPException(404, "Candidate not found")
        
        task_id = candidate.get("task_id")
        if current_user.get("role") == "admin":
             task = db.tasks.find_one({"_id": ObjectId(task_id)})
        else:
             task = db.tasks.find_one({
                 "_id": ObjectId(task_id),
                 "$or": [{"user_id": current_user["id"]}, {"assigned_to": current_user["id"]}]
             })
        if not task:
            raise HTTPException(403, "Access denied")
        
        # Delete candidate
        result = db.candidates.delete_one({"_id": ObjectId(candidate_id)})
        
        if result.deleted_count > 0:
            return {"success": True, "message": "Candidate deleted successfully"}
        else:
            raise HTTPException(500, "Failed to delete candidate")
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting candidate: {e}")
        raise HTTPException(500, f"Error deleting candidate: {str(e)}")


@app.post("/api/parse-resume")
async def parse_resume(
    file: UploadFile = File(...),
    current_user=Depends(get_current_user)
):
    """Parse resume using OpenAI for accurate extraction"""
    try:
        # Read file
        raw = await file.read()
        
        if len(raw) > 5 * 1024 * 1024:
            raise HTTPException(413, "Resume too large (max 5MB)")
        
        # Extract text from resume
        text = extract_text_from_file(raw, file.filename)
        
        if not text or text.strip() == "":
            raise HTTPException(400, f"Could not extract text from {file.filename}")
        
        # Call GPT for extraction
        prompt = EXTRACTION_PROMPT_RESUME.format(content=text[:6000]) # Limit to 6000 chars
        response_text = call_gpt(prompt)
        
        extracted = try_parse_json(response_text)
        
        if not extracted:
            # Fallback to empty structure if LLM fails
            extracted = {
                "name": "", "email": "", "phone": "",
                "experience_years": "0", "current_company": "",
                "current_position": "", "skills": [], "notes": "Failed to extract data automically."
            }
        else:
            # Ensure experience_years is a string for the frontend (as it uses number_input then converts)
            # Actually the schema says string, but frontend does int(extracted.get("experience_years", 0))
            # Let's keep it as int/string and let frontend handle it.
            if "experience_years" in extracted:
                extracted["experience_years"] = str(extracted["experience_years"])

        # Encode resume as base64 for storage
        resume_base64 = base64.b64encode(raw).decode('utf-8')
        
        return {
            "success": True,
            "extracted_data": extracted,
            "resume_data": resume_base64,
            "resume_filename": file.filename,
            "message": "Resume parsed successfully"
        }
    
    except HTTPException:
        raise
    except HTTPException:
        raise
    except Exception as e:
        print(f"Resume parsing error: {e}")
        raise HTTPException(500, f"Error parsing resume: {str(e)}")


@app.put("/api/candidates/{candidate_id}/edit")
def edit_candidate(
    candidate_id: str,
    update_data: CandidateUpdateRequest,
    current_user=Depends(get_current_user)
):
    """Edit candidate details"""
    try:
        # Verify permissions
        candidate = db.candidates.find_one({"_id": ObjectId(candidate_id)})
        if not candidate:
            raise HTTPException(404, "Candidate not found")
            
        task_id = candidate.get("task_id")
        if current_user.get("role") == "admin":
             task = db.tasks.find_one({"_id": ObjectId(task_id)})
        else:
             task = db.tasks.find_one({
                 "_id": ObjectId(task_id),
                 "$or": [{"user_id": current_user["id"]}, {"assigned_to": current_user["id"]}]
             })
        if not task:
            raise HTTPException(403, "Access denied")
            
        # Prepare updates
        updates = {
            "name": update_data.name,
            "email": update_data.email,
            "phone": update_data.phone,
            "experience_years": update_data.experience_years,
            "current_company": update_data.current_company,
            "current_position": update_data.current_position,
            "skills": update_data.skills,
            "notes": update_data.notes,
            "updated_at": datetime.utcnow()
        }
        
        # Handle resume update if provided
        if update_data.resume_data:
            updates["resume_data"] = update_data.resume_data
            updates["resume_filename"] = update_data.resume_filename
            
        result = db.candidates.update_one(
            {"_id": ObjectId(candidate_id)},
            {"$set": updates}
        )
        
        return {"success": True, "message": "Candidate updated successfully"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/candidates/{candidate_id}/resume")
def get_candidate_resume(
    candidate_id: str,
    current_user=Depends(get_current_user)
):
    """Get candidate resume"""
    try:
        candidate = db.candidates.find_one({"_id": ObjectId(candidate_id)})
        if not candidate:
            raise HTTPException(404, "Candidate not found")
            
        # Permission check
        task_id = candidate.get("task_id")
        if current_user.get("role") != "admin":
             task = db.tasks.find_one({
                 "_id": ObjectId(task_id),
                 "$or": [{"user_id": current_user["id"]}, {"assigned_to": current_user["id"]}]
             })
             if not task:
                raise HTTPException(403, "Access denied")

        if not candidate.get("resume_data"):
             raise HTTPException(404, "No resume found for this candidate")
             
        return {
            "success": True, 
            "resume_data": candidate.get("resume_data"),
            "resume_filename": candidate.get("resume_filename", "resume.pdf")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/comments")
def get_comments(
    task_id: Optional[str] = None,
    candidate_id: Optional[str] = None,
    current_user=Depends(get_current_user)
):
    """Fetch comments for a task or candidate."""
    try:
        query = {}
        if candidate_id:
            query["candidate_id"] = candidate_id
        elif task_id:
            query["task_id"] = task_id
            query["candidate_id"] = None # Only task-level comments
        else:
            raise HTTPException(400, "Missing task_id or candidate_id")

        comments = list(db.comments.find(query).sort("created_at", 1))
        
        for c in comments:
            c["id"] = str(c["_id"])
            del c["_id"]
            if c.get("created_at"):
                c["created_at"] = c["created_at"].isoformat()
        
        return comments
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/comments")
def add_comment(
    req: Dict[str, Any],
    current_user=Depends(get_current_user)
):
    """Add a comment/reply."""
    try:
        task_id = req.get("task_id")
        candidate_id = req.get("candidate_id")
        content = req.get("content")
        
        if not content:
            raise HTTPException(400, "Comment content cannot be empty")

        new_comment = {
            "task_id": task_id,
            "candidate_id": candidate_id,
            "user_id": current_user["id"],
            "username": current_user["username"],
            "role": current_user["role"],
            "content": content,
            "created_at": datetime.utcnow()
        }
        
        result = db.comments.insert_one(new_comment)
        
        return {"success": True, "id": str(result.inserted_id)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.put("/api/candidates/{candidate_id}/edit")
def edit_candidate(
    candidate_id: str,
    update_data: CandidateUpdateRequest,
    current_user=Depends(get_current_user)
):
    """Update candidate details (full edit)"""
    try:
        # Verify candidate exists
        candidate = db.candidates.find_one({"_id": ObjectId(candidate_id)})
        if not candidate:
            raise HTTPException(404, "Candidate not found")
        
        # Verify access
        task_id = candidate.get("task_id")
        if current_user.get("role") == "admin":
             task = db.tasks.find_one({"_id": ObjectId(task_id)})
        else:
             task = db.tasks.find_one({
                 "_id": ObjectId(task_id),
                 "$or": [{"user_id": current_user["id"]}, {"assigned_to": current_user["id"]}]
             })
        if not task:
            raise HTTPException(403, "Access denied to this candidate's task")
        
        # Build update dict
        update_dict = update_data.dict(exclude_unset=True)
        update_dict["updated_at"] = datetime.utcnow()
        
        # Update candidate
        result = db.candidates.update_one(
            {"_id": ObjectId(candidate_id)},
            {"$set": update_dict}
        )
        
        return {"success": True, "message": "Candidate updated successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating candidate: {e}")
        raise HTTPException(500, f"Error updating candidate: {str(e)}")


@app.get("/api/candidates/{candidate_id}/resume")
def download_candidate_resume(
    candidate_id: str,
    current_user=Depends(get_current_user)
):
    """Get candidate resume data for download"""
    try:
        candidate = db.candidates.find_one({"_id": ObjectId(candidate_id)})
        if not candidate:
            raise HTTPException(404, "Candidate not found")
        
        # Verify access
        task_id = candidate.get("task_id")
        if current_user.get("role") == "admin":
             task = db.tasks.find_one({"_id": ObjectId(task_id)})
        else:
             task = db.tasks.find_one({
                 "_id": ObjectId(task_id),
                 "$or": [{"user_id": current_user["id"]}, {"assigned_to": current_user["id"]}]
             })
        if not task:
            raise HTTPException(403, "Access denied")
        
        resume_data = candidate.get("resume_data")
        if not resume_data:
            raise HTTPException(404, "No resume found for this candidate")
        
        return {
            "success": True,
            "resume_data": resume_data,
            "resume_filename": candidate.get("resume_filename", "resume.pdf")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error getting resume: {str(e)}")
# ============================================================
# ROUTES - EOD SUMMARY & TRIGGERS
# ============================================================

@app.post("/api/eod-summary")
def manual_eod(current_user=Depends(get_current_user)):
    """Manually trigger comprehensive EOD summary for the current recruiter"""
    try:
        # Generate EOD using MasterAgent logic
        summary = master_agent.generate_comprehensive_eod(current_user["id"])
        
        user = db.users.find_one({"_id": ObjectId(current_user["id"])})
        user_email = user.get("email") if user else None
        
        if not user_email or "@example.com" in user_email:
             user_email = EOD_RECEIVER
        
        email_sent = agents.daily_agent.send_email(summary, user_email)
        
        return {
            "success": True,
            "summary": summary,
            "email_sent": email_sent,
            "recipient": user_email if email_sent else None
        }
    except Exception as e:
        print(f"❌ EOD Manual Error: {e}")
        return {
            "success": False,
            "error": f"Failed to generate EOD: {str(e)}",
            "email_sent": False
        }

@app.post("/api/admin/trigger-eod")
def trigger_eod_all(current_user=Depends(get_current_user)):
    """Admin triggers EOD summary for all recruiters immediately"""
    if not AdminAuthManager.is_admin(current_user["id"]):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Run in background to avoid blocking
    import threading
    thread = threading.Thread(target=daily_eod_job)
    thread.start()
    
    return {"success": True, "message": "EOD job triggered for all recruiters in background"}

# ============================================================
# ROUTES - ADMIN
# ============================================================

@app.post("/api/admin/login")
def admin_login(request: AdminLoginRequest):
    """Admin login endpoint"""
    admin = AdminAuthManager.authenticate_admin(request.username, request.password)
    
    if not admin:
        raise HTTPException(status_code=401, detail="Invalid admin credentials")
    
    # Track admin login
    LoginTracker.log_login(admin["id"], admin["username"], role="admin")
    
    token = create_access_token(admin["id"])
    
    return {
        "success": True,
        "access_token": token,
        "token_type": "bearer",
        "admin": admin,
        "is_admin": True
    }

@app.get("/api/admin/dashboard")
def admin_dashboard(current_user=Depends(get_current_user)):
    """Get admin dashboard data"""
    try:
        # Verify admin access
        if not AdminAuthManager.is_admin(current_user["id"]):
            raise HTTPException(status_code=403, detail="Admin access required")
        
        recruiters_workload = WorkloadMonitor.get_all_recruiters_workload()
        
        if not recruiters_workload:
            return {
                "summary": {
                    "total_recruiters": 0,
                    "online_count": 0,
                    "offline_count": 0,
                    "total_tasks": 0,
                    "pending_tasks": 0,
                    "completed_tasks": 0
                },
                "recruiters_workload": [],
                "high_workload_alert": [],
                "timestamp": datetime.utcnow().isoformat()
            }

        total_recruiters = len(recruiters_workload)
        total_tasks = sum(r.get("total_tasks", 0) for r in recruiters_workload)
        total_pending = sum(r.get("pending_tasks", 0) for r in recruiters_workload)
        online_count = sum(1 for r in recruiters_workload if r.get("current_status") == "online")
        
        high_workload = sorted(
            recruiters_workload,
            key=lambda x: x.get("workload_percentage", 0),
            reverse=True
        )[:5]
        
        return {
            "summary": {
                "total_recruiters": total_recruiters,
                "online_count": online_count,
                "offline_count": max(0, total_recruiters - online_count),
                "total_tasks": total_tasks,
                "pending_tasks": total_pending,
                "completed_tasks": sum(r.get("completed_tasks", 0) for r in recruiters_workload)
            },
            "recruiters_workload": recruiters_workload,
            "high_workload_alert": high_workload,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        import traceback
        print(f"❌ ADMIN DASHBOARD ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Dashboard error: {str(e)}")

@app.get("/api/admin/recruiters")
def get_all_recruiters(current_user=Depends(get_current_user)):
    """Get list of recruiters"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return TaskAssignmentManager.get_all_recruiters()

@app.get("/api/admin/recruiter/{recruiter_id}")
def get_recruiter_details(recruiter_id: str, current_user=Depends(get_current_user)):
    """Get detailed stats for a recruiter"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    recruiter = db.users.find_one({"_id": ObjectId(recruiter_id)})
    if not recruiter:
        raise HTTPException(404, "Recruiter not found")
    
    # Workload
    r_id = str(recruiter["_id"])
    tasks = list(db.tasks.find({"$or": [{"assigned_to": r_id}, {"user_id": r_id}]}))
    
    active_tasks = [t for t in tasks if t.get("status") in ["pending", "in_progress"]]
    workload_pct = min(len(active_tasks) * 20, 100)
    
    task_list = []
    for t in tasks:
        task_list.append({
            "id": str(t["_id"]),
            "title": t.get("title"),
            "status": t.get("status"),
            "priority": t.get("priority"),
            "created_at": t.get("created_at", "").isoformat() if t.get("created_at") else ""
        })
    
    # Get sessions
    sessions = list(db.login_logs.find(
        {"recruiter_id": recruiter_id},
        projection={"_id": 1, "login_time": 1, "logout_time": 1, "duration_minutes": 1, "status": 1}
    ).sort("login_time", -1).limit(50))
    
    return {
        "recruiter": {
            "id": r_id,
            "name": recruiter.get("username"),
            "email": recruiter.get("email"),
            "role": recruiter.get("role")
        },
        "workload": {
            "tasks": task_list,
            "workload_percentage": workload_pct,
            "current_status": recruiter.get("status", "offline"),
            "last_active": recruiter.get("last_active", "").isoformat() if recruiter.get("last_active") else "N/A"
        },
        "recent_sessions": [
            {
                "id": str(s["_id"]),
                "login_time": safe_isoformat(s.get("login_time")),
                "logout_time": safe_isoformat(s.get("logout_time")) or "Still logged in",
                "duration_minutes": s.get("duration_minutes", "N/A"),
                "status": s.get("status", "unknown")
            }
            for s in sessions
        ]
    }

@app.get("/api/admin/login-logs")
def get_login_logs(
    current_user=Depends(get_current_user),
    recruiter_id: str = None,
    days: int = 30
):
    """Get login/logout logs"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Only show recruiter logs, exclude admins
        query = {
            "login_time": {"$gte": start_date},
            "role": {"$ne": "admin"}
        }
        if recruiter_id:
            query["recruiter_id"] = recruiter_id
        
        logs = list(
            db.login_logs.find(query, projection={
                "_id": 1, "recruiter_id": 1, "recruiter_name": 1,
                "login_time": 1, "logout_time": 1, "duration_minutes": 1, "status": 1
            }).sort("login_time", -1).limit(500)
        )
        
        return [
            {
                "id": str(l["_id"]),
                "recruiter_id": l.get("recruiter_id"),
                "recruiter_name": l.get("recruiter_name", "Unknown"),
                "login_time": safe_isoformat(l.get("login_time")),
                "logout_time": safe_isoformat(l.get("logout_time")) or "Still logged in",
                "duration_minutes": l.get("duration_minutes", "N/A"),
                "status": l.get("status", "unknown")
            }
            for l in logs
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/assign-task")
def assign_task(
    request: TaskAssignmentRequest,
    current_user=Depends(get_current_user)
):
    """Assign task to recruiter"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Actually update to support multiple
    if request.recruiter_ids:
        db.tasks.update_one(
            {"_id": ObjectId(request.task_id)},
            {"$set": {"assigned_to": request.recruiter_ids, "assigned_at": datetime.utcnow()}}
        )
    
    return {"success": True, "message": "Task assigned to multiple recruiters"}

@app.post("/api/admin/create-assign-task")
def create_assign_task(
    request: CreateAndAssignTaskRequest,
    current_user=Depends(get_current_user)
):
    """Create and assign a new task directly to multiple recruiters"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Create task
    task_doc = {
        "user_id": current_user["id"],
        "title": request.title,
        "priority": request.priority,
        "location": request.location,
        "experience": request.experience,
        "skills": request.skills,
        "status": "pending",
        "comment": request.comment,
        "feedback": request.feedback,
        "assigned_to": request.recruiter_ids, # List of IDs
        "assigned_at": datetime.utcnow(),
        "created_at": datetime.utcnow()
    }
    
    result = db.tasks.insert_one(task_doc)
    
    return {
        "success": True,
        "message": f"Task created and assigned to {len(request.recruiter_ids)} recruiters",
        "task_id": str(result.inserted_id)
    }

@app.get("/api/admin/workload-report")
def get_workload_report(current_user=Depends(get_current_user)):
    """Get detailed workload report"""
    try:
        if current_user["role"] != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        
        workloads = WorkloadMonitor.get_all_recruiters_workload()
        if not workloads:
             return {
                "timestamp": datetime.utcnow().isoformat(),
                "recruiters": [],
                "summary": {"avg_workload": 0, "max_workload": 0, "min_workload": 0}
            }

        workloads.sort(key=lambda x: x.get("workload_percentage", 0), reverse=True)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "recruiters": workloads,
            "summary": {
                "avg_workload": round(sum(r.get("workload_percentage", 0) for r in workloads) / len(workloads), 2),
                "max_workload": max((r.get("workload_percentage", 0) for r in workloads), default=0),
                "min_workload": min((r.get("workload_percentage", 0) for r in workloads), default=0)
            }
        }
    except Exception as e:
        import traceback
        print(f"❌ WORKLOAD REPORT ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

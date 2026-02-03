"""
AI-Powered Recruiter OS v3.0 - Streamlit Frontend (ENHANCED)
HuggingFace + MongoDB Powered

Features:
✅ Task Management Dashboard
✅ Add Tasks from JD Extraction
✅ Complete/Delete Task Button
✅ No Duplicate Tasks
✅ EOD Email with Pending Tasks Only
✅ Enhanced Extraction with NER
"""

import streamlit as st
import requests
import json
from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd
import time

# Configuration
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="🎯 Recruiter OS v3.0",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        color: #1f77b4;
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        margin: 20px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .task-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .priority-high {
        color: #d62728;
        font-weight: bold;
        padding: 5px 10px;
        background: #ffcccc;
        border-radius: 5px;
    }
    .priority-medium {
        color: #ff7f0e;
        font-weight: bold;
        padding: 5px 10px;
        background: #ffe6cc;
        border-radius: 5px;
    }
    .priority-low {
        color: #2ca02c;
        font-weight: bold;
        padding: 5px 10px;
        background: #ccffcc;
        border-radius: 5px;
    }
    .urgency-immediate {
        color: #d62728;
        font-weight: bold;
        padding: 3px 8px;
        background: #ffcccc;
        border-radius: 3px;
        font-size: 0.85em;
    }
    .urgency-week {
        color: #ff7f0e;
        font-weight: bold;
        padding: 3px 8px;
        background: #ffe6cc;
        border-radius: 3px;
        font-size: 0.85em;
    }
    .urgency-flexible {
        color: #2ca02c;
        font-weight: bold;
        padding: 3px 8px;
        background: #ccffcc;
        border-radius: 3px;
        font-size: 0.85em;
    }
    .status-pending {
        color: #ff7f0e;
        padding: 3px 8px;
        background: #ffe6cc;
        border-radius: 3px;
        font-size: 0.85em;
    }
    .status-completed {
        color: #2ca02c;
        padding: 3px 8px;
        background: #ccffcc;
        border-radius: 3px;
        font-size: 0.85em;
    }
    </style>
""", unsafe_allow_html=True)

# Session state
if "token" not in st.session_state:
    st.session_state.token = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None
if "page" not in st.session_state:
    st.session_state.page = "login"


# ============================================================================
# API FUNCTIONS
# ============================================================================

def api_request(
    method: str,
    endpoint: str,
    data: Optional[Dict] = None,
    files: Optional[Dict] = None
) -> Dict:
    """Make API request"""
    headers = {}
    if st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=30)
        elif method == "POST":
            if files:
                # Longer timeout for file uploads (90 seconds)
                response = requests.post(url, headers=headers, files=files, timeout=90)
            else:
                response = requests.post(url, headers=headers, json=data, timeout=60)
        elif method == "PUT":
            response = requests.put(url, headers=headers, json=data, timeout=30)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, timeout=30)
        else:
            return {"error": f"Unknown method: {method}"}
        
        if response.status_code in [200, 201]:
            return response.json()
        else:
            error_detail = "Unknown error"
            try:
                error_detail = response.json().get("detail", str(response.status_code))
            except:
                error_detail = str(response.status_code)
            return {"error": error_detail}
    except requests.exceptions.Timeout:
        return {"error": "Request timeout. File might be too large or network is slow. Please try pasting text instead."}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection error. Make sure backend is running on http://localhost:8000"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}


def login(username: str, password: str) -> bool:
    """Login"""
    result = api_request("POST", "/api/login", {"username": username, "password": password})
    
    if "error" in result:
        return False
    
    st.session_state.token = result.get("access_token")
    user = result.get("user")
    if isinstance(user, dict):
        st.session_state.user_id = user.get("id")
        st.session_state.username = user.get("username")
    st.session_state.page = "dashboard"
    return True


def register(username: str, password: str) -> bool:
    """Register"""
    result = api_request("POST", "/api/register", {"username": username, "password": password})
    
    if "error" in result:
        return False
    
    st.session_state.token = result.get("access_token")
    user = result.get("user")
    if isinstance(user, dict):
        st.session_state.user_id = user.get("id")
        st.session_state.username = user.get("username")
    st.session_state.page = "dashboard"
    return True


def logout() -> bool:
    """Logout"""
    api_request("POST", "/api/logout")
    
    st.session_state.token = None
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.page = "login"
    return True


def get_dashboard_data() -> Dict:
    """Get dashboard"""
    return api_request("GET", "/api/dashboard")


def get_tasks(status: str = None) -> List[Dict]:
    """Get tasks"""
    if status:
        result = api_request("GET", f"/api/tasks?status={status}")
    else:
        result = api_request("GET", "/api/tasks")
    
    if isinstance(result, list):
        return result
    return []


def extract_requirement(content: str) -> Dict:
    """Extract requirement"""
    return api_request("POST", "/api/requirements/extract", {
        "content": content
    })


def upload_file(file) -> Dict:
    """Upload file with retry logic"""
    max_retries = 2
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            files = {"file": file}
            result = api_request("POST", "/api/upload-requirement", files=files)
            
            if "error" not in result:
                return result
            else:
                # If specific error about timeout/large file, suggest text paste
                error_msg = result.get("error", "")
                if "timeout" in error_msg.lower() or "large" in error_msg.lower():
                    return {
                        "error": f"{error_msg}\n\n💡 Tip: Try using 'Paste Content' instead for faster extraction.",
                        "suggestion": "use_text"
                    }
                return result
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                return {
                    "error": f"Upload failed after {max_retries} attempts: {str(e)}\n\n💡 Tip: Try using 'Paste Content' instead.",
                    "suggestion": "use_text"
                }
    
    return result


def upload_multiple_jds(files: List) -> Dict:
    """Upload and process multiple JDs via individual uploads"""
    try:
        results = []
        for f in files:
            try:
                # Reset file position to beginning
                f.seek(0)
                
                # Read file content
                file_content = f.read()
                
                # Determine content type based on file extension
                filename = f.name
                if filename.endswith('.pdf'):
                    content_type = 'application/pdf'
                elif filename.endswith('.docx'):
                    content_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                else:
                    content_type = 'text/plain'
                
                # Upload each file individually using the existing single file endpoint
                file_tuple = (filename, file_content, content_type)
                result = api_request("POST", "/api/upload-requirement", files={"file": file_tuple})
                results.append({"filename": filename, "result": result})
            except Exception as file_error:
                results.append({"filename": f.name, "result": {"error": str(file_error)}})
        
        # Check if any succeeded
        successful = [r for r in results if "error" not in r.get("result", {})]
        failed = [r for r in results if "error" in r.get("result", {})]
        
        return {
            "success": len(successful) > 0,
            "message": f"Processed {len(successful)} files successfully, {len(failed)} failed",
            "results": results
        }
    except Exception as e:
        return {"error": f"Multi-JD upload failed: {str(e)}"}


def complete_task(task_id: str) -> Dict:
    """Complete a task"""
    return api_request("POST", f"/api/tasks/{task_id}/complete", {})


def delete_task(task_id: str) -> Dict:
    """Delete a task"""
    return api_request("DELETE", f"/api/tasks/{task_id}")


def get_eod_summary() -> Dict:
    """Get EOD summary"""
    return api_request("POST", "/api/eod-summary", {})


def add_candidate(task_id: str, candidate_data: Dict) -> Dict:
    """Add candidate to task"""
    return api_request("POST", f"/api/tasks/{task_id}/candidates", candidate_data)


def get_task_candidates(task_id: str) -> List[Dict]:
    """Get all candidates for a task"""
    result = api_request("GET", f"/api/tasks/{task_id}/candidates")
    if isinstance(result, list):
        return result
    elif "error" in result:
        return []
    return []


def update_candidate_status(candidate_id: str, status: str) -> Dict:
    """Update candidate status"""
    # Note: Backend expects query parameter, so we'll pass it in the endpoint
    return api_request("PUT", f"/api/candidates/{candidate_id}?status={status}", {})


def delete_candidate(candidate_id: str) -> Dict:
    """Delete a candidate"""
    return api_request("DELETE", f"/api/candidates/{candidate_id}")



def get_workload_report() -> Dict:
    """Get recruiter workload monitoring report"""
    return api_request("GET", "/api/workload-report")

# ============================================================================
# UI PAGES
# ============================================================================

def show_login():
    """Login page"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="main-header">🎯 Recruiter OS </div>', unsafe_allow_html=True)
        st.divider()
        
        tab1, tab2 = st.tabs(["🔓 Login", "📝 Register"])
        
        with tab1:
            st.write("Welcome back!")
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            
            if st.button("🔓 Login", use_container_width=True):
                if username and password:
                    with st.spinner("Authenticating..."):
                        if login(username, password):
                            st.success("✅ Login successful!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials")
                else:
                    st.warning("⚠️ Enter username and password")
            
            st.divider()
        
        with tab2:
            st.write("Create account")
            new_user = st.text_input("Username", key="reg_user")
            new_pass = st.text_input("Password", type="password", key="reg_pass")
            conf_pass = st.text_input("Confirm", type="password", key="reg_conf")
            
            if st.button("📝 Register", use_container_width=True):
                if not new_user or not new_pass:
                    st.warning("⚠️ Fill all fields")
                elif new_pass != conf_pass:
                    st.warning("⚠️ Passwords don't match")
                else:
                    with st.spinner("Registering..."):
                        if register(new_user, new_pass):
                            st.success("✅ Registered!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Registration failed")


def show_dashboard():
    """Main dashboard"""
    col1, col2, col3 = st.columns([3, 2, 1])
    
    with col1:
        st.markdown('<div class="main-header">🎯 Recruiter OS Dashboard</div>', unsafe_allow_html=True)
    
    with col3:
        if st.button("🚪 Logout"):
            logout()
            st.rerun()
    
    st.markdown(f"**Welcome, {st.session_state.username}!** 👋")
    st.divider()
    
    # Tabs
    # Sidebar Navigation
    with st.sidebar:
        st.subheader("Navigation")
        selected_page = st.radio(
            "Go to",
            [
                "📊 Dashboard",
                "📋 All Tasks",
                "🔍 Extract & Upload",
                "📅 Interview Agent",
                "📊 Workload & EOD"
            ],
            label_visibility="collapsed"
        )
        
        st.divider()
        st.caption("🤖 AI Recruiter OS v1.2")

    # PAGE 1: Dashboard
    if selected_page == "📊 Dashboard":
        # ... (Dashboard content) ...
        pass # This implies we proceed into the dashboard block, stripping indentation later
    
    # We will handle the indentation logic by replacing the 'with tabX:' lines with 'if selected_page == ...:'
    # But since the content is indented inside 'with', we might need to handle indentation or use a trick.
    # The 'with' context manager indents content. 'if' also expects indentation.
    # So replacing 'with tab1:' with 'if selected_page == ...:' works perfectly for indentation preservation!

        st.subheader("📊 Dashboard Overview")
        
        with st.spinner("Loading..."):
            dashboard = get_dashboard_data()
        
        if "error" not in dashboard:
            col1, col2, col3, col4 = st.columns(4)
            
            stats = dashboard.get("stats", {})
            if not isinstance(stats, dict):
                stats = {}
            
            with col1:
                st.metric("📦 Total Tasks", stats.get("total_tasks", 0))
            with col2:
                st.metric("⏳ Pending", stats.get("pending", 0))
            with col3:
                st.metric("⚙️ In Progress", stats.get("in_progress", 0))
            with col4:
                st.metric("✅ Completed", stats.get("completed", 0))
            
            st.divider()
            
            st.subheader("📋 Active Tasks (Pending & In Progress)")
            recent = dashboard.get("recent_tasks", [])
            
            # Filter to show only pending and in_progress tasks (exclude completed)
            active_tasks = [t for t in recent if t.get('status') not in ['completed']]
            
            if active_tasks:
                for task in active_tasks:
                    task_dict = task if isinstance(task, dict) else {}
                    
                    # Status badge
                    status = task_dict.get('status', 'pending')
                    status_color = "🟡" if status == "pending" else "🔵"
                    
                    # Priority badge
                    priority = task_dict.get('priority', 'Medium')
                    priority_emoji = "🔴" if priority == "High" else "🟡" if priority == "Medium" else "🟢"
                    
                    # Urgency badge
                    urgency = task_dict.get('urgency', 'Flexible')
                    urgency_emoji = "🔴" if urgency == "Immediate" else "🟡" if urgency == "1 Week" else "🟢"
                    
                    col_task, col_actions = st.columns([4, 1])
                    
                    with col_task:
                        st.markdown(f"""
                        **{status_color} {task_dict.get('title', 'N/A')}**  
                        {priority_emoji} {priority} | {urgency_emoji} {urgency} | Created: {task_dict.get('created_at', 'N/A')[:10]}
                        """)
                    
                    with col_actions:
                        task_id = task_dict.get('id', '')
                        if task_id:
                            if st.button("✅", key=f"complete_{task_id}", help="Complete task"):
                                result = complete_task(task_id)
                                if result.get("success"):
                                    st.success("✅ Task completed and removed from active tasks!")
                                    time.sleep(1)
                                    st.rerun()
            else:
                st.info("📭 No active tasks. Create one by extracting a JD!")
            
            st.divider()
            
            # Show completed tasks separately
            st.subheader("✅ Completed Tasks")
            completed_tasks = [t for t in recent if t.get('status') == 'completed']
            
            if completed_tasks:
                for task in completed_tasks:
                    task_dict = task if isinstance(task, dict) else {}
                    
                    # Priority badge
                    priority = task_dict.get('priority', 'Medium')
                    priority_emoji = "🔴" if priority == "High" else "🟡" if priority == "Medium" else "🟢"
                    
                    # Urgency badge
                    urgency = task_dict.get('urgency', 'Flexible')
                    urgency_emoji = "🔴" if urgency == "Immediate" else "🟡" if urgency == "1 Week" else "🟢"
                    
                    col_task, col_date = st.columns([4, 1])
                    
                    with col_task:
                        st.markdown(f"""
                        **✅ {task_dict.get('title', 'N/A')}**  
                        {priority_emoji} {priority} | {urgency_emoji} {urgency}
                        """)
                    
                    with col_date:
                        completed_at = task_dict.get('completed_at', 'N/A')[:10] if task_dict.get('completed_at') else 'N/A'
                        st.caption(f"Done: {completed_at}")
            else:
                st.info("No completed tasks yet")
        else:
            st.error(f"❌ {dashboard.get('error')}")
    
    # PAGE 2: All Tasks Management
    if selected_page == "📋 All Tasks":
        st.subheader("📋 All Tasks")
        
        col_filter1, col_filter2 = st.columns(2)
        
        with col_filter1:
            status_filter = st.multiselect(
                "Status",
                ["pending", "in_progress", "completed"],
                default=["pending", "in_progress"]
            )
        
        with col_filter2:
            search_term = st.text_input("🔍 Search tasks")
        
        with st.spinner("Loading tasks..."):
            all_tasks = get_tasks()
        
        if all_tasks:
            # Filter
            filtered_tasks = [
                t for t in all_tasks
                if t.get("status") in status_filter
                and (search_term.lower() in t.get('title', '').lower() or not search_term)
            ]
            
            if filtered_tasks:
                for task in filtered_tasks:
                    col_info, col_action = st.columns([5, 1])
                    
                    with col_info:
                        status = task.get('status', 'pending')
                        priority = task.get('priority', 'Medium')
                        urgency = task.get('urgency', 'Flexible')
                        
                        status_emoji = "🟡" if status == "pending" else "🟢" if status == "completed" else "🔵"
                        priority_emoji = "🔴" if priority == "High" else "🟡" if priority == "Medium" else "🟢"
                        urgency_emoji = "🔴" if urgency == "Immediate" else "🟡" if urgency == "1 Week" else "🟢"
                        
                        with st.expander(f"{status_emoji} {task.get('title')} | {priority_emoji}{priority} | {urgency_emoji}{urgency}"):
                            col_left, col_right = st.columns(2)
                            
                            with col_left:
                                st.write(f"**Priority:** {priority}")
                                st.write(f"**Urgency:** {urgency}")
                                st.write(f"**Complexity:** {task.get('complexity', 'N/A')}")
                            
                            with col_right:
                                st.write(f"**Status:** {status}")
                                st.write(f"**Location:** {task.get('location', 'N/A')}")
                                st.write(f"**Experience:** {task.get('experience', 'N/A')}")
                            
                            st.write(f"**Created:** {task.get('created_at', 'N/A')[:10]}")
                            
                            skills = task.get('skills', [])
                            if skills:
                                st.write("**Skills:**")
                                skill_cols = st.columns(3)
                                for idx, skill in enumerate(skills[:9]):
                                    with skill_cols[idx % 3]:
                                        st.write(f"• {skill}")
                            
                            st.divider()
                            
                            # Candidates section
                            st.subheader("👥 Add Candidate to This Task")
                            
                            task_id = task.get('id', '')
                            
                            # Add candidate form (candidates are stored)
                            with st.form(f"candidate_form_{task_id}", border=True):
                                st.write("📝 **Candidate Details:**")
                                
                                col_c1, col_c2 = st.columns(2)
                                
                                with col_c1:
                                    cand_name = st.text_input(
                                        "Candidate Name *",
                                        placeholder="Full name",
                                        key=f"cand_name_{task_id}"
                                    )
                                    cand_email = st.text_input(
                                        "Email Address *",
                                        placeholder="candidate@email.com",
                                        key=f"cand_email_{task_id}"
                                    )
                                    cand_phone = st.text_input(
                                        "Phone Number *",
                                        placeholder="+1-555-0123",
                                        key=f"cand_phone_{task_id}"
                                    )
                                
                                with col_c2:
                                    cand_exp = st.number_input(
                                        "Years of Experience",
                                        min_value=0,
                                        max_value=60,
                                        value=0,
                                        key=f"cand_exp_{task_id}"
                                    )
                                    cand_company = st.text_input(
                                        "Current Company",
                                        placeholder="Company name",
                                        key=f"cand_company_{task_id}"
                                    )
                                    cand_position = st.text_input(
                                        "Current Position",
                                        placeholder="Job title",
                                        key=f"cand_position_{task_id}"
                                    )
                                
                                cand_skills = st.multiselect(
                                    "Candidate Skills",
                                    options=skills if skills else ["Python", "JavaScript", "Java", "C++", "AWS", "Azure", "MongoDB", "PostgreSQL"],
                                    key=f"cand_skills_{task_id}"
                                )
                                
                                cand_notes = st.text_area(
                                    "Additional Notes",
                                    placeholder="Interview feedback, strengths, concerns, etc.",
                                    height=80,
                                    key=f"cand_notes_{task_id}"
                                )
                                
                                cand_status = st.selectbox(
                                    "Initial Status",
                                    ["Applied", "Shortlisted", "Interview Scheduled", "Interviewed", "Offer Extended", "Rejected"],
                                    index=0,
                                    key=f"cand_status_{task_id}"
                                )
                                
                                # Form submission
                                col_submit_a, col_submit_b = st.columns(2)
                                
                                with col_submit_a:
                                    submitted = st.form_submit_button(
                                        "➕ Add Candidate",
                                        use_container_width=True,
                                        type="primary"
                                    )
                                
                                with col_submit_b:
                                    st.form_submit_button(
                                        "🔄 Clear Form",
                                        use_container_width=True
                                    )
                                
                                if submitted:
                                    # Validate required fields
                                    if not cand_name or not cand_email or not cand_phone:
                                        st.error("❌ Name, Email, and Phone are required!")
                                    else:
                                        # Create candidate data for API
                                        candidate_data = {
                                            "name": cand_name,
                                            "email": cand_email,
                                            "phone": cand_phone,
                                            "experience_years": str(cand_exp),
                                            "current_company": cand_company if cand_company else "N/A",
                                            "current_position": cand_position if cand_position else "N/A",
                                            "skills": cand_skills if cand_skills else [],
                                            "notes": cand_notes if cand_notes else "",
                                            "status": cand_status
                                        }
                                        
                                        # Save to MongoDB via API
                                        result = add_candidate(task_id, candidate_data)
                                        
                                        if result.get("success"):
                                            # Display confirmation
                                            st.success(f"✅ Candidate {cand_name} Added Successfully to MongoDB!")
                                            
                                            # Show candidate summary
                                            st.subheader("👤 Candidate Summary")
                                            col_summary1, col_summary2 = st.columns(2)
                                            
                                            with col_summary1:
                                                st.write(f"**Name:** {cand_name}")
                                                st.write(f"**Email:** {cand_email}")
                                                st.write(f"**Phone:** {cand_phone}")
                                                st.write(f"**Experience:** {cand_exp} years")
                                            
                                            with col_summary2:
                                                st.write(f"**Company:** {cand_company if cand_company else 'Not provided'}")
                                                st.write(f"**Position:** {cand_position if cand_position else 'Not provided'}")
                                                st.write(f"**Status:** {cand_status}")
                                                st.write(f"**Task:** {task.get('title')}")
                                            
                                            if cand_skills:
                                                st.write(f"**Skills:** {', '.join(cand_skills)}")
                                            
                                            if cand_notes:
                                                st.write(f"**Notes:** {cand_notes}")
                                            
                                            st.info(f"✅ Candidate saved to database! ID: {result.get('candidate_id')}")
                                            time.sleep(1)
                                            st.rerun()
                                        else:
                                            st.error(f"❌ Failed to add candidate: {result.get('error', 'Unknown error')}")
                            
                            # Display all candidates for this task (from MongoDB)
                            st.divider()
                            
                            # Fetch candidates from MongoDB API
                            candidates_list = get_task_candidates(task_id)
                            
                            if candidates_list:
                                st.subheader(f"📋 Candidates List ({len(candidates_list)})")
                                
                                # Status mapping for backend compatibility
                                status_map = {
                                    "Applied": "applied",
                                    "Shortlisted": "shortlisted",
                                    "Interview Scheduled": "shortlisted",
                                    "Interviewed": "shortlisted",
                                    "Offer Extended": "hired",
                                    "Rejected": "rejected"
                                }
                                
                                reverse_status_map = {
                                    "applied": "Applied",
                                    "shortlisted": "Shortlisted",
                                    "rejected": "Rejected",
                                    "hired": "Offer Extended"
                                }
                                
                                for idx, candidate in enumerate(candidates_list):
                                    st.markdown("---")
                                    display_status = reverse_status_map.get(candidate.get('status', 'applied'), candidate.get('status', 'Applied'))
                                    st.markdown(f"### 👤 {candidate.get('name', 'N/A')} ({display_status})")

                                    col_cd1, col_cd2 = st.columns(2)

                                    with col_cd1:
                                        st.write(f"**Email:** {candidate.get('email', 'N/A')}")
                                        st.write(f"**Phone:** {candidate.get('phone', 'N/A')}")
                                        st.write(f"**Experience:** {candidate.get('experience_years', 'N/A')} years")
                                        st.write(f"**Added:** {candidate.get('applied_at', 'N/A')[:10] if candidate.get('applied_at') else 'N/A'}")

                                    with col_cd2:
                                        st.write(f"**Company:** {candidate.get('current_company', 'N/A')}")
                                        st.write(f"**Position:** {candidate.get('current_position', 'N/A')}")
                                        st.write(f"**Current Status:** {display_status}")

                                    if candidate.get('skills'):
                                        st.write(f"**Skills:** {', '.join(candidate.get('skills', []))}")

                                    if candidate.get('notes'):
                                        st.write(f"**Notes:** {candidate.get('notes', '')}")

                                    col_act1, col_act2 = st.columns([2, 1])

                                    with col_act1:
                                        status_options = ["applied", "shortlisted", "rejected", "hired"]
                                        current_status = candidate.get('status', 'applied')
                                        current_idx = status_options.index(current_status) if current_status in status_options else 0
                                        
                                        new_status = st.selectbox(
                                            "Change Status",
                                            status_options,
                                            index=current_idx,
                                            key=f"status_{candidate.get('id', idx)}",
                                            format_func=lambda x: reverse_status_map.get(x, x.title())
                                        )
                                        
                                        if new_status != current_status:
                                            update_result = update_candidate_status(candidate.get('id'), new_status)
                                            if update_result.get("success"):
                                                st.success(f"✅ Status updated to {reverse_status_map.get(new_status, new_status)}")
                                                time.sleep(0.5)
                                                st.rerun()
                                            else:
                                                st.error(f"❌ Failed to update status")

                                    with col_act2:
                                        if st.button("🗑️ Delete", key=f"delete_cand_{candidate.get('id', idx)}", use_container_width=True):
                                            delete_result = delete_candidate(candidate.get('id'))
                                            if delete_result.get("success"):
                                                st.success("✅ Candidate deleted from database")
                                                time.sleep(0.5)
                                                st.rerun()
                                            else:
                                                st.error("❌ Failed to delete candidate")
                            else:
                                st.info("👇 No candidates added yet. Use the form above to add candidates to this task!")
                    
                    with col_action:
                        task_id = task.get('id', '')
                        if task_id:
                            col_complete, col_delete = st.columns(2)
                            
                            with col_complete:
                                if status != "completed":
                                    if st.button("✅", key=f"complete2_{task_id}", help="Complete"):
                                        result = complete_task(task_id)
                                        if result.get("success"):
                                            st.success("✅ Task completed!")
                                            time.sleep(1)
                                            st.rerun()
                            
                            with col_delete:
                                if st.button("🗑️", key=f"delete_{task_id}", help="Delete"):
                                    result = delete_task(task_id)
                                    if result.get("success"):
                                        st.success("✅ Task deleted!")
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.info("No tasks found with selected filters")
        else:
            st.info("📭 No tasks yet")
    
    # PAGE 3: Extract & Upload
    if selected_page == "🔍 Extract & Upload":
        with st.container():
            st.subheader("🔍 Extract Requirement from JD")
        
        extraction_result = None
        source_type = None
        
        # Initialize session state for tracking
        if "extract_triggered" not in st.session_state:
            st.session_state.extract_triggered = False
        
        # MULTI-JD UPLOAD SECTION (Top)
        st.markdown("### 📦 Upload Multiple JDs (Agent-Powered)")
        st.info("🤖 **AI Agent Extraction:** Upload multiple JD files.")
        
        multi_files = st.file_uploader(
            "Choose multiple JD files (PDF, TXT, DOCX)",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            key="multi_req_files"
        )
        
        if multi_files and st.button("🚀 Process Multiple JDs with Agent", use_container_width=True, key="multi_upload_btn"):
            with st.spinner(f"🤖 Agent analyzing {len(multi_files)} JDs using NER extraction..."):
                result = upload_multiple_jds(multi_files)
                
                if result.get("success"):
                    st.success(f"✅ {result.get('message', 'JDs processed successfully!')}")
                    st.info("📁 **Tasks have been saved to MongoDB** - They will persist even after logout!")
                    
                    st.subheader("📋 Extraction Results (Saved to Database)")
                    for item in result.get("results", []):
                        item_result = item.get("result", {})
                        if "error" not in item_result:
                            extracted = item_result.get('extracted_data', {})
                            with st.expander(f"✅ {item.get('filename')} → {extracted.get('title', 'Task')}", expanded=True):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**📌 Priority:** {extracted.get('priority', 'N/A')}")
                                    st.write(f"**⏰ Urgency:** {extracted.get('urgency', 'N/A')}")
                                    st.write(f"**📍 Location:** {extracted.get('location', 'N/A')}")
                                with col2:
                                    st.write(f"**📊 Complexity:** {extracted.get('complexity', 'N/A')}")
                                    st.write(f"**💼 Experience:** {extracted.get('experience', 'N/A')}")
                                    st.write(f"**🆔 Task ID:** `{item_result.get('task_id', 'N/A')}`")
                                
                                skills = extracted.get('skills', [])
                                if skills:
                                    st.write(f"**🛠️ Skills Extracted:** {', '.join(skills)}")
                                
                                if extracted.get('summary'):
                                    st.write(f"**📝 Summary:** {extracted.get('summary', '')[:200]}...")
                        else:
                            st.error(f"❌ {item.get('filename')} → {item_result.get('error', 'Unknown error')}")
                    
                    st.divider()
                    st.success("🎉 **All tasks are now in MongoDB!** Go to **Dashboard** or **Tasks** tab to view and manage them.")
                else:
                    st.error(f"Error: {result.get('error', 'Unknown error')}")
        
        st.divider()
        
        # UPLOAD SINGLE FILE SECTION
        st.markdown("### 📤 Upload Single File")
        file = st.file_uploader(
            "Choose file (PDF, TXT, DOCX)",
            type=["pdf", "txt", "docx"],
            key="req_file"
        )
        
        st.divider()
        
        # PASTE TEXT SECTION
        st.markdown("### 📝 Paste Content")
        text = st.text_area(
            "Paste job description:",
            height=200,
            placeholder="Job Title: ...\nResponsibilities: ...\nRequired Skills: ...",
            key="req_text"
        )
        
        st.divider()
        
        # Common Extract Button
        if st.button("🔍 Extract & Create Task", use_container_width=True, key="extract_btn"):
            st.session_state.extract_triggered = True
        
        # Process extraction if button was clicked
        if st.session_state.extract_triggered:
            if text.strip() and file:
                st.warning("⚠️ Please use either paste OR upload, not both")
            elif text.strip():
                with st.spinner("Analyzing text..."):
                    result = extract_requirement(text)
                    source_type = "text"
                    extraction_result = result
            elif file:
                with st.spinner("Processing file..."):
                    result = upload_file(file)
                    source_type = "file"
                    extraction_result = result
            else:
                st.warning("⚠️ Please paste text or upload a file")
            
            st.session_state.extract_triggered = False
        
        # Display Results
        if extraction_result:
            if "error" not in extraction_result:
                st.success("✅ Extracted & Task Created Successfully!")
                
                st.subheader("📊 Extracted Data")
                data = extraction_result.get("extracted_data", {})
                
                if not isinstance(data, dict):
                    data = {}
                
                # Display in two columns
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.write(f"**Title:** {data.get('title', 'N/A')}")
                    st.write(f"**Priority:** {data.get('priority', 'N/A')}")
                    st.write(f"**Urgency:** {data.get('urgency', 'N/A')}")
                    st.write(f"**Complexity:** {data.get('complexity', 'N/A')}")
                
                with col_b:
                    st.write(f"**Location:** {data.get('location', 'N/A')}")
                    st.write(f"**Experience:** {data.get('experience', 'N/A')}")
                    st.write(f"**Source:** {source_type.upper() if source_type else 'N/A'}")
                
                skills = data.get("skills", [])
                if skills and isinstance(skills, list):
                    st.write("**Required Skills:**")
                    skill_cols = st.columns(3)
                    for idx, skill in enumerate(skills[:9]):
                        with skill_cols[idx % 3]:
                            st.write(f"• {skill}")
                
                st.info(f"✅ Task ID: {extraction_result.get('task_id', 'N/A')}")
                st.info(f"📌 Message: {extraction_result.get('message', '')}")
            else:
                error_msg = extraction_result.get('error', 'Unknown error')
                st.error(f"❌ Error: {error_msg}")
                
                # Provide helpful suggestions
                if "pdf" in error_msg.lower() or "Could not extract" in error_msg:
                    st.info("💡 **PDF Extraction Tips:**\n"
                           "1. **Copy & Paste** text from PDF instead (fastest!)\n"
                           "2. Ensure PDF is **digital** (not scanned/image)\n"
                           "3. Try a **smaller PDF** file first\n"
                           "4. If PDF is protected, try converting it first\n"
                           "5. Use **pdftotext** or Adobe to extract first\n\n"
                           "**Quick Test:**\n"
                           "• Try the sample JD from QUICKSTART.md by pasting text")
                elif "timeout" in error_msg.lower():
                    st.info("💡 **Timeout Tips:**\n"
                           "• Try using smaller files\n"
                           "• Or paste the job description text directly\n"
                           "• Or paste just the key sections (title, skills, location)\n"
                           "• Text extraction is faster than file processing")
                elif "large" in error_msg.lower():
                    st.info("💡 **File Too Large:**\n"
                           "• Maximum file size: 10 MB\n"
                           "• Try extracting key sections only\n"
                           "• Or paste the content in text form")
                elif "extract" in error_msg.lower():
                    st.info("💡 **Extraction Failed:**\n"
                           "• Make sure the file has readable content\n"
                           "• PDF: Ensure it's not scanned/image-based\n"
                           "• DOCX: Check the file isn't corrupted\n"
                           "• Try pasting the text content instead\n"
                           "• Test with sample PDF to verify system works")
    
    
    # PAGE 4: Interview Agent
    if selected_page == "📅 Interview Agent":
        st.subheader("📅 Interview Scheduling Agent")
        st.caption("✅ Synced with Google Calendar")
        st.info("Automate interview invites and scheduling based on available slots.")
        
        # Form to schedule interviews
        with st.form("schedule_form"):
            # Fetch both pending and in_progress tasks
            all_tasks = get_tasks() or []
            tasks = [t for t in all_tasks if t.get('status') in ["pending", "in_progress"]]
            task_options = {t['id']: t['title'] for t in tasks}
            
            selected_task_id = None
            if task_options:
                selected_task_id = st.selectbox(
                    "Select Role",
                    options=list(task_options.keys()),
                    format_func=lambda x: f"{task_options[x]} ({next((t['status'] for t in tasks if t['id'] == x), 'Unknown')})"
                )
                if selected_task_id:
                    # Fetch candidates for this task
                    task_cand_list = get_task_candidates(selected_task_id)
                    # Include 'applied' as well so they can schedule for any applicant
                    shortlisted_cands = [c for c in task_cand_list if c.get('status') in ['applied', 'shortlisted', 'Interview Scheduled', 'Interviewed']]
                    
                    if shortlisted_cands:
                        selected_candidate_ids = st.multiselect(
                            "Select Candidates to Invite",
                            options=[c['id'] for c in shortlisted_cands],
                            format_func=lambda x: next((c['name'] for c in shortlisted_cands if c['id'] == x), "Unknown"),
                            default=[c['id'] for c in shortlisted_cands] # Default select all
                        )
                        st.caption(f"Selecting {len(selected_candidate_ids)} candidates")
                    else:
                        st.warning("⚠️ No shortlisted candidates found for this role.")
                        selected_candidate_ids = []
                else:
                    selected_candidate_ids = []

            else:
                st.warning("No active tasks found for scheduling")
                selected_candidate_ids = []
            
            schedule_date = st.date_input("Interview Date")
            
            # Get suggestions
            if st.form_submit_button("🔍 Get Slot Suggestions"):
                suggestions = api_request("GET", f"/api/suggest-slots?date={schedule_date}")
                if "time_slots" in suggestions:
                    st.session_state.suggested_slots = suggestions["time_slots"]
                    st.success("✅ Slots analyzed based on your calendar")
            
            # Show slots if available
            interview_time = st.selectbox(
                "Select Time Slot",
                options=[s['slot'] for s in st.session_state.get('suggested_slots', [])],
                placeholder="Choose a slot..."
            )
            
            location_type = st.selectbox("Location", ["Virtual", "In-Person"])
            meeting_link = st.text_input("Meeting Link (e.g., Zoom/Meet)")
            
            if st.form_submit_button("✉️ Send Invites to Selected Candidates"):
                if selected_task_id and schedule_date and interview_time and selected_candidate_ids:
                    payload = {
                        "task_id": selected_task_id,
                        "candidate_ids": selected_candidate_ids,
                        "date": str(schedule_date),
                        "time": interview_time,
                        "location": location_type,
                        "meeting_link": meeting_link
                    }
                    
                    with st.spinner("🤖 Scheduling Agent is sending emails..."):
                        result = api_request("POST", "/api/schedule-interviews", payload)
                        
                    if result.get("success"):
                        st.success(f"✅ {result.get('message')}")
                        if result.get("sent_details"):
                            with st.expander("Email Details"):
                                st.json(result.get("sent_details"))
                    else:
                        st.error(f"❌ Error: {result.get('error')}")
                else:
                    st.warning("⚠️ Please select all fields")

    # PAGE 5: Workload & EOD
    if selected_page == "📊 Workload & EOD":
        st.subheader("📊 EOD & Workload Agent")
        st.info("Monitor daily progress and generate end-of-day reports.")
        
        if st.button("📝 Generate Comprehensive EOD Report"):
            with st.spinner("🤖 Analysing daily activities..."):
                result = api_request("POST", "/api/eod-summary", {})
                
            if result.get("email_sent"):
                st.success(f"✅ EOD Report Sent to {result.get('recipient')}")
                with st.expander("📄 View Report", expanded=True):
                    st.text(result.get("summary"))
            else:
                if "summary" in result:
                    st.warning("⚠️ Report generated but email failed")
                    with st.expander("📄 View Report", expanded=True):
                        st.text(result.get("summary"))
                else:
                    st.error("❌ Failed to generate EOD summary")
        
        st.divider()
        
        st.markdown("### ⚠️ Workload Monitor")
        if st.button("🔄 Check Workload Status"):
            with st.spinner("Fetching workload data..."):
                report = api_request("GET", "/api/workload-report")
                
            if "workload_report" in report:
                workload_data = report["workload_report"]
                if workload_data:
                    import pandas as pd
                    df = pd.DataFrame(workload_data)
                    st.dataframe(df, use_container_width=True)
                    
                    for rec in workload_data:
                        if rec.get("risk_level") in ["High", "Medium"]:
                            st.warning(f"⚠️ {rec['name']}: {rec['risk_level']} Risk")
                else:
                    st.info("No workload data available.")
            else:
                    st.error("Failed to fetch workload report")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main app"""
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if health.status_code != 200:
            st.error("❌ Backend not responding")
            st.stop()
    except Exception as e:
        st.error(f"❌ Cannot connect: {str(e)}")
        st.error("Start backend: python -m uvicorn app:app --reload")
        st.stop()
    
    if st.session_state.token is None:
        show_login()
    else:
        show_dashboard()


if __name__ == "__main__":
    main()
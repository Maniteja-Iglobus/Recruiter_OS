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
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
# Configuration
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="🎯 Recruiter OS v3.0",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

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
    .online-badge {
        background-color: #28a745;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 0.85em;
    }
    .offline-badge {
        background-color: #dc3545;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 0.85em;
    }
    .high-workload {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .normal-workload {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    div[data-testid="stToast"] {
        background-color: #e3f2fd !important;
        color: #0d47a1 !important;
        border: 1px solid #bbdefb !important;
    }
    div[data-testid="stToast"] p {
        color: #0d47a1 !important;
    }
    </style>
""", unsafe_allow_html=True)
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
    .time-picker-label {
        text-align: center;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 20px;
        font-size: 1.2em;
    }
    .done-btn-container {
        margin-top: 20px;
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
if "admin_token" not in st.session_state:
    st.session_state.admin_token = None
if "admin_user" not in st.session_state:
    st.session_state.admin_user = None
if "page" not in st.session_state:
    st.session_state.page = "login"

# ============================================================================
# API FUNCTIONS
# ============================================================================

def api_request(
    method: str,
    endpoint: str,
    data: Optional[Dict] = None,
    files: Optional[Dict] = None,
    params: Optional[Dict] = None
) -> Dict:
    """Unified API request function for both User and Admin sessions"""
    headers = {}
    
    # Use admin_token if available (especially on admin pages)
    # Most admin endpoints are prefixed with /api/admin
    is_admin_endpoint = endpoint.startswith("/api/admin")
    
    if is_admin_endpoint and st.session_state.get("admin_token"):
        headers["Authorization"] = f"Bearer {st.session_state.admin_token}"
    elif st.session_state.get("token"):
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    elif st.session_state.get("admin_token"):
        headers["Authorization"] = f"Bearer {st.session_state.admin_token}"

    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, params=params, timeout=30)
        elif method == "POST":
            if files:
                # Longer timeout for file uploads
                response = requests.post(url, headers=headers, files=files, timeout=60)
            else:
                response = requests.post(url, headers=headers, json=data, timeout=180)
        elif method == "PUT":
            response = requests.put(url, headers=headers, json=data, timeout=30)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, timeout=30)
        else:
            return {"error": f"Unknown method: {method}"}
        
        if response.status_code in [200, 201]:
            try:
                return response.json()
            except:
                return {"success": True, "message": response.text}
        else:
            error_detail = "Unknown error"
            try:
                error_detail = response.json().get("detail", str(response.status_code))
            except:
                error_detail = str(response.status_code)
            return {"error": error_detail}
            
    except requests.exceptions.Timeout:
        return {"error": "Request timeout. Please try again."}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection error. Make sure backend is running on http://localhost:8000"}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

def show_admin_login():
    """Show admin login page"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<h1 style='text-align:center; color:#1f77b4;'>🔐 Admin Dashboard</h1>", 
                    unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; font-size:1.2em;'>Recruiter OS - Admin Panel</p>", 
                    unsafe_allow_html=True)
        
        st.divider()
        
        st.subheader("Login to Admin Panel")
        
        username = st.text_input("Admin Username", placeholder="Enter your admin username")
        password = st.text_input("Admin Password", type="password", placeholder="Enter your admin password")
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🔐 Login", use_container_width=True, type="primary"):
                if not username or not password:
                    st.error("❌ Please enter both username and password")
                else:
                    with st.spinner("Authenticating..."):
                        result = api_request(
                            "POST",
                            "/api/admin/login",
                            data={"username": username, "password": password}
                        )
                        
                        if "error" not in result and result.get("is_admin"):
                            st.session_state.admin_token = result["access_token"]
                            st.session_state.admin_user = result.get("admin") # Use 'admin' key for admin login
                            st.session_state.page = "dashboard"
                            st.toast("✅ Admin login successful!", icon="🔐")
                            st.rerun()
                        else:
                            st.error(f"❌ Login failed: {result.get('error', 'Unknown error')}")
                            st.toast("❌ Admin login failed!", icon="⚠️")
                            st.rerun()
        
        st.divider()
        
        st.info("""
        ### 🔐 Admin Access
                
        **Features:**
        - 📊 Monitor all recruiters
        - 📈 Workload tracking in real-time
        - 🕐 Login/Logout activity logs
        - 👥 Task assignments by recruiter
        - ⚠️ High workload alerts
        - 📋 Detailed performance reports
        """)


    

def login(username: str, password: str) -> bool:
    """Login"""
    username = username.strip()
    password = password.strip()
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


def register(username: str, password: str, email: str) -> bool:
    """Register"""
    username = username.strip()
    password = password.strip()
    result = api_request("POST", "/api/register", {"username": username, "password": password, "email": email})
    
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



def show_admin_dashboard():
    """Show admin dashboard"""
    # Safety check for session state
    if not st.session_state.get("admin_user"):
        st.error("⚠️ Session expired. Please login again.")
        if st.button("Back to Login"):
            logout()
            st.rerun()
        st.stop()

    # Sidebar
    with st.sidebar:
        st.markdown(f"### 🔐 Admin: {st.session_state.admin_user.get('username', 'Admin')}")
        st.markdown(f"📧 {st.session_state.admin_user.get('email', 'N/A')}")
        st.divider()
        
        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "👥 Recruiters", "🕐 Activity Logs", "📈 Workload Report", "⚙️ Settings"],
            key="admin_page"
        )
        
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            logout()
            st.session_state.admin_token = None
            st.session_state.admin_user = None
            st.session_state.page = "login"
            st.rerun()
        
    # Main content
    if page == "📊 Dashboard":
        show_dashboard_main()
    elif page == "👥 Recruiters":
        show_recruiters_page()
    elif page == "🕐 Activity Logs":
        show_activity_logs()
    elif page == "📈 Workload Report":
        show_workload_report()
    elif page == "⚙️ Settings":
        show_settings()

# ============================================================
# DASHBOARD MAIN
# ============================================================

@st.cache_data(ttl=30)
def get_admin_dashboard_data():
    """Get dashboard data"""
    return api_request("GET", "/api/admin/dashboard")

def show_dashboard_main():
    """Show main dashboard"""
    st.markdown("<h1 class='main-header'>📊 Admin Dashboard</h1>", unsafe_allow_html=True)
    
    dashboard = get_admin_dashboard_data()
    
    if "error" in dashboard:
        st.error(f"❌ Error loading dashboard: {dashboard['error']}")
        return
    
    summary = dashboard.get("summary", {})
    recruiters = dashboard.get("recruiters_workload", [])
    
    # Summary Metrics
    st.subheader("📊 Real-Time Summary Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("👥 Total Recruiters", summary.get("total_recruiters", 0))
    with col2:
        st.metric("🟢 Online Now", summary.get("online_count", 0))
    with col3:
        st.metric("📋 Total Tasks", summary.get("total_tasks", 0))
    with col4:
        st.metric("⏳ Pending Tasks", summary.get("pending_tasks", 0))
    with col5:
        st.metric("✅ Completed", summary.get("completed_tasks", 0))
    
    st.divider()
    
    # High Workload Alerts
    high_workload_recruiters = [r for r in recruiters if r["workload_percentage"] > 70]
    if high_workload_recruiters:
        st.subheader("⚠️ High Workload Alert - Attention Required")
        
        for recruiter in high_workload_recruiters:
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                st.write(f"**{recruiter['recruiter_name']}**")
                st.caption(f"Email: {recruiter.get('recruiter_email', 'N/A')}")
                st.caption(f"Tasks: {recruiter['total_tasks']} | Pending: {recruiter['pending_tasks']} | In Progress: {recruiter['in_progress_tasks']}")
            
            with col2:
                progress_color = "🔴" if recruiter["workload_percentage"] > 80 else "🟠"
                st.metric(f"{progress_color} Workload", f"{recruiter['workload_percentage']:.0f}%")
            
            with col3:
                st.metric("High Priority", recruiter.get("high_priority_count", 0))
            
            with col4:
                if recruiter["current_status"] == "online":
                    st.write("🟢 **Online**")
                else:
                    st.write("🔴 **Offline**")
        
        st.divider()
    
    # Recruiter Status Overview
    st.subheader("👥 Recruiter Status Overview")
    
    recruiter_data = []
    for r in recruiters:
        recruiter_data.append({
            "Name": r["recruiter_name"],
            "Status": "🟢 Online" if r["current_status"] == "online" else "🔴 Offline",
            "Tasks": r["total_tasks"],
            "Pending": r["pending_tasks"],
            "In Progress": r["in_progress_tasks"],
            "Completed": r["completed_tasks"],
            "Workload": f"{r['workload_percentage']:.0f}%",
            "Avg Time (h)": f"{r['avg_completion_hours']:.1f}"
        })
    
    if recruiter_data:
        df = pd.DataFrame(recruiter_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Charts
    if recruiters:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Workload Distribution")
            fig = px.bar(
                x=[r["recruiter_name"] for r in recruiters],
                y=[r["workload_percentage"] for r in recruiters],
                title="Recruiter Workload %",
                labels={"x": "Recruiter", "y": "Workload %"},
                color=[r["workload_percentage"] for r in recruiters],
                color_continuous_scale=["green", "orange", "red"]
            )
            fig.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="⚠️ Warning (70%)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🟢 Online/Offline Status")
            online = sum(1 for r in recruiters if r["current_status"] == "online")
            offline = len(recruiters) - online
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=["🟢 Online", "🔴 Offline"],
                    values=[online, offline],
                    marker=dict(colors=["#28a745", "#dc3545"])
                )
            ])
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# RECRUITERS PAGE
# ============================================================

def show_recruiter_profile(recruiter_id: str):
    """Show detailed recruiter profile"""
    if st.button("⬅️ Back to List"):
        st.session_state.selected_recruiter = None
        st.rerun()
    
    details = api_request("GET", f"/api/admin/recruiter/{recruiter_id}")
    
    if "error" in details:
        st.error(f"❌ Error loading details: {details['error']}")
        return
    
    recruiter = details.get("recruiter", {})
    workload = details.get("workload", {})
    sessions = details.get("recent_sessions", [])
    
    st.markdown(f"<h1 class='main-header'>👤 {recruiter.get('name', 'Unknown')}</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Email:** {recruiter.get('email', 'N/A')}")
        st.write(f"**ID:** {recruiter.get('id', 'N/A')}")
    with col2:
        status = workload.get("current_status", "offline")
        st.write(f"**Status:** {'🟢 Online' if status == 'online' else '🔴 Offline'}")
        st.write(f"**Last Active:** {workload.get('last_active', 'N/A')[:16]}")
    with col3:
        st.metric("Workload %", f"{workload.get('workload_percentage', 0):.0f}%")

    st.divider()
    
    tab_tasks, tab_sessions = st.tabs(["📋 Assigned Tasks", "🕐 Activity History"])
    
    with tab_tasks:
        tasks = workload.get("tasks", [])
        if not tasks:
            st.info("No tasks assigned to this recruiter.")
        else:
            for task in tasks:
                with st.expander(f"{task.get('title', 'Untitled')}"):
                    st.write(f"**Status:** {task.get('status', 'pending')}")
                    st.write(f"**Priority:** {task.get('priority', 'Medium')}")
                    st.write(f"**ID:** {task.get('id', 'N/A')}")
                    
                    # Add candidate display section
                    st.divider()
                    st.subheader("👥 Candidates")
                    candidates = get_task_candidates(task.get('id'))
                    if candidates:
                        cand_df = pd.DataFrame([
                            {
                                "Name": c.get("name"),
                                "Status": c.get("status"),
                                "Experience": f"{c.get('experience_years')} yrs",
                                "Email": c.get("email")
                            } for c in candidates
                        ])
                        st.dataframe(cand_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No candidates added for this task yet.")
    
    with tab_sessions:
        if not sessions:
            st.info("No session history found.")
        else:
            session_data = []
            for s in sessions:
                session_data.append({
                    "Login": s.get("login_time", "N/A"),
                    "Logout": s.get("logout_time", "N/A"),
                    "Duration (min)": str(s.get("duration_minutes", "N/A")),
                    "Status": s.get("status", "N/A")
                })
            st.table(session_data)

def show_recruiters_page():
    """Show recruiters page"""
    # Initialize state if not present
    if "selected_recruiter" not in st.session_state:
        st.session_state.selected_recruiter = None
    
    # Show profile if one is selected
    if st.session_state.selected_recruiter:
        show_recruiter_profile(st.session_state.selected_recruiter)
        return

    st.markdown("<h1 class='main-header'>👥 Recruiter Management</h1>", unsafe_allow_html=True)
    
    recruiters = api_request("GET", "/api/admin/recruiters")
    
    if "error" in recruiters:
        st.error(f"❌ Error: {recruiters['error']}")
        return
    
    dashboard = get_admin_dashboard_data()
    workloads = {r["recruiter_id"]: r for r in dashboard.get("recruiters_workload", [])}
    
    tab1, tab2 = st.tabs(["👥 All Recruiters", "📋 Assign Task"])
    
    with tab1:
        st.subheader("All Recruiters Overview")
        
        if not recruiters:
            st.info("No recruiters found")
            return
        
        for recruiter in recruiters:
            workload = workloads.get(recruiter["id"], {})
            workload_pct = workload.get("workload_percentage", 0)
            
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                st.markdown(f"### {recruiter['name']}")
                st.caption(f"📧 {recruiter['email']}")
                
                # Tasks info
                tasks_list = workload.get("tasks", [])
                if tasks_list:
                    st.write(f"**Assigned Tasks ({len(tasks_list)}):**")
                    for task in tasks_list[:5]:
                        priority_icon = "🔴" if task["priority"] == "High" else "🟠" if task["priority"] == "Medium" else "🟢"
                        st.caption(f"{priority_icon} {task['title'][:50]} ({task['status']})")
                    if len(tasks_list) > 5:
                        st.caption(f"... and {len(tasks_list) - 5} more tasks")
                else:
                    st.caption("No tasks assigned")
            
            with col2:
                status = workload.get("current_status", "offline")
                if status == "online":
                    st.markdown('<span class="online-badge">🟢 Online</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="offline-badge">🔴 Offline</span>', unsafe_allow_html=True)
                
                last_active = workload.get("last_active", "N/A")
                if isinstance(last_active, str):
                    last_active_date = last_active[:10]
                else:
                    last_active_date = "N/A"
                st.caption(f"Last active: {last_active_date}")
            
            with col3:
                st.metric("Workload", f"{workload_pct:.0f}%")
                
                if workload_pct > 70:
                    st.warning("⚠️ High")
                elif workload_pct > 50:
                    st.info("ℹ️ Moderate")
                else:
                    st.success("✅ Low")
            
            with col4:
                if st.button("View Details", key=f"details_{recruiter['id']}", use_container_width=True):
                    st.session_state.selected_recruiter = recruiter["id"]
                    st.rerun()
            
            st.divider()
    
    with tab2:
        st.subheader("📋 Assign & Create Tasks")
        
        assign_mode = st.radio("Assignment Mode", ["Assign Existing Task", "Create & Assign New Task"], horizontal=True)
        
        if assign_mode == "Assign Existing Task":
            # Get unassigned tasks
            try:
                tasks_response = api_request("GET", "/api/tasks")
                if isinstance(tasks_response, dict) and "tasks" in tasks_response:
                    tasks = tasks_response["tasks"]
                elif isinstance(tasks_response, list):
                    tasks = tasks_response
                else:
                    tasks = []
            except:
                tasks = []
            
            unassigned_tasks = [t for t in tasks if not t.get("assigned_to")]
            
            if unassigned_tasks:
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    selected_task = st.selectbox(
                        "Select Unassigned Task",
                        options=unassigned_tasks,
                        format_func=lambda x: f"{x.get('title', 'Untitled')[:40]} ({x.get('priority', 'Medium')})"
                    )
                
                with col2:
                    selected_recruiters = st.multiselect(
                        "Assign to Recruiters",
                        options=recruiters,
                        format_func=lambda x: x["name"],
                        placeholder="Select one or more..."
                    )
                
                with col3:
                    st.write("")
                    if st.button("✅ Assign Task", use_container_width=True, type="primary"):
                        if not selected_recruiters:
                            st.error("❌ Please select at least one recruiter")
                        else:
                            result = api_request(
                                "POST",
                                "/api/admin/assign-task",
                                data={
                                    "task_id": selected_task["id"],
                                    "recruiter_ids": [r["id"] for r in selected_recruiters]
                                }
                            )
                            
                            if "success" in result and result["success"]:
                                st.success(f"✅ Task assigned to {len(selected_recruiters)} recruiters")
                                st.toast(f"✅ Task assigned to {len(selected_recruiters)} recruiters!", icon="✅")
                                get_admin_dashboard_data.clear()
                                st.rerun()
                            else:
                                st.error(f"❌ Error: {result.get('message', result.get('error'))}")
                                st.toast("❌ Task assignment failed!", icon="⚠️")
            else:
                st.success("✅ All tasks are already assigned!")
        
        else: # Create & Assign New Task
            st.write("Type a task details to assign it directly")
            
            with st.form("create_assign_form"):
                col_row1_1, col_row1_2, col_row1_3 = st.columns([2, 2, 1])
                with col_row1_1:
                    task_title = st.text_input("Task Title", placeholder="e.g., Review Frontend CVs")
                with col_row1_2:
                    selected_recruiters = st.multiselect(
                        "Assign to Recruiters",
                        options=recruiters,
                        format_func=lambda x: x["name"],
                        placeholder="Select one or more..."
                    )
                with col_row1_3:
                    priority = st.selectbox("Priority", ["High", "Medium", "Low"], index=1)
                
                col_row2_1, col_row2_2 = st.columns([2, 1])
                with col_row2_1:
                    comment = st.text_area("Comment / Instructions", placeholder="Add specific details or steps for the recruiters...", height=100)
                with col_row2_2:
                    location = st.text_input("Location", value="Remote")
                
                submit = st.form_submit_button("🚀 Create & Assign", use_container_width=True, type="primary")
                
                if submit:
                    if not task_title:
                        st.error("❌ Task title is required")
                    elif not selected_recruiters:
                        st.error("❌ Please select at least one recruiter")
                    else:
                        result = api_request(
                            "POST",
                            "/api/admin/create-assign-task",
                            data={
                                "title": task_title,
                                "recruiter_ids": [r["id"] for r in selected_recruiters],
                                "priority": priority,
                                "location": location,
                                "comment": comment
                            }
                        )
                        
                        if "success" in result and result["success"]:
                            st.success(f"✅ Task created and assigned to {len(selected_recruiters)} recruiters")
                            st.toast("✅ Task created and assigned!", icon="🚀")
                            get_admin_dashboard_data.clear()
                            # st.rerun()
                        else:
                            st.error(f"❌ Error: {result.get('message', result.get('error'))}")
                            st.toast("❌ Failed to create/assign task!", icon="⚠️")

# ============================================================
# ACTIVITY LOGS PAGE
# ============================================================

def show_activity_logs():
    """Show activity logs"""
    st.markdown("<h1 class='main-header'>🕐 Activity Logs - Login/Logout History</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        days_filter = st.slider("Show logs from last X days", 1, 90, 7)
    
    with col2:
        recruiter_filter = st.text_input("Filter by recruiter name (optional)", placeholder="Type name...")
    
    with col3:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
    
    # Get logs
    params = {"days": days_filter}
    logs = api_request("GET", "/api/admin/login-logs", params=params)
    
    if "error" in logs:
        st.error(f"❌ Error: {logs['error']}")
        return
    
    log_data = []
    for log in logs:
        log_data.append({
            "Recruiter": log.get("recruiter_name", "Unknown"),
            "Login Time": log.get("login_time", "")[:19],
            "Logout Time": log.get("logout_time", "Still logged in")[:19] if log.get("logout_time") != "Still logged in" else "Still logged in",
            "Duration (min)": str(log.get("duration_minutes", "N/A")),
            "Status": log.get("status", "N/A")
        })
    
    if log_data:
        df = pd.DataFrame(log_data)
        
        if recruiter_filter:
            df = df[df["Recruiter"].str.contains(recruiter_filter, case=False, na=False)]
        
        st.subheader(f"📋 Login/Logout Logs ({len(df)} records)")
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Export
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"activity_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No activity logs found for the selected period.")

# ============================================================
# WORKLOAD REPORT PAGE
# ============================================================

def show_workload_report():
    """Show detailed workload report"""
    st.markdown("<h1 class='main-header'>📈 Workload Report</h1>", unsafe_allow_html=True)
    
    report = api_request("GET", "/api/admin/workload-report")
    
    if "error" in report:
        st.error(f"❌ Error: {report['error']}")
        return
    
    recruiters = report.get("recruiters", [])
    summary = report.get("summary", {})
    
    # Summary
    st.subheader("📊 Team Workload Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Recruiters", len(recruiters))
    with col2:
        st.metric("Avg Workload", f"{summary.get('avg_workload', 0):.0f}%")
    with col3:
        st.metric("Max Workload", f"{summary.get('max_workload', 0):.0f}%")
    with col4:
        st.metric("Min Workload", f"{summary.get('min_workload', 0):.0f}%")
    
    st.divider()
    
    # Detailed table
    st.subheader("📋 Detailed Workload by Recruiter")
    
    report_data = []
    for r in recruiters:
        report_data.append({
            "Recruiter": r["recruiter_name"],
            "Status": "🟢 Online" if r["current_status"] == "online" else "🔴 Offline",
            "Total": r["total_tasks"],
            "Pending": r["pending_tasks"],
            "Progress": r["in_progress_tasks"],
            "Completed": r["completed_tasks"],
            "Workload %": f"{r['workload_percentage']:.1f}%",
            "Avg Time (h)": f"{r['avg_completion_hours']:.1f}",
            "High Priority": r["high_priority_count"]
        })
    
    df = pd.DataFrame(report_data)
    df["Workload_sort"] = df["Workload %"].str.rstrip("%").astype(float)
    df = df.sort_values("Workload_sort", ascending=False).drop("Workload_sort", axis=1)
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Export
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Report",
        data=csv,
        file_name=f"workload_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    st.divider()
    
    # Recommendations
    st.subheader("💡 Smart Recommendations")
    
    high_workload_count = len([r for r in recruiters if r["workload_percentage"] > 70])
    low_workload_count = len([r for r in recruiters if r["workload_percentage"] < 30])
    
    if high_workload_count > 0:
        st.warning(f"⚠️ {high_workload_count} recruiter(s) with HIGH workload (>70%). Consider task redistribution.")
    
    if low_workload_count > 0:
        st.info(f"ℹ️ {low_workload_count} recruiter(s) with LOW workload (<30%). Assign more tasks to optimize.")
    
    avg_workload = summary.get("avg_workload", 0)
    if avg_workload > 70:
        st.error("🚨 Team workload is CRITICAL. Consider hiring or load balancing.")
    elif avg_workload > 50:
        st.warning("⚠️ Team workload is MODERATE-HIGH. Monitor closely.")
    else:
        st.success("✅ Team workload is HEALTHY and BALANCED.")

# ============================================================
# SETTINGS PAGE
# ============================================================

def show_settings():
    """Show settings page"""
    st.markdown("<h1 class='main-header'>⚙️ Admin Settings</h1>", unsafe_allow_html=True)
    
    st.subheader("🔧 Your Admin Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Username:** {st.session_state.admin_user.get('username', 'Unknown')}")
        st.write(f"**Email:** {st.session_state.admin_user.get('email', 'Unknown')}")
    
    with col2:
        st.write(f"**Role:** Admin")
        st.write(f"**Status:** ✅ Active")
    
    st.divider()
    
    st.subheader("📊 System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        ### Workload Monitoring
        - **Real-time Updates:** Every 30 seconds
        - **Alert Threshold:** 70% workload
        - **High Priority Weight:** 15 points per task
        - **Normal Task Weight:** 10 points per task
        """)
    
    with col2:
        st.write("""
        ### Data Retention
        - **Login Logs:** 90 days
        - **Activity History:** 60 days
        - **Task History:** Unlimited
        - **Workload Snapshots:** 30 days
        """)
    
    st.divider()
    
    st.subheader("ℹ️ System Information")
    
    st.write(f"""
    - **Admin Dashboard Version:** 2.0
    - **Last Updated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    - **Backend API:** {API_BASE_URL}
    - **Authentication:** JWT Token-based
    - **Database:** MongoDB
    """)

def parse_resume(file) -> Dict:
    """Parse resume and extract candidate details"""
    try:
        file.seek(0)
        files = {"file": file}
        return api_request("POST", "/api/parse-resume", files=files)
    except Exception as e:
        return {"error": f"Resume parsing failed: {str(e)}"}


def edit_candidate(candidate_id: str, update_data: Dict) -> Dict:
    """Update candidate details"""
    return api_request("PUT", f"/api/candidates/{candidate_id}/edit", data=update_data)


def get_candidate_resume(candidate_id: str) -> Dict:
    """Get candidate resume data"""
    return api_request("GET", f"/api/candidates/{candidate_id}/resume")


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

def show_recruiter_login():
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
                            st.toast(f"✅ Welcome back, {username}!", icon="🎯")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials")
                            st.toast("❌ Login failed. Check credentials.", icon="⚠️")
                else:
                    st.warning("⚠️ Enter username and password")
            
            st.divider()
        
        with tab2:
            st.write("Create account")
            new_user = st.text_input("Username", key="reg_user")
            new_email = st.text_input("Email Address", key="reg_email", placeholder="your@email.com")
            new_pass = st.text_input("Password", type="password", key="reg_pass")
            conf_pass = st.text_input("Confirm", type="password", key="reg_conf")
            
            if st.button("📝 Register", use_container_width=True):
                if not new_user or not new_pass or not new_email:
                    st.warning("⚠️ Fill all fields")
                elif new_pass != conf_pass:
                    st.warning("⚠️ Passwords don't match")
                else:
                    with st.spinner("Registering..."):
                        if register(new_user, new_pass, new_email):
                            st.success("✅ Registered!")
                            st.toast("✅ Account created successfully!", icon="📝")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Registration failed")
                            st.toast("❌ Registration failed.", icon="⚠️")


def show_recruiter_dashboard():
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
                        
                        if task_dict.get('comment'):
                            with st.expander("📝 View Instructions/Comments"):
                                st.write(task_dict.get('comment'))
                    
                    with col_actions:
                        task_id = task_dict.get('id', '')
                        if task_id:
                            if st.button("✅", key=f"complete_{task_id}", help="Complete task"):
                                result = complete_task(task_id)
                                if result.get("success"):
                                    st.success("✅ Task completed and removed from active tasks!")
                                    st.toast("✅ Task completed!", icon="✅")
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
                            
                            # Tabbed interface for Add Candidate
                            add_tab1, add_tab2 = st.tabs(["📄 Autofill with Resume", "✍️ Manual Fill"])
                            
                            # TAB 1: Autofill with Resume
                            with add_tab1:
                                st.info("🤖 Upload a resume and AI will extract candidate details automatically!")
                                
                                resume_file = st.file_uploader(
                                    "Upload Resume (PDF, DOCX, TXT)",
                                    type=["pdf", "docx", "txt"],
                                    key=f"resume_upload_{task_id}"
                                )
                                
                                # Initialize session state for parsed data
                                parsed_key = f"parsed_resume_{task_id}"
                                if parsed_key not in st.session_state:
                                    st.session_state[parsed_key] = None
                                
                                if resume_file:
                                    if st.button("🔍 Parse Resume", key=f"parse_btn_{task_id}", use_container_width=True):
                                        with st.spinner("🤖 Extracting candidate details from resume..."):
                                            parse_result = parse_resume(resume_file)
                                        
                                        if "error" not in parse_result and parse_result.get("success"):
                                            st.session_state[parsed_key] = parse_result
                                            st.success("✅ Resume parsed successfully!")
                                            st.toast("✅ Resume parsed!", icon="📄")
                                        else:
                                            st.error(f"❌ Failed to parse resume: {parse_result.get('error', 'Unknown error')}")
                                
                                # If resume is parsed, show editable form with extracted data
                                if st.session_state.get(parsed_key):
                                    parsed = st.session_state[parsed_key]
                                    extracted = parsed.get("extracted_data", {})
                                    
                                    st.divider()
                                    st.write("📝 **Review & Edit Extracted Details:**")
                                    
                                    with st.form(f"autofill_form_{task_id}", border=True):
                                        col_af1, col_af2 = st.columns(2)
                                        
                                        with col_af1:
                                            af_name = st.text_input("Name *", value=extracted.get("name", ""), key=f"af_name_{task_id}")
                                            af_email = st.text_input("Email *", value=extracted.get("email", ""), key=f"af_email_{task_id}")
                                            af_phone = st.text_input("Phone *", value=extracted.get("phone", ""), key=f"af_phone_{task_id}")
                                        
                                        with col_af2:
                                            af_exp = st.number_input("Experience (Years)", min_value=0, max_value=60, 
                                                                    value=int(extracted.get("experience_years", 0) or 0), key=f"af_exp_{task_id}")
                                            af_company = st.text_input("Current Company", value=extracted.get("current_company", ""), key=f"af_company_{task_id}")
                                            af_position = st.text_input("Current Position", value=extracted.get("current_position", ""), key=f"af_position_{task_id}")
                                        
                                        # Handle skills which might be list
                                        extracted_skills = extracted.get("skills", [])
                                        all_skills_options = list(set(extracted_skills + (skills if skills else [])))
                                        af_skills = st.multiselect("Skills", options=all_skills_options if all_skills_options else ["Python"], 
                                                                   default=extracted_skills if extracted_skills else [], key=f"af_skills_{task_id}")
                                        
                                        af_notes = st.text_area("Notes", value=extracted.get("notes", ""), height=80, key=f"af_notes_{task_id}")
                                        af_status = st.selectbox("Initial Status", 
                                                                ["Applied", "Shortlisted", "Interview Scheduled", "Interviewed", "Offer Extended", "Rejected"],
                                                                index=0, key=f"af_status_{task_id}")
                                        
                                        af_submitted = st.form_submit_button("➕ Add Candidate with Resume", use_container_width=True, type="primary")
                                        
                                        if af_submitted:
                                            if not af_name or not af_email or not af_phone:
                                                st.error("❌ Name, Email, and Phone are required!")
                                            else:
                                                candidate_data = {
                                                    "name": af_name,
                                                    "email": af_email,
                                                    "phone": af_phone,
                                                    "experience_years": str(af_exp),
                                                    "current_company": af_company if af_company else "N/A",
                                                    "current_position": af_position if af_position else "N/A",
                                                    "skills": af_skills if af_skills else [],
                                                    "notes": af_notes if af_notes else "",
                                                    "status": af_status,
                                                    "resume_data": parsed.get("resume_data"),
                                                    "resume_filename": parsed.get("resume_filename")
                                                }
                                                
                                                result = add_candidate(task_id, candidate_data)
                                                
                                                if result.get("success"):
                                                    st.success(f"✅ Candidate {af_name} added with resume!")
                                                    st.toast(f"✅ Candidate {af_name} added!", icon="👥")
                                                    st.session_state[parsed_key] = None
                                                    time.sleep(1)
                                                    st.rerun()
                                                else:
                                                    st.error(f"❌ Failed: {result.get('error', 'Unknown error')}")

                            # TAB 2: Manual Fill
                            with add_tab2:
                                with st.form(f"candidate_form_{task_id}", border=True):
                                    st.write("📝 **Candidate Details:**")
                                    
                                    col_c1, col_c2 = st.columns(2)
                                    
                                    with col_c1:
                                        cand_name = st.text_input("Candidate Name *", placeholder="Full name", key=f"cand_name_{task_id}")
                                        cand_email = st.text_input("Email Address *", placeholder="candidate@email.com", key=f"cand_email_{task_id}")
                                        cand_phone = st.text_input("Phone Number *", placeholder="+91-1234567890", key=f"cand_phone_{task_id}")
                                    
                                    with col_c2:
                                        cand_exp = st.number_input("Years of Experience", min_value=0, max_value=60, value=0, key=f"cand_exp_{task_id}")
                                        cand_company = st.text_input("Current Company", placeholder="Company name", key=f"cand_company_{task_id}")
                                        cand_position = st.text_input("Current Position", placeholder="Job title", key=f"cand_position_{task_id}")
                                    
                                    cand_skills = st.multiselect("Candidate Skills",
                                        options=skills if skills else ["Python", "JavaScript", "Java", "C++", "AWS", "Azure"],
                                        key=f"cand_skills_{task_id}")
                                    
                                    # Manual Resume Upload
                                    cand_resume = st.file_uploader("Upload Resume (Optional)", type=["pdf", "docx", "txt"], key=f"manual_resume_{task_id}")
                                    
                                    cand_notes = st.text_area("Additional Notes", placeholder="Notes...", height=80, key=f"cand_notes_{task_id}")
                                    cand_status = st.selectbox("Initial Status",
                                        ["Applied", "Shortlisted", "Interview Scheduled", "Interviewed", "Offer Extended", "Rejected"],
                                        index=0, key=f"cand_status_{task_id}")
                                    
                                    submitted = st.form_submit_button("➕ Add Candidate", use_container_width=True, type="primary")
                                    
                                    if submitted:
                                        if not cand_name or not cand_email or not cand_phone:
                                            st.error("❌ Name, Email, and Phone are required!")
                                        else:
                                            # Handle optional resume
                                            resume_data_manual = None
                                            resume_filename_manual = None
                                            if cand_resume:
                                                import base64
                                                cand_resume.seek(0)
                                                resume_data_manual = base64.b64encode(cand_resume.read()).decode('utf-8')
                                                resume_filename_manual = cand_resume.name

                                            candidate_data = {
                                                "name": cand_name,
                                                "email": cand_email,
                                                "phone": cand_phone,
                                                "experience_years": str(cand_exp),
                                                "current_company": cand_company if cand_company else "N/A",
                                                "current_position": cand_position if cand_position else "N/A",
                                                "skills": cand_skills if cand_skills else [],
                                                "notes": cand_notes if cand_notes else "",
                                                "status": cand_status,
                                                "resume_data": resume_data_manual,
                                                "resume_filename": resume_filename_manual
                                            }
                                            
                                            result = add_candidate(task_id, candidate_data)
                                            
                                            if result.get("success"):
                                                st.success(f"✅ Candidate {cand_name} Added Successfully!")
                                                st.toast(f"✅ Candidate {cand_name} added!", icon="👥")
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
                                
                                # Status mapping
                                reverse_status_map = {
                                    "applied": "Applied",
                                    "shortlisted": "Shortlisted",
                                    "rejected": "Rejected",
                                    "hired": "Offer Extended",
                                    "interview scheduled": "Interview Scheduled",
                                    "interviewed": "Interviewed"
                                }
                                
                                for idx, candidate in enumerate(candidates_list):
                                    st.markdown("---")
                                    display_status = reverse_status_map.get(candidate.get('status', 'applied').lower(), candidate.get('status', 'Applied'))
                                    
                                    # Header with name and edit button
                                    col_header, col_btns = st.columns([3, 1])
                                    with col_header:
                                        st.markdown(f"### 👤 {candidate.get('name', 'N/A')} ({display_status})")
                                    
                                    with col_btns:
                                        edit_key = f"edit_mode_{candidate.get('id', idx)}"
                                        if edit_key not in st.session_state:
                                            st.session_state[edit_key] = False
                                        
                                        if st.button("✏️ Edit", key=f"edit_btn_{candidate.get('id', idx)}", use_container_width=True):
                                            st.session_state[edit_key] = not st.session_state[edit_key]
                                            st.rerun()
                                        
                                        if candidate.get('resume_filename'):
                                            view_key = f"view_resume_{candidate.get('id', idx)}"
                                            is_viewing = st.session_state.get(view_key, False)
                                            btn_label = "❌ Close Resume" if is_viewing else "📄 View Resume"
                                            
                                            if st.button(btn_label, key=f"view_res_btn_{candidate.get('id', idx)}", use_container_width=True):
                                                st.session_state[view_key] = not is_viewing
                                                st.rerun()

                                    # EDIT MODE
                                    if st.session_state.get(edit_key, False):
                                        with st.form(f"edit_form_{candidate.get('id', idx)}", border=True):
                                            st.write("✏️ **Edit Candidate Details:**")
                                            
                                            col_e1, col_e2 = st.columns(2)
                                            with col_e1:
                                                edit_name = st.text_input("Name", value=candidate.get('name', ''), key=f"edit_name_{candidate.get('id', idx)}")
                                                edit_email = st.text_input("Email", value=candidate.get('email', ''), key=f"edit_email_{candidate.get('id', idx)}")
                                                edit_phone = st.text_input("Phone", value=candidate.get('phone', ''), key=f"edit_phone_{candidate.get('id', idx)}")
                                            
                                            with col_e2:
                                                edit_exp = st.number_input("Experience", min_value=0, max_value=60, 
                                                                           value=int(float(candidate.get('experience_years', 0) if candidate.get('experience_years') else 0)), 
                                                                           key=f"edit_exp_{candidate.get('id', idx)}")
                                                edit_company = st.text_input("Company", value=candidate.get('current_company', ''), key=f"edit_company_{candidate.get('id', idx)}")
                                                edit_position = st.text_input("Position", value=candidate.get('current_position', ''), key=f"edit_position_{candidate.get('id', idx)}")
                                            
                                            edit_skills = st.multiselect("Skills", 
                                                options=list(set(candidate.get('skills', []) + (skills if skills else []))),
                                                default=candidate.get('skills', []),
                                                key=f"edit_skills_{candidate.get('id', idx)}")
                                            
                                            edit_notes = st.text_area("Notes", value=candidate.get('notes', ''), key=f"edit_notes_{candidate.get('id', idx)}")
                                            
                                            edit_resume = st.file_uploader("Replace Resume (Optional)", type=["pdf", "docx", "txt"], key=f"edit_resume_{candidate.get('id', idx)}")
                                            
                                            col_save, col_cancel = st.columns(2)
                                            with col_save:
                                                save_btn = st.form_submit_button("💾 Save Changes", use_container_width=True, type="primary")
                                            with col_cancel:
                                                cancel_btn = st.form_submit_button("❌ Cancel", use_container_width=True)
                                            
                                            if save_btn:
                                                update_data = {
                                                    "name": edit_name,
                                                    "email": edit_email,
                                                    "phone": edit_phone,
                                                    "experience_years": str(edit_exp),
                                                    "current_company": edit_company,
                                                    "current_position": edit_position,
                                                    "skills": edit_skills,
                                                    "notes": edit_notes,
                                                }
                                                
                                                if edit_resume:
                                                    import base64
                                                    edit_resume.seek(0)
                                                    update_data["resume_data"] = base64.b64encode(edit_resume.read()).decode('utf-8')
                                                    update_data["resume_filename"] = edit_resume.name
                                                
                                                result = edit_candidate(candidate.get('id'), update_data)
                                                if result.get("success"):
                                                    st.success("✅ Candidate updated!")
                                                    st.session_state[edit_key] = False
                                                    time.sleep(0.5)
                                                    st.rerun()
                                                else:
                                                    st.error(f"❌ Failed: {result.get('error', 'Unknown')}")
                                            
                                            if cancel_btn:
                                                st.session_state[edit_key] = False
                                                st.rerun()

                                    # DISPLAY MODE
                                    else:
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
                                            if candidate.get('resume_filename'):
                                                st.write(f"**Resume:** 📄 {candidate.get('resume_filename')}")

                                        if candidate.get('skills'):
                                            st.write(f"**Skills:** {', '.join(candidate.get('skills', []))}")

                                        if candidate.get('notes'):
                                            st.write(f"**Notes:** {candidate.get('notes', '')}")

                                        col_act1, col_act2, col_act3 = st.columns([2, 1, 1])

                                        with col_act1:
                                            status_options = ["applied", "shortlisted", "rejected", "hired", "interview scheduled", "interviewed"]
                                            current_status = candidate.get('status', 'applied').lower()
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
                                                    st.success(f"✅ Status updated!")
                                                    time.sleep(0.5)
                                                    st.rerun()
                                                else:
                                                    st.error(f"❌ Failed to update status")

                                        with col_act2:
                                            pass
                                        
                                        with col_act3:
                                            if st.button("🗑️ Delete", key=f"delete_cand_{candidate.get('id', idx)}", use_container_width=True):
                                                delete_result = delete_candidate(candidate.get('id'))
                                                if delete_result.get("success"):
                                                    st.success("✅ deleted")
                                                    time.sleep(0.5)
                                                    st.rerun()
                                                else:
                                                    st.error("❌ Failed")

                                        # Resume Viewer Section (Full width)
                                        view_key = f"view_resume_{candidate.get('id', idx)}"
                                        if st.session_state.get(view_key):
                                            st.markdown("---")
                                            st.info(f"📄 Viewing Resume: {candidate.get('resume_filename')}")
                                            
                                            with st.spinner("Fetching resume..."):
                                                resume_res = get_candidate_resume(candidate.get('id'))
                                                
                                            if resume_res.get("success"):
                                                b64_data = resume_res.get("resume_data")
                                                fname = resume_res.get("resume_filename", "").lower()
                                                
                                                # Display PDF
                                                if fname.endswith(".pdf"):
                                                    pdf_display = f'<iframe src="data:application/pdf;base64,{b64_data}" width="100%" height="800" type="application/pdf"></iframe>'
                                                    st.markdown(pdf_display, unsafe_allow_html=True)
                                                else:
                                                    st.warning("⚠️ Preview only available for PDF files.")
                                                
                                                # Download Button
                                                import base64
                                                try:
                                                    b_data = base64.b64decode(b64_data)
                                                    st.download_button(
                                                        label="⬇️ Download Resume",
                                                        data=b_data,
                                                        file_name=fname,
                                                        mime="application/octet-stream",
                                                        key=f"dl_res_{candidate.get('id', idx)}"
                                                    )
                                                except Exception as e:
                                                    st.error(f"Error preparing download: {e}")
                                            else:
                                                st.error("❌ Failed to fetch resume data")
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
                                            st.toast("✅ Task completed!", icon="✅")
                                            time.sleep(1)
                                            st.rerun()
                            
                            with col_delete:
                                if st.button("🗑️", key=f"delete_{task_id}", help="Delete"):
                                    result = delete_task(task_id)
                                    if result.get("success"):
                                        st.success("✅ Task deleted!")
                                        st.toast("✅ Task deleted!", icon="🗑️")
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
        
        # MERGED JD UPLOAD SECTION
        st.info("🤖 **AI Agent Extraction:** Upload one or more JD files (PDF, TXT, DOCX).")
        
        uploaded_files = st.file_uploader(
            "Choose JD file(s)",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            key="req_files"
        )
        
        st.divider()
        
        # PASTE TEXT SECTION
        st.markdown("### 📝 Paste Content")
        text = st.text_area(
            "Paste job description if not uploading files:",
            height=200,
            placeholder="Job Title: ...\nResponsibilities: ...\nRequired Skills: ...",
            key="req_text"
        )
        
        st.divider()
        
        # Unified Extract Button
        if st.button("🔍 Extract & Create Task(s)", use_container_width=True, key="extract_btn"):
            if text.strip() and uploaded_files:
                st.warning("⚠️ Please use either paste OR upload, not both")
            elif text.strip():
                with st.spinner("Analyzing text..."):
                    extraction_result = extract_requirement(text)
                    source_type = "text"
            elif uploaded_files:
                if len(uploaded_files) > 1:
                    with st.spinner(f"🤖 Agent analyzing {len(uploaded_files)} JDs using NER extraction..."):
                        extraction_result = upload_multiple_jds(uploaded_files)
                        source_type = "multi_file"
                else:
                    with st.spinner("Processing file..."):
                        extraction_result = upload_file(uploaded_files[0])
                        source_type = "file"
            else:
                st.warning("⚠️ Please paste text or upload file(s)")
        
        # Display Results
        if extraction_result:
            # Handle Multi-File Result
            if source_type == "multi_file":
                if extraction_result.get("success"):
                    st.success(f"✅ {extraction_result.get('message', 'JDs processed successfully!')}")
                    st.toast("✅ All JDs processed!", icon="📦")
                    st.info("📁 **Tasks have been saved to MongoDB** - They will persist even after logout!")
                    
                    st.subheader("📋 Extraction Results (Saved to Database)")
                    for item in extraction_result.get("results", []):
                        item_result = item.get("result", {})
                        if "error" not in item_result:
                            extracted = item_result.get('extracted_data', {})
                            with st.expander(f"✅ {item.get('filename')} → {extracted.get('title', 'Task')}", expanded=False):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**📌 Priority:** {extracted.get('priority', 'N/A')}")
                                    st.write(f"**⏰ Urgency:** {extracted.get('urgency', 'N/A')}")
                                    st.write(f"**📍 Location:** {extracted.get('location', 'N/A')}")
                                with col2:
                                    st.write(f"**📊 Complexity:** {extracted.get('complexity', 'N/A')}")
                                    st.write(f"**💼 Experience:** {extracted.get('experience', 'N/A')}")
                                    st.write(f"**🆔 Task ID:** `{item_result.get('task_id', 'N/A')}`")
                        else:
                            st.error(f"❌ {item.get('filename')} → {item_result.get('error', 'Unknown error')}")
                    st.divider()
                else:
                    st.error(f"Error: {extraction_result.get('error', 'Unknown error')}")
            
            # Handle Single File/Text Result
            elif "error" not in extraction_result:
                st.success("✅ Extracted & Task Created Successfully!")
                st.toast("✅ Task extracted successfully!", icon="🔍")
                
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
                st.toast("❌ Extraction failed!", icon="❌")
                
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
        
        # Form to schedule interviews (Removed st.form to allow interactive buttons)
        st.write("### 📅 Schedule New Interview")
        
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
        if st.button("🔍 Get Slot Suggestions", key="get_slots_btn"):
            suggestions = api_request("GET", f"/api/suggest-slots?date={schedule_date}")
            if "time_slots" in suggestions:
                st.session_state.suggested_slots = suggestions["time_slots"]
                st.success("✅ Slots analyzed based on your calendar")
        
        # Show slots if available
        col_slots, col_custom = st.columns([3, 1])
        with col_slots:
            interview_time = st.selectbox(
                "Select Time Slot",
                options=[s['slot'] for s in st.session_state.get('suggested_slots', [])],
                placeholder="Choose a slot..."
            )
        
        with col_custom:
            if st.button("⏰ Customize", help="Set a custom time"):
                st.session_state.show_custom_time = not st.session_state.get("show_custom_time", False)
        
        if st.session_state.get("show_custom_time", False):
            with st.container(border=True):
                st.markdown("<div class='time-picker-label'>Select Time</div>", unsafe_allow_html=True)
                
                # Hour, Minute, AM/PM Columns
                col_h, col_sep, col_m, col_p = st.columns([2, 0.5, 2, 2])
                
                # Initialize session state for the picker
                if "custom_time_h" not in st.session_state: st.session_state.custom_time_h = 9
                if "custom_time_m" not in st.session_state: st.session_state.custom_time_m = "00"
                if "custom_time_p" not in st.session_state: st.session_state.custom_time_p = "am"
                
                with col_h:
                    h_options = [i for i in range(1, 13)]
                    try:
                        h_idx = h_options.index(st.session_state.custom_time_h)
                    except:
                        h_idx = 8
                    st.session_state.custom_time_h = st.selectbox("H", options=h_options, 
                        index=h_idx, label_visibility="collapsed", key="picker_h")
                
                with col_sep:
                    st.markdown("<h2 style='text-align: center; margin-top: 0;'>:</h2>", unsafe_allow_html=True)
                
                with col_m:
                    m_options = [f"{i:02d}" for i in range(0, 60, 5)] # 5-min increments
                    try:
                        m_idx = m_options.index(st.session_state.custom_time_m)
                    except:
                        m_idx = 0
                    st.session_state.custom_time_m = st.selectbox("M", options=m_options, 
                        index=m_idx, label_visibility="collapsed", key="picker_m")
                
                with col_p:
                    p_options = ["am", "pm"]
                    try:
                        p_idx = p_options.index(st.session_state.custom_time_p)
                    except:
                        p_idx = 0
                    st.session_state.custom_time_p = st.selectbox("P", options=p_options, 
                        index=p_idx, label_visibility="collapsed", key="picker_p")
                
                st.markdown("<div style='margin-top: 20px; font-weight: bold;'>Presets</div>", unsafe_allow_html=True)
                col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                
                # Preset logic
                def set_preset(h, m, p):
                    st.session_state.custom_time_h = h
                    st.session_state.custom_time_m = m
                    st.session_state.custom_time_p = p
                    st.rerun()

                if col_p1.button("9 am", use_container_width=True, key="p_9am"): set_preset(9, "00", "am")
                if col_p2.button("12 pm", use_container_width=True, key="p_12pm"): set_preset(12, "00", "pm")
                if col_p3.button("4 pm", use_container_width=True, key="p_4pm"): set_preset(4, "00", "pm")
                if col_p4.button("6 pm", use_container_width=True, key="p_6pm"): set_preset(6, "00", "pm")

                st.markdown("<div class='done-btn-container'></div>", unsafe_allow_html=True)
                if st.button("Done", use_container_width=True, type="primary", key="picker_done"):
                    custom_time_str = f"{st.session_state.custom_time_h}:{st.session_state.custom_time_m} {st.session_state.custom_time_p}"
                    # Calculate end time (assuming 1h duration)
                    h_val = st.session_state.custom_time_h
                    end_h = (h_val % 12) + 1
                    end_p = st.session_state.custom_time_p
                    if h_val == 11:
                        end_p = "pm" if st.session_state.custom_time_p == "am" else "am"
                    
                    custom_end_str = f"{end_h}:{st.session_state.custom_time_m} {end_p}"
                    new_slot = {"slot": f"{custom_time_str} - {custom_end_str} (Custom)"}
                    
                    if 'suggested_slots' not in st.session_state:
                        st.session_state.suggested_slots = []
                    st.session_state.suggested_slots.append(new_slot)
                    st.session_state.show_custom_time = False
                    st.success(f"✅ Custom slot {custom_time_str} added!")
                    st.rerun()
        
        location_type = st.selectbox("Location", ["Virtual", "In-Person"])
        meeting_link = st.text_input("Meeting Link (e.g., Zoom/Meet)")
        
        if st.button("✉️ Send Invites to Selected Candidates", key="send_invites_btn"):
            if selected_task_id and schedule_date and interview_time and selected_candidate_ids:
                payload = {
                    "task_id": selected_task_id,
                    "candidate_ids": selected_candidate_ids,
                    "interview_date": str(schedule_date),
                    "interview_time": interview_time,
                    "location": location_type,
                    "meeting_link": meeting_link
                }
                
                with st.spinner("🤖 Scheduling Agent is sending emails..."):
                    result = api_request("POST", "/api/schedule-interview", payload)
                    
                if result.get("success"):
                    st.success(f"✅ {result.get('message')}")
                    st.toast("✅ Interview invites sent!", icon="📅")
                    if result.get("sent_details"):
                        with st.expander("Email Details"):
                            st.json(result.get("sent_details"))
                else:
                    st.error(f"❌ Error: {result.get('error', 'Unknown Error')}")
                    st.toast("❌ Scheduling failed!", icon="⚠️")
            else:
                st.warning("⚠️ Please select all fields (Task, Date, Time, Candidates)")

    # PAGE 5: Workload & EOD
    if selected_page == "📊 Workload & EOD":
        st.subheader("📊 EOD & Workload Agent")
        st.info("Monitor daily progress and generate end-of-day reports.")
        
        if st.button("📝 Generate Comprehensive EOD Report"):
            with st.spinner("🤖 Analysing daily activities..."):
                result = api_request("POST", "/api/eod-summary", {})
                
            if result.get("email_sent"):
                st.success(f"✅ EOD Report Sent to {result.get('recipient')}")
                st.toast("✅ EOD email sent!", icon="📧")
                with st.expander("📄 View Report", expanded=True):
                    st.text(result.get("summary"))
            else:
                if "summary" in result:
                    st.warning("⚠️ Report generated but email failed")
                    st.toast("⚠️ Report generated, email failed.", icon="⚠️")
                    with st.expander("📄 View Report", expanded=True):
                        st.text(result.get("summary"))
                else:
                    st.error("❌ Failed to generate EOD summary")
                    st.toast("❌ EOD report failed!", icon="⚠️")
        
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
        health = requests.get(f"{API_BASE_URL}/health", timeout=15)
        if health.status_code != 200:
            st.error("❌ Backend not responding")
            st.stop()
    except Exception as e:
        st.error(f"❌ Cannot connect: {str(e)}")
        st.error("Start backend: python -m uvicorn app:app --reload")
        st.stop()
    
    # Determine which login/dashboard to show
    if st.session_state.get("admin_token"):
        show_admin_dashboard()
    elif st.session_state.token:
        show_recruiter_dashboard()
    else:
        # Default to recruiter login with a toggle for admin
        role = st.sidebar.radio("View Mode", ["Recruiter", "Admin"], index=0)
        if role == "Admin":
            show_admin_login()
        else:
            show_recruiter_login()


if __name__ == "__main__":
    main()

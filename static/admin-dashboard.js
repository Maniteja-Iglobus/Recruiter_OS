/* ============================================
   Recruiter OS - Main Application Logic
   ============================================ */

// Robust API Base URL detection
// Default to http://localhost:8000 if running from file:// or a different port (e.g. VSCode Live Server)
const API_BASE = (window.location.protocol === 'http:' || window.location.protocol === 'https:')
    ? window.location.origin
    : 'http://localhost:8000';

console.log('🔗 API_BASE set to:', API_BASE);

const CHAT_API = API_BASE;

// ---- State ----
let currentUserRole = localStorage.getItem('role') || null;
let authToken = localStorage.getItem('token') || null;
let adminToken = localStorage.getItem('admin_token') || null;

let state = {
    adminToken: adminToken,
    adminUser: JSON.parse(localStorage.getItem('admin_user') || 'null'),
    currentPage: 'dashboard',
    selectedRecruiter: null,
    dashboardData: null,
    recruiters: [],
};

// ==================== UTILITIES ====================

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    let icon = 'ℹ️';
    if (type === 'success') icon = '✅';
    if (type === 'error') icon = '❌';

    toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.remove();
    }, 3000);
}

async function apiRequest(endpoint, method = 'GET', data = null, isFormData = false) {
    const headers = {};

    // Choose token based on endpoint or current role context
    if (endpoint.startsWith('/api/admin') && adminToken) {
        headers['Authorization'] = `Bearer ${adminToken}`;
    } else if (authToken) {
        headers['Authorization'] = `Bearer ${authToken}`;
    }

    const config = {
        method,
        headers,
    };

    if (data) {
        if (isFormData) {
            config.body = data; // Content-Type header let browser set for FormData
        } else {
            headers['Content-Type'] = 'application/json';
            config.body = JSON.stringify(data);
        }
    }

    try {
        const fullUrl = `${API_BASE}${endpoint}`;
        console.log(`🌐 API Request: ${method} ${fullUrl}`);
        const response = await fetch(fullUrl, config);
        if (response.status === 401) {
            // Don't auto-logout if we are just trying to log in
            if (!endpoint.includes('/login')) {
                handleLogout();
            }
            return { error: 'Unauthorized' };
        }
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        return { error: 'Connection error' };
    }
}

// ==================== AUTHENTICATION ====================

function checkAuth() {
    if (currentUserRole === 'admin' && adminToken) {
        showDashboard();
    } else if (currentUserRole === 'recruiter' && authToken) {
        showRecruiterDashboard();
    } else {
        showLogin();
    }
}

function showLogin() {
    document.getElementById('login-page').style.display = 'flex';
    document.getElementById('dashboard-page').classList.add('hidden');
    // Default to recruiter view
    switchLoginRole('recruiter');
}

function handleLogout() {
    localStorage.clear();
    currentUserRole = null;
    authToken = null;
    adminToken = null;
    window.location.reload();
}

// Login Page UI Switchers
function switchLoginRole(role) {
    document.querySelectorAll('.role-btn').forEach(b => b.classList.remove('active'));
    document.querySelector(`.role-btn[data-role="${role}"]`).classList.add('active');

    if (role === 'admin') {
        document.getElementById('recruiter-login-panel').style.display = 'none';
        document.getElementById('admin-login-panel').style.display = 'block';
        document.getElementById('login-features-title').innerText = '🔐 ADMIN FEATURES';
        document.getElementById('login-features-list').innerHTML = `
            <li>📊 Team performance analytics</li>
            <li>👥 Manage recruiters & workloads</li>
            <li>🕐 Activity logging & auditing</li>
            <li>⚙️ System configuration</li>
        `;
    } else {
        document.getElementById('recruiter-login-panel').style.display = 'block';
        document.getElementById('admin-login-panel').style.display = 'none';
        document.getElementById('login-features-title').innerText = '🎯 RECRUITER FEATURES';
        document.getElementById('login-features-list').innerHTML = `
            <li>📋 Smart task management with AI extraction</li>
            <li>👥 Candidate tracking & resume parsing</li>
            <li>📅 Interview scheduling with automation</li>
            <li>📊 Workload monitoring & EOD reports</li>
        `;
    }
}

function switchLoginTab(tab) {
    document.querySelectorAll('.login-tab').forEach(b => b.classList.remove('active'));
    // Find button by text roughly or index
    const tabs = document.querySelectorAll('.login-tab');
    if (tab === 'login') tabs[0].classList.add('active');
    else tabs[1].classList.add('active');

    if (tab === 'login') {
        document.getElementById('recruiter-login-form-wrap').style.display = 'block';
        document.getElementById('recruiter-register-form-wrap').style.display = 'none';
    } else {
        document.getElementById('recruiter-login-form-wrap').style.display = 'none';
        document.getElementById('recruiter-register-form-wrap').style.display = 'block';
    }
}

// Sidebar Toggle
window.toggleSidebar = function () {
    const sidebar = document.getElementById('main-sidebar');
    sidebar.classList.toggle('collapsed');
    const isCollapsed = sidebar.classList.contains('collapsed');

    // Persist state
    localStorage.setItem('sidebar_collapsed', isCollapsed);
}

// Restore Sidebar State on Init
document.addEventListener('DOMContentLoaded', () => {
    const isCollapsed = localStorage.getItem('sidebar_collapsed') === 'true';
    if (isCollapsed) {
        const sidebar = document.getElementById('main-sidebar');
        if (sidebar) {
            sidebar.classList.add('collapsed');
        }
    }
});

// ==================== RECRUITER FUNCTIONS ====================

function showRecruiterDashboard() {
    document.getElementById('login-page').style.display = 'none';
    document.getElementById('dashboard-page').classList.remove('hidden');

    // Sidebar setup
    document.getElementById('admin-nav').classList.add('hidden');
    document.getElementById('recruiter-nav').classList.remove('hidden');
    document.getElementById('sidebar-role-label').innerText = 'Recruiter Workspace';

    const user = JSON.parse(localStorage.getItem('user') || '{}');
    document.getElementById('sidebar-user-name').innerText = user.username || 'Recruiter';
    document.getElementById('sidebar-user-email').innerText = user.email || 'User';
    document.getElementById('sidebar-user-avatar').innerText = (user.username || 'U')[0].toUpperCase();

    document.getElementById('rec-welcome-msg').innerText = `Welcome back, ${user.username || 'Recruiter'}! 👋`;

    // Load initial page
    navigateTo('rec-dashboard');
}

// --- Recruiter: Dashboard Overview ---
async function loadRecDashboard() {
    const container = document.getElementById('rec-dashboard-content');
    container.innerHTML = '<div class="loading-state"><div class="spinner"></div><p>Loading dashboard...</p></div>';

    const data = await apiRequest('/api/dashboard');

    if (data.error) {
        container.innerHTML = `<div class="empty-state"><p class="text-danger">Error: ${data.error}</p></div>`;
        return;
    }

    const stats = data.stats || {};
    const recent = data.recent_tasks || [];

    let tasksHtml = '';
    const activeTasks = recent.filter(t => t.status !== 'completed');

    if (activeTasks.length === 0) {
        tasksHtml = `<div class="empty-state"><div class="icon">📭</div><p>No active tasks. Use "Extract & Upload" to create one!</p></div>`;
    } else {
        tasksHtml = activeTasks.map(task => `
            <div class="alert alert-info" style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <strong>${task.title}</strong>
                    <div style="font-size:11px; margin-top:2px; opacity:0.8;">
                        ${task.priority} Priority • ${task.urgency} • ${task.created_at.substring(0, 10)}
                    </div>
                </div>
                <button class="btn btn-sm btn-success" onclick="completeTask('${task.id}')">✅ Complete</button>
            </div>
        `).join('');
    }

    container.innerHTML = `
        <div class="metrics-grid">
            <div class="metric-card info">
                <div class="label">Total Tasks</div><div class="value">${stats.total_tasks || 0}</div>
            </div>
            <div class="metric-card warning">
                <div class="label">Pending</div><div class="value">${stats.pending || 0}</div>
            </div>
            <div class="metric-card accent">
                <div class="label">In Progress</div><div class="value">${stats.in_progress || 0}</div>
            </div>
            <div class="metric-card success">
                <div class="label">Completed</div><div class="value">${stats.completed || 0}</div>
            </div>
        </div>
        
        <div class="data-card">
            <div class="data-card-header"><h3>🔥 Active Tasks (Pending & In Progress)</h3></div>
            <div class="data-card-body" style="padding: 20px;">
                ${tasksHtml}
            </div>
        </div>
    `;
}

// --- Recruiter: All Tasks ---
let allTasksCache = [];

async function loadRecTasks() {
    const container = document.getElementById('rec-tasks-content');
    container.innerHTML = '<div class="loading-state"><div class="spinner"></div><p>Loading tasks...</p></div>';

    const tasks = await apiRequest('/api/tasks');
    if (tasks.error) {
        container.innerHTML = `<div class="empty-state"><p>Error loading tasks</p></div>`;
        return;
    }

    allTasksCache = tasks;

    // Sort tasks by date (newest first)
    tasks.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));

    renderTasksList(tasks);
}

function renderTasksList(tasks) {
    const container = document.getElementById('rec-tasks-content');

    if (!tasks || tasks.length === 0) {
        container.innerHTML = `<div class="empty-state"><div class="icon">📋</div><p>No tasks found.</p></div>`;
        return;
    }

    const pendingTasks = tasks.filter(t => t.status === 'pending');
    const inProgressTasks = tasks.filter(t => t.status === 'in_progress');
    const completedTasks = tasks.filter(t => t.status === 'completed');

    const renderTaskCard = (task) => {
        const colorClass = task.status === 'completed' ? 'success' : (task.status === 'pending' ? 'warning' : 'info');
        const priorityEmoji = task.priority === 'High' ? '🔴' : (task.priority === 'Medium' ? '🟡' : '🟢');
        const urgencyEmoji = task.urgency === 'Immediate' ? '🔴' : (task.urgency === '1 Week' ? '🟡' : '🟢');

        // Status class for better visibility
        const statusClass = `task-status-${(task.status || 'pending').replace('_', '-')}`;

        return `
            <div class="task-expander ${statusClass}" id="task-${task.id}">
                <div class="task-expander-header" onclick="toggleTask('${task.id}')">
                    <div style="display:flex; align-items:center; gap:12px;">
                        <span class="badge badge-${colorClass}">${(task.status || 'pending').toUpperCase().replace('_', ' ')}</span>
                        <span style="font-weight:600; font-size:14px;">${task.title}</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:12px; color:var(--text-muted); font-size:13px;">
                        <span>${priorityEmoji} ${task.priority || 'Medium'}</span>
                        <span>${urgencyEmoji} ${task.urgency || 'Flexible'}</span>
                        <span>${task.location || ''}</span>
                        <span class="arrow">▼</span>
                    </div>
                </div>
                <div class="task-expander-body">
                    <div class="info-grid">
                        <div>
                            <p><strong>Complexity:</strong> ${task.complexity || 'Medium'}</p>
                            <p><strong>Experience:</strong> ${task.experience || 'N/A'}</p>
                            <p><strong>Skills:</strong> ${task.skills && task.skills.length > 0 ? task.skills.join(', ') : 'N/A'}</p>
                            ${task.comment ? `<div class="mt-2 text-muted"><em>${task.comment}</em></div>` : ''}
                        </div>
                        <div style="text-align:right;">
                            <div style="display:flex; gap:8px; justify-content:flex-end; margin-bottom:10px;">
                                ${task.status !== 'completed' ? `<button class="btn btn-sm btn-success" onclick="completeTask('${task.id}')">✅ Finish Task</button>` : ''}
                                <button class="btn btn-sm btn-danger" onclick="deleteTask('${task.id}')">🗑️ Delete</button>
                            </div>
                            <small class="text-muted">Created: ${task.created_at ? task.created_at.substring(0, 10) : 'N/A'}</small>
                        </div>
                    </div>
                    
                    <div style="margin-top:20px; padding-top:20px; border-top:1px solid var(--border-light);">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                            <h4 style="font-size:14px; font-weight:700;">👥 Candidates</h4>
                            <button class="btn btn-sm btn-outline" onclick="openAddCandidateModal('${task.id}')">➕ Add Candidate</button>
                        </div>
                        <div id="candidates-list-${task.id}">
                            <div class="spinner"></div> Loading candidates...
                        </div>
                    </div>
                </div>
            </div>
        `;
    };

    container.innerHTML = `
        <div class="filters-row">
            <div class="filter-group">
                <input type="text" placeholder="🔍 Search tasks..." onkeyup="filterTasks(this.value)">
            </div>
        </div>
        
        <div id="tasks-list-container">
            ${inProgressTasks.length > 0 ? `
                <div class="section-title" style="margin: 20px 0 10px; font-weight:700; color:var(--accent);">⚙️ In Progress (${inProgressTasks.length})</div>
                ${inProgressTasks.map(renderTaskCard).join('')}
            ` : ''}

            ${pendingTasks.length > 0 ? `
                <div class="section-title" style="margin: 20px 0 10px; font-weight:700; color:var(--warning, #f59e0b);">⏳ Pending (${pendingTasks.length})</div>
                ${pendingTasks.map(renderTaskCard).join('')}
            ` : ''}

            ${inProgressTasks.length === 0 && pendingTasks.length === 0 ? '<div class="alert alert-info">📭 No active tasks. Create one by extracting a JD!</div>' : ''}

            ${completedTasks.length > 0 ? `
                <div class="section-title" style="margin: 30px 0 10px; font-weight:700; color:var(--success, #22c55e);">✅ Completed (${completedTasks.length})</div>
                ${completedTasks.map(renderTaskCard).join('')}
            ` : ''}
        </div>
    `;
}

function filterTasks(query) {
    const filtered = allTasksCache.filter(t => t.title.toLowerCase().includes(query.toLowerCase()));
    const container = document.getElementById('tasks-list-container');
    if (container) {
        // Re-render just the list part manually or simplify
        // For now, simpler to re-call render if we split it properly, but here duplicate logic
        // Let's just re-render full for simplicity
        // Note: this resets open states, but it's acceptable for MVP
        // A better way is hiding elements
        document.querySelectorAll('.task-expander').forEach(el => {
            const title = el.querySelector('.task-expander-header span:nth-child(2)').innerText.toLowerCase();
            el.style.display = title.includes(query.toLowerCase()) ? 'block' : 'none';
        });
    }
}

function toggleTask(taskId) {
    const el = document.getElementById(`task-${taskId}`);
    el.classList.toggle('open');
    if (el.classList.contains('open')) {
        loadTaskCandidates(taskId);
    }
}

async function loadTaskCandidates(taskId) {
    const container = document.getElementById(`candidates-list-${taskId}`);
    const cands = await apiRequest(`/api/tasks/${taskId}/candidates`);

    if (cands.error || cands.length === 0) {
        container.innerHTML = '<p class="text-muted" style="font-size:13px; font-style:italic;">No candidates yet.</p>';
        return;
    }

    container.innerHTML = `
        <table class="data-table">
            <thead>
                <tr>
                    <th>Name</th>
                    <th>Status</th>
                    <th>Skills & Exp</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                ${cands.map(c => `
                    <tr>
                        <td>
                            <div style="font-weight:600; font-size:13px;">${c.name}</div>
                            <div style="font-size:11px; color:var(--text-muted);">${c.email}</div>
                        </td>
                        <td>
                            <select onchange="updateCandidateStatus('${c.id}', this.value)" style="padding:4px; border:1px solid var(--border); border-radius:4px; font-size:12px;">
                                <option ${(c.status || '').toLowerCase() === 'applied' ? 'selected' : ''}>Applied</option>
                                <option ${(c.status || '').toLowerCase() === 'shortlisted' ? 'selected' : ''}>Shortlisted</option>
                                <option ${(c.status || '').toLowerCase() === 'interview scheduled' ? 'selected' : ''}>Interview Scheduled</option>
                                <option ${(c.status || '').toLowerCase() === 'rejected' ? 'selected' : ''}>Rejected</option>
                                <option ${(c.status || '').toLowerCase() === 'hired' ? 'selected' : ''}>Hired</option>
                            </select>
                        </td>
                        <td>
                             <div style="font-weight:600; font-size:13px; margin-bottom:4px;">${c.skills ? c.skills.slice(0, 3).join(', ') : 'No skills'}</div>
                             <div style="font-size:11px;">${c.experience_years}y • ${c.current_position || 'N/A'}</div>
                        </td>
                        <td>
                            <div style="display:flex; gap:6px;">
                                ${c.resume_filename ? `<button class="btn btn-sm btn-outline" onclick="viewResume('${c.id}')" title="View Resume">📄</button>` : ''}
                                <button class="btn btn-sm btn-info" onclick="openEditCandidateModal('${c.id}', '${taskId}')" title="Edit">✏️</button>
                                <button class="btn btn-sm btn-danger" onclick="deleteCandidate('${c.id}')" title="Delete">🗑️</button>
                            </div>
                        </td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

// Global functions for candidates
// Global functions for candidates
window.updateCandidateStatus = async (candId, newStatus) => {
    // Backend expects status as query param
    const res = await apiRequest(`/api/candidates/${candId}?status=${encodeURIComponent(newStatus)}`, 'PUT');
    if (res.success) {
        showToast(`Status updated to ${newStatus}`, 'success');

        // Reload dashboard to update task status (e.g. pending -> in_progress)
        loadRecDashboard();

        // Reload candidates list to show new status
        const activeExpander = document.querySelector('.task-expander.open');
        if (activeExpander) {
            const taskId = activeExpander.id.replace('task-', '');
            loadTaskCandidates(taskId);
        }
    } else {
        showToast(res.error || 'Failed to update status', 'error');
    }
};

// --- Delete Candidate ---
window.deleteCandidate = async (candidateId) => {
    if (!confirm('Are you sure you want to delete this candidate?')) return;

    // Check if within a task context to reload
    const activeExpander = document.querySelector('.task-expander.open');
    let taskId = null;
    if (activeExpander) {
        taskId = activeExpander.id.replace('task-', '');
    }

    const res = await apiRequest(`/api/candidates/${candidateId}`, 'DELETE');

    if (res.success) {
        showToast('Candidate deleted', 'success');
        if (taskId) {
            loadTaskCandidates(taskId);
        } else {
            // Fallback if somehow called outside context
            loadRecTasks();
        }
    } else {
        showToast(res.error || 'Failed to delete candidate', 'error');
    }
};

let currentEditingCandId = null;

window.openEditCandidateModal = async function (candId, taskId) {
    // Simplified fetch of full candidate details from task list cache is unreliable if not fully loaded.
    // Ideally we should cache or fetch single candidate. For now, let's just find in current list.
    const container = document.getElementById(`candidates-list-${taskId}`);
    if (!container) { showToast('Please load task candidates first', 'warning'); return; }

    // We can't easily grab the full object from DOM. Let's fetch the list again to be safe and get full data.
    const cands = await apiRequest(`/api/tasks/${taskId}/candidates`);
    const cand = cands.find(c => c.id === candId);

    if (!cand) return showToast('Candidate not found', 'error');

    currentEditingCandId = candId;

    const modal = document.getElementById('candidate-modal');
    const modalTitle = document.getElementById('candidate-modal-title');

    modal.classList.add('active');
    modalTitle.innerText = '✏️ Edit Candidate';

    document.getElementById('candidate-modal-body').innerHTML = `
        <div class="form-row">
            <div class="form-col"><label>Name</label><input id="c-edit-name" value="${cand.name || ''}"></div>
            <div class="form-col"><label>Email</label><input id="c-edit-email" value="${cand.email || ''}"></div>
        </div>
        <div class="form-row">
            <div class="form-col"><label>Phone</label><input id="c-edit-phone" value="${cand.phone || ''}"></div>
            <div class="form-col"><label>Exp (Years)</label><input type="number" id="c-edit-exp" value="${cand.experience_years || ''}"></div>
        </div>
        <div class="form-row">
            <div class="form-col"><label>Current Company</label><input id="c-edit-comp" value="${cand.current_company || ''}"></div>
            <div class="form-col"><label>Current Position</label><input id="c-edit-pos" value="${cand.current_position || ''}"></div>
        </div>
        <div class="form-col" style="margin-bottom:12px;">
            <label>Skills (comma separated)</label>
            <input id="c-edit-skills" value="${cand.skills ? cand.skills.join(', ') : ''}" placeholder="Java, Python...">
        </div>
        <div class="form-col" style="margin-bottom:12px;">
            <label>Notes</label>
            <textarea id="c-edit-notes" rows="3">${cand.notes || ''}</textarea>
        </div>
        
        <button class="btn btn-primary" onclick="submitEditCandidate('${taskId}')" style="width:100%;">💾 Save Changes</button>
        <button class="btn btn-outline" onclick="closeCandidateModal()" style="width:100%; margin-top:8px;">❌ Cancel</button>
    `;
};

window.submitEditCandidate = async (taskId) => {
    if (!currentEditingCandId) return;

    const skillsStr = document.getElementById('c-edit-skills').value;
    const skills = skillsStr ? skillsStr.split(',').map(s => s.trim()).filter(s => s) : [];

    const data = {
        name: document.getElementById('c-edit-name').value,
        email: document.getElementById('c-edit-email').value,
        phone: document.getElementById('c-edit-phone').value,
        experience_years: document.getElementById('c-edit-exp').value,
        current_company: document.getElementById('c-edit-comp').value,
        current_position: document.getElementById('c-edit-pos').value,
        skills: skills,
        notes: document.getElementById('c-edit-notes').value
    };

    const res = await apiRequest(`/api/candidates/${currentEditingCandId}/edit`, 'PUT', data);

    if (res.success) {
        showToast('Candidate updated!', 'success');
        closeCandidateModal();
        loadTaskCandidates(taskId);
        currentEditingCandId = null;
    } else {
        showToast(res.error || 'Failed to update', 'error');
    }
};

// --- Recruiter: Candidate Management ---

function openAddCandidateModal(taskId) {
    const modal = document.getElementById('candidate-modal');
    const modalTitle = document.getElementById('candidate-modal-title');
    modal.classList.add('active');
    modalTitle.innerText = '👥 Add Candidate';

    // Clear any previous state
    window.parsedCandidateCache = null;

    document.getElementById('candidate-modal-body').innerHTML = `
        <div class="tabs-container">
            <button class="tab-btn active" onclick="switchCandTab(0)">📄 Autofill (Resume)</button>
            <button class="tab-btn" onclick="switchCandTab(1)">✍️ Manual Entry</button>
        </div>
        
        <!-- Autofill Tab -->
        <div class="cand-tab-content active" id="cand-tab-0">
            <div id="autofill-upload-section">
                <div class="alert alert-info">🤖 Upload a resume and AI will extract candidate details automatically!</div>
                <input type="file" id="resume-upload-${taskId}" class="file-input" accept=".pdf,.docx,.txt">
                <button class="btn btn-primary" style="margin-top:12px; width:100%;" onclick="parseResumeForReview('${taskId}')">🔍 Parse Resume</button>
            </div>
            
            <div id="autofill-review-section" style="display:none; margin-top:20px; border-top:1px solid var(--border); padding-top:15px;">
                <h4 style="margin-bottom:15px;">📝 Review & Edit Extracted Details</h4>
                <div class="form-row">
                    <div class="form-col"><label>Name *</label><input id="af-name-${taskId}"></div>
                    <div class="form-col"><label>Email *</label><input id="af-email-${taskId}"></div>
                </div>
                <div class="form-row">
                    <div class="form-col"><label>Phone *</label><input id="af-phone-${taskId}"></div>
                    <div class="form-col"><label>Exp (Years)</label><input type="number" id="af-exp-${taskId}"></div>
                </div>
                <div class="form-row">
                    <div class="form-col"><label>Current Company</label><input id="af-comp-${taskId}"></div>
                    <div class="form-col"><label>Current Position</label><input id="af-pos-${taskId}"></div>
                </div>
                <div class="form-col" style="margin-bottom:12px;">
                     <label>Skills (comma separated)</label>
                     <input id="af-skills-${taskId}" placeholder="Python, Java, AWS...">
                </div>
                <div class="form-col" style="margin-bottom:12px;">
                     <label>Notes</label>
                     <textarea id="af-notes-${taskId}" rows="2"></textarea>
                </div>
                <div class="form-col" style="margin-bottom:12px;"><label>Initial Status</label>
                    <select id="af-status-${taskId}">
                        <option>Applied</option><option>Shortlisted</option><option>Interview Scheduled</option>
                        <option>Interviewed</option><option>Offer Extended</option><option>Rejected</option>
                    </select>
                </div>
                <button class="btn btn-primary" onclick="submitParsedCandidate('${taskId}')" style="width:100%;">➕ Add Candidate with Resume</button>
                <button class="btn btn-outline" onclick="resetAutofill('${taskId}')" style="width:100%; margin-top:8px;">❌ Cancel / Re-upload</button>
            </div>
            
            <div id="parse-result-${taskId}" style="margin-top:10px;"></div>
        </div>
        
        <!-- Manual Tab -->
        <div class="cand-tab-content" id="cand-tab-1" style="display:none;">
            <div class="form-row">
                <div class="form-col"><label>Name *</label><input id="c-name-${taskId}"></div>
                <div class="form-col"><label>Email *</label><input id="c-email-${taskId}"></div>
            </div>
            <div class="form-row">
                <div class="form-col"><label>Phone *</label><input id="c-phone-${taskId}"></div>
                <div class="form-col"><label>Exp (Years)</label><input type="number" id="c-exp-${taskId}"></div>
            </div>
            <div class="form-row">
                <div class="form-col"><label>Current Company</label><input id="c-comp-${taskId}"></div>
                <div class="form-col"><label>Current Position</label><input id="c-pos-${taskId}"></div>
            </div>
            <div class="form-col" style="margin-bottom:12px;">
                <label>Skills (comma separated)</label>
                <input id="c-skills-${taskId}" placeholder="Python, Java, AWS...">
            </div>
            <div class="form-col" style="margin-bottom:12px;">
                <label>Notes</label>
                <textarea id="c-notes-${taskId}" rows="2"></textarea>
            </div>
            <div class="form-col" style="margin-bottom:12px;"><label>Initial Status</label>
                <select id="c-status-${taskId}">
                    <option>Applied</option><option>Shortlisted</option><option>Interview Scheduled</option>
                    <option>Interviewed</option><option>Offer Extended</option><option>Rejected</option>
                </select>
            </div>
            <button class="btn btn-primary" onclick="addManualCandidate('${taskId}')" style="width:100%;">➕ Add Candidate</button>
        </div>
    `;
}

function switchCandTab(idx) {
    document.querySelectorAll('#candidate-modal .tab-btn').forEach((b, i) => {
        b.classList.toggle('active', i === idx);
    });
    document.getElementById('cand-tab-0').style.display = idx === 0 ? 'block' : 'none';
    document.getElementById('cand-tab-1').style.display = idx === 1 ? 'block' : 'none';
}

function closeCandidateModal() {
    document.getElementById('candidate-modal').classList.remove('active');
}

function resetAutofill(taskId) {
    document.getElementById('autofill-upload-section').style.display = 'block';
    document.getElementById('autofill-review-section').style.display = 'none';
    document.getElementById(`parse-result-${taskId}`).innerHTML = '';
    window.parsedCandidateCache = null;
}

async function addManualCandidate(taskId) {
    const name = document.getElementById(`c-name-${taskId}`).value;
    const email = document.getElementById(`c-email-${taskId}`).value;
    const phone = document.getElementById(`c-phone-${taskId}`).value;

    if (!name || !email || !phone) return showToast('Name, Email, and Phone are required', 'error');

    const skillsStr = document.getElementById(`c-skills-${taskId}`).value;
    const skills = skillsStr ? skillsStr.split(',').map(s => s.trim()).filter(s => s) : [];

    const data = {
        name: name,
        email: email,
        phone: phone,
        experience_years: document.getElementById(`c-exp-${taskId}`).value,
        current_company: document.getElementById(`c-comp-${taskId}`).value,
        current_position: document.getElementById(`c-pos-${taskId}`).value,
        status: document.getElementById(`c-status-${taskId}`).value,
        skills: skills,
        notes: document.getElementById(`c-notes-${taskId}`).value
    };

    const res = await apiRequest(`/api/tasks/${taskId}/candidates`, 'POST', data);
    if (res.success) {
        showToast('Candidate added!', 'success');
        closeCandidateModal();
        loadTaskCandidates(taskId);
    } else {
        showToast(res.error || 'Failed to add', 'error');
    }
}

async function parseResumeForReview(taskId) {
    const fileInput = document.getElementById(`resume-upload-${taskId}`);
    if (!fileInput.files[0]) return showToast('Please select a file', 'error');

    const statusDiv = document.getElementById(`parse-result-${taskId}`);
    statusDiv.innerHTML = '<div class="spinner"></div> 🤖 Extracting candidate details from resume...';

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    const parseRes = await apiRequest('/api/parse-resume', 'POST', formData, true);

    if (parseRes.success) {
        statusDiv.innerHTML = '';
        const extracted = parseRes.extracted_data;

        // Cache for submission
        window.parsedCandidateCache = {
            resume_data: parseRes.resume_data,
            resume_filename: parseRes.resume_filename
        };

        // Populate fields
        document.getElementById(`af-name-${taskId}`).value = extracted.name || '';
        document.getElementById(`af-email-${taskId}`).value = extracted.email || '';
        document.getElementById(`af-phone-${taskId}`).value = extracted.phone || '';
        document.getElementById(`af-exp-${taskId}`).value = extracted.experience_years || 0;
        document.getElementById(`af-comp-${taskId}`).value = extracted.current_company || '';
        document.getElementById(`af-pos-${taskId}`).value = extracted.current_position || '';
        document.getElementById(`af-skills-${taskId}`).value = (extracted.skills || []).join(', ');
        document.getElementById(`af-notes-${taskId}`).value = extracted.notes || '';

        // Show review section, hide upload
        document.getElementById('autofill-upload-section').style.display = 'none';
        document.getElementById('autofill-review-section').style.display = 'block';
        showToast('Resume parsed! Please review details.', 'success');

    } else {
        statusDiv.innerHTML = `<span class="text-danger">Parse failed: ${parseRes.error}</span>`;
    }
}

async function submitParsedCandidate(taskId) {
    if (!window.parsedCandidateCache) return;

    const name = document.getElementById(`af-name-${taskId}`).value;
    const email = document.getElementById(`af-email-${taskId}`).value;
    const phone = document.getElementById(`af-phone-${taskId}`).value;

    if (!name || !email || !phone) return showToast('Name, Email, and Phone are required', 'error');

    const skillsStr = document.getElementById(`af-skills-${taskId}`).value;
    const skills = skillsStr ? skillsStr.split(',').map(s => s.trim()).filter(s => s) : [];

    const candidateData = {
        name: name,
        email: email,
        phone: phone,
        experience_years: document.getElementById(`af-exp-${taskId}`).value,
        current_company: document.getElementById(`af-comp-${taskId}`).value,
        current_position: document.getElementById(`af-pos-${taskId}`).value,
        skills: skills,
        notes: document.getElementById(`af-notes-${taskId}`).value,
        status: document.getElementById(`af-status-${taskId}`).value,
        resume_data: window.parsedCandidateCache.resume_data,
        resume_filename: window.parsedCandidateCache.resume_filename
    };

    const addRes = await apiRequest(`/api/tasks/${taskId}/candidates`, 'POST', candidateData);
    if (addRes.success) {
        showToast('Candidate added successfully with resume!', 'success');
        closeCandidateModal();
        loadTaskCandidates(taskId);
    } else {
        showToast(addRes.error || 'Failed to add parsed candidate', 'error');
    }
}

async function viewResume(candId) {
    const modal = document.getElementById('resume-modal');
    const body = document.getElementById('resume-modal-body');
    const title = document.getElementById('resume-modal-title');

    modal.classList.add('active');
    body.innerHTML = '<div class="loading-state"><div class="spinner"></div></div>';

    const res = await apiRequest(`/api/candidates/${candId}/resume`);

    if (res.success && res.resume_data) {
        title.innerText = `📄 Resume: ${res.resume_filename}`;

        if (res.resume_filename.toLowerCase().endsWith('.pdf')) {
            body.innerHTML = `<iframe src="data:application/pdf;base64,${res.resume_data}" width="100%" height="100%" style="min-height:80vh; border:none;"></iframe>`;
        } else {
            body.innerHTML = `
                <div class="empty-state">
                    <p>Preview not available for this file type.</p>
                    <a href="data:application/octet-stream;base64,${res.resume_data}" download="${res.resume_filename}" class="btn btn-primary">⬇️ Download File</a>
                </div>
            `;
        }
    } else {
        body.innerHTML = '<div class="empty-state"><p>Resume not found or error loading.</p></div>';
    }
}

function closeResumeModal() {
    document.getElementById('resume-modal').classList.remove('active');
}

// --- Recruiter: JD Extraction ---

async function handleExtractJD() {
    const fileInput = document.getElementById('jd-file-input');
    const textInput = document.getElementById('jd-text-input');
    const resultsDiv = document.getElementById('extract-results');

    resultsDiv.innerHTML = '<div class="loading-state"><div class="spinner"></div><p>🤖 AI Agent is analyzing JDs...</p></div>';

    if (textInput.value.trim()) {
        const res = await apiRequest('/api/requirements/extract', 'POST', { content: textInput.value });
        if (!res.error) {
            showToast('Task created successfully!', 'success');
            textInput.value = '';
            resultsDiv.innerHTML = renderExtractionSuccess(res.extracted_data, res.task_id);
            // Auto refresh tasks page
            loadRecTasks();
            setTimeout(() => {
                const btn = document.querySelector('.nav-item[data-page="rec-tasks"]');
                if (btn) btn.click(); // Redirect to all tasks
            }, 1000);
        } else {
            resultsDiv.innerHTML = `<div class="alert alert-danger">${res.error}</div>`;
        }
    } else if (fileInput.files.length > 0) {
        // Multi-file upload
        const promises = Array.from(fileInput.files).map(async file => {
            const fd = new FormData();
            fd.append('file', file);
            return apiRequest('/api/upload-requirement', 'POST', fd, true);
        });

        const responses = await Promise.all(promises);
        const successes = responses.filter(r => !r.error);

        if (successes.length > 0) {
            showToast(`processed ${successes.length} JDs`, 'success');
            fileInput.value = ''; // clear
            resultsDiv.innerHTML = successes.map(r => renderExtractionSuccess(r.extracted_data, r.task_id)).join('');
            loadRecTasks();
            setTimeout(() => {
                const btn = document.querySelector('.nav-item[data-page="rec-tasks"]');
                if (btn) btn.click();
            }, 1500);
        } else {
            resultsDiv.innerHTML = `<div class="alert alert-danger">All uploads failed.</div>`;
        }
    } else {
        showToast('Please upload a file or paste text', 'warning');
        resultsDiv.innerHTML = '';
    }
}

function renderExtractionSuccess(data, taskId) {
    return `
        <div class="data-card" style="border-left: 4px solid var(--success);">
            <div class="data-card-body" style="padding:16px;">
                <div style="display:flex; justify-content:space-between;">
                    <div>
                        <h4 style="margin-bottom:8px;">✅ ${data.title || 'Extracted Task'}</h4>
                        <div class="badge badge-info">${data.priority} Priority</div>
                        <div class="badge badge-warning">${data.urgency}</div>
                    </div>
                    <div style="text-align:right;">
                        <small>Task ID: ${taskId}</small>
                        <div style="margin-top:5px;">Complexity: ${data.complexity}</div>
                    </div>
                </div>
                <div style="margin-top:10px; font-size:13px; color:var(--text-secondary);">
                    <strong>Skills:</strong> ${data.skills ? data.skills.join(', ') : 'N/A'}
                </div>
            </div>
        </div>
    `;
}

// --- Recruiter: Interview Agent ---
let interviewTaskCache = [];

async function initInterviewPage() {
    const select = document.getElementById('interview-task-select');
    select.innerHTML = '<option>Loading...</option>';

    // Fetch active tasks
    const tasks = await apiRequest('/api/tasks');
    interviewTaskCache = tasks.filter(t => t.status !== 'completed'); // only active

    if (interviewTaskCache.length === 0) {
        select.innerHTML = '<option value="">No active tasks available</option>';
        return;
    }

    select.innerHTML = '<option value="">-- Select Role --</option>' +
        interviewTaskCache.map(t => `<option value="${t.id}">${t.title}</option>`).join('');

    // Listener for task change to load candidates
    select.onchange = async () => {
        const taskId = select.value;
        const candDiv = document.getElementById('interview-candidate-multi');
        candDiv.innerHTML = '<span class="spinner"></span>';

        if (!taskId) { candDiv.innerHTML = ''; return; }

        const cands = await apiRequest(`/api/tasks/${taskId}/candidates`);
        const eligible = cands.filter(c => ['applied', 'shortlisted', 'Interview Scheduled'].includes(c.status)); // loosened filter

        if (eligible.length === 0) {
            candDiv.innerHTML = '<small class="text-danger">No eligible candidates</small>';
            return;
        }

        // Render custom multi-select checkbox style
        candDiv.innerHTML = eligible.map(c => `
            <label style="display:inline-flex; gap:8px; margin-right:12px; font-size:13px; cursor:pointer; align-items:center;">
                <input type="checkbox" name="interview-cand" value="${c.id}" checked> ${c.name}
            </label>
        `).join('');
    };
}

async function getSlotSuggestions() {
    const date = document.getElementById('interview-date').value;
    if (!date) return showToast('Pick a date first', 'warning');

    const btn = document.querySelector('button[onclick="getSlotSuggestions()"]');
    const origText = btn.innerText;
    btn.innerText = '⌛...';

    const res = await apiRequest(`/api/suggest-slots?date=${date}`);
    btn.innerText = origText;

    const select = document.getElementById('interview-time-select');
    select.innerHTML = '';

    if (res.time_slots && res.time_slots.length > 0) {
        res.time_slots.forEach(s => {
            const opt = document.createElement('option');
            opt.value = s.slot;
            opt.innerText = s.slot;
            select.appendChild(opt);
        });
        showToast('Slots updated based on calendar', 'success');
    } else {
        select.innerHTML = '<option value="">No slots found</option>';
    }
}

async function handleSendInvites() {
    const taskId = document.getElementById('interview-task-select').value;
    const date = document.getElementById('interview-date').value;
    const time = document.getElementById('interview-time-select').value;
    const location = document.getElementById('interview-location').value;
    const link = document.getElementById('interview-meeting-link').value;

    const checkboxes = document.querySelectorAll('input[name="interview-cand"]:checked');
    const candidateIds = Array.from(checkboxes).map(cb => cb.value);

    if (!taskId || !date || !time || candidateIds.length === 0) {
        return showToast('Missing required fields', 'error');
    }

    showToast('Sending invites...', 'info');

    const payload = {
        task_id: taskId,
        candidate_ids: candidateIds,
        interview_date: date,
        interview_time: time,
        location: location,
        meeting_link: link
    };

    const res = await apiRequest('/api/schedule-interview', 'POST', payload);

    if (res.success) {
        showToast('Invites sent successfully!', 'success');
        document.getElementById('interview-results').innerHTML = `<div class="alert alert-success">✅ Invites sent to ${candidateIds.length} candidates for ${date} at ${time}.</div>`;
    } else {
        showToast(res.error || 'Failed to send', 'error');
    }
}

// --- Recruiter: EOD ---

async function handleGenerateEOD() {
    const btn = document.getElementById('eod-btn');
    const container = document.getElementById('eod-report-container');
    const area = document.getElementById('eod-report-area');

    const originalText = btn.innerText;
    btn.innerText = '⌛ Generating...';
    btn.disabled = true;

    const res = await apiRequest('/api/eod-summary', 'POST', {});

    btn.innerText = originalText;
    btn.disabled = false;

    if (res.summary) {
        container.style.display = 'block';

        // Simple parsing to make it look "proper"
        const lines = res.summary.split('\n');
        let html = '<div class="report-document" style="color:#1e293b; font-family:serif;">';

        lines.forEach(line => {
            if (line.trim().startsWith('=')) return;
            if (line.includes('COMPREHENSIVE EOD SUMMARY')) {
                html += `<h1 style="text-align:center; color:var(--accent); font-size:24px; margin-bottom:10px;">${line.replace('🎯 ', '')}</h1>`;
            } else if (line.includes('SUMMARY') || line.includes('INTERVIEWS') || line.includes('TASKS') || line.includes('WORKLOAD')) {
                html += `<h3 style="border-bottom:2px solid var(--border); padding-bottom:8px; margin:25px 0 15px; color:var(--bg-primary); text-transform:uppercase; font-size:16px;">${line}</h3>`;
            } else if (line.trim().startsWith('•')) {
                html += `<div style="padding:10px 15px; background:var(--bg-body); border-radius:8px; margin-bottom:8px; border-left:4px solid var(--accent); font-size:14px;">${line.trim().substring(2)}</div>`;
            } else if (line.trim().startsWith('🔴') || line.trim().startsWith('🟡') || line.trim().startsWith('🟢') || line.trim().startsWith('⚙️')) {
                html += `<div style="font-weight:700; margin:15px 0 10px; font-size:15px;">${line}</div>`;
            } else if (line.trim()) {
                html += `<p style="margin-bottom:8px; font-size:14px; line-height:1.6;">${line}</p>`;
            }
        });

        html += '</div>';
        area.innerHTML = html;

        if (res.email_sent) showToast('EOD Report generated and emailed successfully!', 'success');

        // Scroll to report
        container.scrollIntoView({ behavior: 'smooth' });
    } else {
        showToast(res.error || 'Failed to generate report', 'error');
    }
}

async function handleCheckWorkload() {
    const analysisArea = document.getElementById('rec-workload-analysis');

    const res = await apiRequest('/api/admin/workload-report');

    if (res.recruiters) {
        const user = JSON.parse(localStorage.getItem('user') || '{}');
        const myLoad = res.recruiters.find(r => r.recruiter_name === user.username);

        if (myLoad) {
            // Update Metric Cards
            document.getElementById('val-total-tasks').innerText = myLoad.total_tasks;
            document.getElementById('val-pending-tasks').innerText = myLoad.total_tasks - myLoad.completed_tasks; // Simple approx
            document.getElementById('val-workload-percent').innerText = myLoad.workload_percentage.toFixed(1) + '%';
            document.getElementById('val-avg-time').innerText = myLoad.avg_completion_hours + 'h';

            // Risk Analysis
            let riskClass = myLoad.workload_percentage > 75 ? 'danger' : (myLoad.workload_percentage > 45 ? 'warning' : 'success');
            let riskLabel = myLoad.workload_percentage > 75 ? 'High Risk' : (myLoad.workload_percentage > 45 ? 'Moderate load' : 'Optimal Capacity');
            let recommendation = myLoad.workload_percentage > 75 ? "Your workload is critical. Consider completing high-priority tasks before starting new ones." :
                (myLoad.workload_percentage > 45 ? "You are at moderate capacity. Focus on maintaining your current pace." : "You have capacity to take on more complex tasks.");

            analysisArea.innerHTML = `
                <div class="alert alert-${riskClass}" style="margin-bottom:15px;">
                    <h4 style="margin-bottom:5px;">${riskLabel} (${myLoad.workload_percentage.toFixed(1)}%)</h4>
                    <p style="font-size:13px; opacity:0.9;">Based on your active tasks and average completion speeds.</p>
                </div>
                <div style="font-size:14px; line-height:1.6; color:var(--text-secondary);">
                    <p><strong>Recommendation:</strong> ${recommendation}</p>
                    <ul style="margin-top:12px; padding-left:20px;">
                        <li>Active Roles: ${myLoad.total_tasks}</li>
                        <li>Avg. Burn rate: ${myLoad.avg_completion_hours} hrs/task</li>
                    </ul>
                </div>
             `;
        } else {
            analysisArea.innerHTML = '<div class="alert alert-info">No workload data found for your account. Start by creating or being assigned to a task.</div>';
        }
    } else {
        analysisArea.innerHTML = '<div class="alert alert-danger">Unable to fetch workload analysis. Please contact admin.</div>';
    }
}

window.printReport = function () {
    const content = document.getElementById('eod-report-area').innerHTML;
    const printWindow = window.open('', '_blank');
    printWindow.document.write('<html><head><title>EOD Report</title><style>body{font-family:serif; padding:40px;} .report-document{max-width:800px; margin:auto;}</style></head><body>');
    printWindow.document.write(content);
    printWindow.document.write('</body></html>');
    printWindow.document.close();
    printWindow.print();
};


// ==================== ADMIN FUNCTIONS (Legacy + Updated) ====================

function showDashboard() {
    document.getElementById('login-page').style.display = 'none';
    document.getElementById('dashboard-page').classList.remove('hidden');

    // Sidebar setup
    document.getElementById('admin-nav').classList.remove('hidden');
    document.getElementById('recruiter-nav').classList.add('hidden');
    document.getElementById('sidebar-role-label').innerText = 'Administrator';

    const admin = JSON.parse(localStorage.getItem('admin_user') || '{}');
    document.getElementById('sidebar-user-name').innerText = admin.username || 'Admin';
    document.getElementById('sidebar-user-email').innerText = 'admin@recruiter-os.com';
    document.getElementById('sidebar-user-avatar').innerText = 'A';

    navigateTo('dashboard');
}

// --- Admin Dashboard Loaders (abbreviated from previous impl) ---
async function loadDashboard() {
    const container = document.getElementById('dashboard-content');
    container.innerHTML = '<div class="loading-state"><div class="spinner"></div><p>Loading...</p></div>';

    const data = await apiRequest('/api/admin/dashboard');
    if (data.error) { container.innerHTML = `<div class="alert alert-danger">${data.error}</div>`; return; }

    // Summary
    const summary = data.summary;
    container.innerHTML = `
        <div class="metrics-grid">
            <div class="metric-card info"><div class="label">Total Recruiters</div><div class="value">${summary.total_recruiters}</div></div>
            <div class="metric-card success"><div class="label">Online Now</div><div class="value">${summary.online_count}</div></div>
            <div class="metric-card accent"><div class="label">Total Tasks</div><div class="value">${summary.total_tasks}</div></div>
            <div class="metric-card warning"><div class="label">Pending Tasks</div><div class="value">${summary.pending_tasks}</div></div>
        </div>
        <div class="data-card">
            <div class="data-card-header"><h3>👥 Recruiter Status</h3></div>
            <div class="data-card-body">
                <table class="data-table">
                    <thead><tr><th>Recruiter</th><th>Status</th><th>Workload</th><th>Tasks</th></tr></thead>
                    <tbody>
                        ${data.recruiters_workload.map(r => `
                            <tr>
                                <td>${r.recruiter_name}</td>
                                <td><span class="badge badge-${r.current_status === 'online' ? 'online' : 'offline'}">${r.current_status.toUpperCase()}</span></td>
                                <td>
                                    <div style="display:flex;align-items:center;gap:8px;">
                                        <div class="progress-bar-wrapper" style="width:80px;"><div class="progress-bar-fill ${r.workload_percentage > 80 ? 'progress-red' : 'progress-green'}" style="width:${r.workload_percentage}%"></div></div>
                                        <span style="font-size:11px;">${r.workload_percentage.toFixed(0)}%</span>
                                    </div>
                                </td>
                                <td>${r.total_tasks} active</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        </div>
    `;
}

async function loadRecruiters() {
    const container = document.getElementById('recruiters-content');
    container.innerHTML = '<div class="loading-state"><div class="spinner"></div></div>';

    const res = await apiRequest('/api/admin/recruiters');
    if (res.error) return elementError(container, res.error);

    // Simplified list view
    container.innerHTML = `
        <div class="metrics-grid">
            ${res.map(r => `
                <div class="metric-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                        <div style="font-weight:700; font-size:16px;">${r.username || r.name}</div>
                        <span class="badge badge-${r.status === 'online' ? 'online' : 'offline'}">${r.status || 'offline'}</span>
                    </div>
                    <div style="font-size:12px; color:var(--text-muted); margin-bottom:15px;">
                        <p>ID: ${r.id}</p>
                        <p>Email: ${r.email || 'N/A'}</p>
                    </div>
                    <button class="btn btn-sm btn-primary" onclick="openRecruiterDetail('${r.id}')" style="width:100%;">👤 View Details</button>
                </div>
            `).join('')}
        </div>
    `;
}

async function loadWorkloadReport() {
    const container = document.getElementById('workload-content');
    container.innerHTML = '<div class="loading-state"><div class="spinner"></div></div>';
    const res = await apiRequest('/api/admin/workload-report');

    if (res.error) return elementError(container, res.error);

    let totalTasks = 0;
    let highRiskCount = 0;
    let avgLoad = 0;

    const rows = res.recruiters.map(r => {
        const load = r.workload_percentage || 0;
        totalTasks += r.total_tasks;
        avgLoad += load;

        const risk = load > 70 ? 'High' : (load > 40 ? 'Medium' : 'Low');
        if (risk === 'High') highRiskCount++;

        const riskClass = risk.toLowerCase();
        const recommendation = load > 70 ? '⚠️ High Load: Redistribute' : (load < 30 ? '📉 Low Load: Assign More' : '✅ Optimal');

        return `
            <tr>
                <td>
                    <div style="font-weight:600;">${r.recruiter_name}</div>
                    <div style="font-size:11px; color:var(--text-muted);">Last active: ${r.last_active || 'N/A'}</div>
                </td>
                <td><span class="badge badge-${riskClass}">${risk} Risk</span></td>
                <td>
                    <div style="display:flex; align-items:center; gap:10px;">
                        <div class="progress-bar-wrapper" style="flex:1; height:8px; width:100px;">
                            <div class="progress-bar-fill ${load > 70 ? 'progress-red' : (load > 40 ? 'progress-yellow' : 'progress-green')}" style="width:${load}%"></div>
                        </div>
                        <span style="font-weight:700; font-size:12px;">${load.toFixed(1)}%</span>
                    </div>
                </td>
                <td style="text-align:center;">${r.total_tasks}</td>
                <td style="text-align:center;">${r.completed_tasks}</td>
                <td style="font-size:13px; font-weight:500;">${recommendation}</td>
            </tr>
        `;
    }).join('');

    avgLoad = res.recruiters.length > 0 ? (avgLoad / res.recruiters.length) : 0;

    container.innerHTML = `
        <div class="metrics-grid">
            <div class="metric-card info">
                <div class="label">Total Team Tasks</div>
                <div class="value">${totalTasks}</div>
            </div>
            <div class="metric-card ${avgLoad > 60 ? 'warning' : 'success'}">
                <div class="label">Average Team Load</div>
                <div class="value">${avgLoad.toFixed(1)}%</div>
            </div>
            <div class="metric-card ${highRiskCount > 0 ? 'danger' : 'success'}">
                <div class="label">Recruiters at High Risk</div>
                <div class="value">${highRiskCount}</div>
            </div>
        </div>

        <div class="data-card">
            <div class="data-card-header">
                <h3>📈 Detailed Team Workload Analysis</h3>
                <div style="font-size:12px; color:var(--text-muted);">Real-time metrics based on active assignments</div>
            </div>
            <div class="data-card-body">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Recruiter</th>
                            <th>Status</th>
                            <th>Current Load</th>
                            <th style="text-align:center;">Active</th>
                            <th style="text-align:center;">Done</th>
                            <th>AI Recommendation</th>
                        </tr>
                    </thead>
                    <tbody>${rows || '<tr><td colspan="6" style="text-align:center; padding:40px;">No recruiter data available</td></tr>'}</tbody>
                </table>
            </div>
        </div>
    `;
}

async function loadActivityLogs() {
    const container = document.getElementById('activity-content');
    container.innerHTML = '<div class="loading-state"><div class="spinner"></div></div>';

    const res = await apiRequest('/api/admin/logs');
    if (res.error) return elementError(container, res.error);

    container.innerHTML = `
        <div class="data-card">
            <div class="data-card-body" style="padding:0">
                <table class="data-table">
                    <thead><tr><th>Recruiter</th><th>Login</th><th>Logout</th><th>Minutes</th></tr></thead>
                    <tbody>
                        ${res.map(l => `
                            <tr>
                                <td><strong>${l.recruiter_name}</strong></td>
                                <td>${l.login_time?.substring(0, 16).replace('T', ' ')}</td>
                                <td>${l.logout_time?.substring(0, 16).replace('T', ' ')}</td>
                                <td>${typeof l.duration_minutes === 'number' ? l.duration_minutes.toFixed(1) : l.duration_minutes}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        </div>
    `;
}

// ==================== CORE NAVIGATION & HANDLERS ====================

// Navigation Logic
const pages = {
    // Recruiter Pages
    'rec-dashboard': { id: 'page-rec-dashboard', load: loadRecDashboard },
    'rec-tasks': { id: 'page-rec-tasks', load: loadRecTasks },
    'rec-extract': { id: 'page-rec-extract', load: () => { } }, // static
    'rec-interview': { id: 'page-rec-interview', load: initInterviewPage },
    'rec-eod': { id: 'page-rec-eod', load: () => { handleCheckWorkload(); } },

    // Admin Pages
    'dashboard': { id: 'page-dashboard', load: loadDashboard },
    'recruiters': { id: 'page-recruiters', load: loadRecruiters },
    'activity': { id: 'page-activity', load: loadActivityLogs },
    'workload': { id: 'page-workload', load: loadWorkloadReport },
    'settings': { id: 'page-settings', load: loadSettingsPage }
};

function navigateTo(pageName) {
    // Updating active class in sidebar
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.querySelector(`.nav-item[data-page="${pageName}"]`)?.classList.add('active');

    // Hiding all sections
    document.querySelectorAll('.page-section').forEach(s => s.classList.remove('active'));

    // Showing target section
    const page = pages[pageName];
    if (page) {
        document.getElementById(page.id).classList.add('active');
        if (page.load) page.load();
    }
}

// Global Event Listeners
document.addEventListener('DOMContentLoaded', () => {

    // Check auth on load
    checkAuth();

    // Navigation Click
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            const page = item.getAttribute('data-page');
            navigateTo(page);
            // Mobile close sidebar
            if (window.innerWidth <= 1024) {
                document.getElementById('main-sidebar').classList.remove('open');
            }
        });
    });

    // Mobile Toggle
    document.getElementById('mobile-toggle').addEventListener('click', () => {
        document.getElementById('main-sidebar').classList.toggle('open');
    });

    // Login Form Submit (Admin)
    document.getElementById('admin-login-form')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const u = document.getElementById('login-username').value;
        const p = document.getElementById('login-password').value;
        const btn = document.getElementById('login-btn');

        btn.innerText = 'Authenticating...';
        const res = await apiRequest('/api/admin/login', 'POST', { username: u, password: p });
        btn.innerText = '🔐 Admin Login';

        if (res.access_token) {
            localStorage.setItem('admin_token', res.access_token);
            localStorage.setItem('role', 'admin');
            localStorage.setItem('admin_user', JSON.stringify(res.admin));
            currentUserRole = 'admin';
            adminToken = res.access_token;
            showDashboard();
            showToast('Welcome Admin!', 'success');
        } else {
            showToast(res.detail || 'Login failed', 'error');
        }
    });

    // Login Form Submit (Recruiter)
    const recLoginForm = document.getElementById('recruiter-login-form');
    if (recLoginForm) {
        recLoginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            console.log('Login submit clicked');

            const uInput = document.getElementById('rec-login-username');
            const pInput = document.getElementById('rec-login-password');
            const btn = document.getElementById('rec-login-btn');

            if (!uInput || !pInput) {
                console.error('Login inputs not found');
                return;
            }

            const u = uInput.value;
            const p = pInput.value;

            console.log('Attempting login for:', u);

            const originalText = btn.innerText;
            btn.innerText = 'Logging in...';
            btn.disabled = true;

            try {
                const res = await apiRequest('/api/login', 'POST', { username: u, password: p });

                if (res.access_token) {
                    console.log('Login success');
                    localStorage.setItem('token', res.access_token);
                    localStorage.setItem('role', 'recruiter');
                    localStorage.setItem('user', JSON.stringify(res.user));

                    // Set global state
                    currentUserRole = 'recruiter';
                    authToken = res.access_token;

                    showRecruiterDashboard();
                    showToast('Login successful!', 'success');
                } else {
                    console.warn('Login failed:', res);
                    showToast(res.detail || 'Invalid credentials', 'error');
                }
            } catch (err) {
                console.error('Login error:', err);
                showToast('Login error occurred', 'error');
            } finally {
                btn.innerText = originalText;
                btn.disabled = false;
            }
        });
    } else {
        console.error('Recruiter login form not found in DOM');
    }

    // Register Form
    document.getElementById('recruiter-register-form')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const u = document.getElementById('rec-reg-username').value;
        const em = document.getElementById('rec-reg-email').value;
        const p = document.getElementById('rec-reg-password').value;
        const c = document.getElementById('rec-reg-confirm').value;

        if (p !== c) return showToast('Passwords do not match', 'error');

        const btn = document.getElementById('rec-reg-btn');
        btn.innerText = 'Creating account...';

        const res = await apiRequest('/api/register', 'POST', { username: u, password: p, email: em });
        btn.innerText = '📝 Register';

        if (res.access_token) {
            localStorage.setItem('token', res.access_token);
            localStorage.setItem('role', 'recruiter');
            localStorage.setItem('user', JSON.stringify(res.user));
            currentUserRole = 'recruiter';
            authToken = res.access_token;
            showRecruiterDashboard();
            showToast('Account created!', 'success');
        } else {
            showToast(res.detail || 'Registration failed', 'error');
        }
    });

    // Logout
    document.getElementById('logout-btn').addEventListener('click', handleLogout);
});

// Helper for generic element error
function elementError(el, msg) {
    el.innerHTML = `<div class="alert alert-danger">${msg}</div>`;
}

// Global functions for inline onclicks
window.switchLoginRole = switchLoginRole;
window.switchLoginTab = switchLoginTab;
window.loadRecDashboard = loadRecDashboard;
window.loadRecTasks = loadRecTasks;
window.handleExtractJD = handleExtractJD;
window.handleGenerateEOD = handleGenerateEOD;
window.handleCheckWorkload = handleCheckWorkload;
window.toggleTask = toggleTask;
window.completeTask = async (id) => {
    if (confirm('Mark task as complete?')) {
        const res = await apiRequest(`/api/tasks/${id}/complete`, 'POST');
        if (res.success) { showToast('Task completed', 'success'); loadRecDashboard(); loadRecTasks(); }
    }
};
window.deleteTask = async (id) => {
    if (confirm('Delete this task?')) {
        const res = await apiRequest(`/api/tasks/${id}`, 'DELETE');
        if (res.success) { showToast('Task deleted', 'success'); loadRecTasks(); }
    }
};
window.openAddCandidateModal = openAddCandidateModal;
window.closeCandidateModal = closeCandidateModal;
window.switchCandTab = switchCandTab;
window.parseAndAddCandidate = parseResumeForReview;
window.parseResumeForReview = parseResumeForReview;
window.submitParsedCandidate = submitParsedCandidate;
window.resetAutofill = resetAutofill;
window.addManualCandidate = addManualCandidate;
window.viewResume = viewResume;
window.closeResumeModal = closeResumeModal;
window.openEditCandidateModal = openEditCandidateModal;
window.submitEditCandidate = submitEditCandidate;
window.updateCandidateStatus = updateCandidateStatus;
window.deleteCandidate = async (id) => {
    if (confirm('Delete candidate?')) {
        const res = await apiRequest(`/api/candidates/${id}`, 'DELETE');
        if (res.success) { showToast('Deleted', 'success'); const active = document.querySelector('.task-expander.open'); if (active) loadTaskCandidates(active.id.replace('task-', '')); }
    }
};
window.getSlotSuggestions = getSlotSuggestions;
window.handleSendInvites = handleSendInvites;

window.openRecruiterDetail = async (id) => {
    const modal = document.getElementById('recruiter-detail-modal');
    const body = document.getElementById('recruiter-detail-body');
    modal.classList.add('active');
    body.innerHTML = '<div class="loading-state"><div class="spinner"></div><p>Fetching deep details...</p></div>';

    const res = await apiRequest(`/api/admin/recruiter/${id}`);
    if (res.error) {
        body.innerHTML = `<div class="alert alert-danger">${res.error}</div>`;
        return;
    }

    const { recruiter, workload, recent_sessions } = res;

    body.innerHTML = `
        <div style="background:rgba(255,255,255,0.05); padding:20px; border-radius:12px; margin-bottom:24px;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                    <h2 style="margin:0; font-size:24px;">${recruiter.name}</h2>
                    <p style="color:var(--text-muted); margin:4px 0;">${recruiter.email}</p>
                    <div style="margin-top:10px;">
                        <span class="badge badge-${workload.current_status === 'online' ? 'online' : 'offline'}">${workload.current_status.toUpperCase()}</span>
                        <span style="font-size:12px; margin-left:10px; color:var(--text-muted);">Last active: ${workload.last_active}</span>
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:12px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Workload</div>
                    <div style="font-size:32px; font-weight:800; color:var(--accent);">${workload.workload_percentage}%</div>
                </div>
            </div>
        </div>

        <div class="tabs-container">
            <button class="tab-btn active" id="btn-tab-tasks" onclick="switchDetailTab('tasks')">📋 Assigned Tasks</button>
            <button class="tab-btn" id="btn-tab-history" onclick="switchDetailTab('history')">🕐 Activity History</button>
        </div>

        <div id="detail-tab-tasks" class="detail-tab-pane">
            ${workload.tasks.length === 0 ? '<div class="empty-state">No tasks assigned</div>' : `
                <div class="data-card">
                    <div class="data-card-body">
                        <table class="data-table">
                            <thead><tr><th>Task</th><th>Status</th><th>Priority</th></tr></thead>
                            <tbody>
                                ${workload.tasks.map(t => `
                                    <tr>
                                        <td>
                                            <div style="font-weight:600;">${t.title}</div>
                                            <div style="font-size:11px; color:var(--text-muted);">ID: ${t.id}</div>
                                        </td>
                                        <td><span class="badge badge-${t.status === 'completed' ? 'success' : (t.status === 'pending' ? 'warning' : 'info')}">${t.status}</span></td>
                                        <td>${t.priority}</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            `}
        </div>

        <div id="detail-tab-history" class="detail-tab-pane" style="display:none">
            ${recent_sessions.length === 0 ? '<div class="empty-state">No activity logs found</div>' : `
                <div class="data-card">
                    <div class="data-card-body">
                        <table class="data-table">
                            <thead><tr><th>Login</th><th>Logout</th><th>Minutes</th></tr></thead>
                            <tbody>
                                ${recent_sessions.map(s => `
                                    <tr>
                                        <td>${s.login_time.substring(0, 16).replace('T', ' ')}</td>
                                        <td>${s.logout_time ? s.logout_time.substring(0, 16).replace('T', ' ') : 'Active'}</td>
                                        <td>${s.duration_minutes ? Math.round(s.duration_minutes) : '-'}</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            `}
        </div>
    `;
};

window.closeRecruiterDetailModal = () => {
    document.getElementById('recruiter-detail-modal').classList.remove('active');
};

window.switchDetailTab = (tab) => {
    document.querySelectorAll('.detail-tab-pane').forEach(p => p.style.display = 'none');
    document.getElementById(`detail-tab-${tab}`).style.display = 'block';

    document.querySelectorAll('#recruiter-detail-modal .tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(`btn-tab-${tab === 'tasks' ? 'tasks' : 'history'}`).classList.add('active');
};

window.switchRecPage = (page) => {
    document.getElementById('rec-tab-all').style.display = page === 'all' ? 'block' : 'none';
    document.getElementById('rec-tab-assign').style.display = page === 'assign' ? 'block' : 'none';
    document.querySelectorAll('#page-recruiters .tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(`btn-rec-${page}`).classList.add('active');
    if (page === 'assign') initAssignPage();
};

window.switchAssignMode = (mode) => {
    document.getElementById('assign-existing-wrap').style.display = mode === 'existing' ? 'block' : 'none';
    document.getElementById('assign-new-wrap').style.display = mode === 'new' ? 'block' : 'none';
};

async function initAssignPage() {
    const taskSelect = document.getElementById('assign-task-select');
    const recSelect1 = document.getElementById('assign-rec-select');
    const recSelect2 = document.getElementById('new-task-recs');

    // Fetch Recruiters
    const recs = await apiRequest('/api/admin/recruiters');
    const recOptions = recs.map(r => `<option value="${r.id}">${r.name}</option>`).join('');
    recSelect1.innerHTML = recOptions;
    recSelect2.innerHTML = recOptions;

    // Fetch Tasks
    const tasks = await apiRequest('/api/tasks');
    // Filter for unassigned or relevant ones? In Streamlit it's "Unassigned"
    // Backend doesn't have a specific unassigned filter easily, let's just show all active ones
    const unassigned = tasks.filter(t => !t.assigned_to || t.assigned_to.length === 0);
    taskSelect.innerHTML = unassigned.length > 0
        ? unassigned.map(t => `<option value="${t.id}">${t.title}</option>`).join('')
        : '<option value="">No unassigned tasks found</option>';
}

window.handleAssignExisting = async () => {
    const taskId = document.getElementById('assign-task-select').value;
    const recSelect = document.getElementById('assign-rec-select');
    const recruiterIds = Array.from(recSelect.selectedOptions).map(o => o.value);

    if (!taskId || recruiterIds.length === 0) return showToast('Select task and recruiter(s)', 'warning');

    const res = await apiRequest('/api/admin/assign-task', 'POST', { task_id: taskId, recruiter_ids: recruiterIds });
    if (res.success) {
        showToast('Task assigned!', 'success');
        switchRecPage('all');
        loadRecruiters();
    } else {
        showToast(res.error || 'Failed to assign', 'error');
    }
};

window.handleCreateAssignNew = async () => {
    const data = {
        title: document.getElementById('new-task-title').value,
        priority: document.getElementById('new-task-priority').value,
        location: document.getElementById('new-task-loc').value,
        experience: document.getElementById('new-task-exp').value,
        recruiter_ids: Array.from(document.getElementById('new-task-recs').selectedOptions).map(o => o.value),
        skills: [], // simplified
        comment: "",
        feedback: ""
    };

    if (!data.title || data.recruiter_ids.length === 0) return showToast('Title and recruiter(s) required', 'warning');

    const res = await apiRequest('/api/admin/create-assign-task', 'POST', data);
    if (res.success) {
        showToast('Task created and assigned!', 'success');
        switchRecPage('all');
        loadRecruiters();
    } else {
        showToast(res.error || 'Failed to create', 'error');
    }
};

// Admin globals
window.loadDashboard = loadDashboard;
window.loadRecruiters = loadRecruiters;
window.loadWorkloadReport = loadWorkloadReport;

// Settings Page Implementation
async function loadSettingsPage() {
    const container = document.getElementById('settings-content');
    const admin = JSON.parse(localStorage.getItem('admin_user') || '{}');
    const timestamp = new Date().toLocaleString();
    const apiBase = window.location.origin;

    container.innerHTML = `
        <div class="data-card">
            <div class="data-card-header"><h3>wrench 🔧 Your Admin Profile</h3></div>
            <div class="data-card-body" style="padding:24px;">
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px;">
                    <div>
                        <p><strong>Username:</strong> ${admin.username || 'Unknown'}</p>
                        <p><strong>Email:</strong> ${admin.email || admin.username + '@admin.com'}</p>
                    </div>
                    <div>
                        <p><strong>Role:</strong> Administrator</p>
                        <p><strong>Status:</strong> ✅ Active</p>
                    </div>
                </div>
            </div>
        </div>

        <div class="data-card" style="margin-top:24px;">
            <div class="data-card-header"><h3>📊 System Configuration</h3></div>
            <div class="data-card-body" style="padding:24px;">
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px;">
                    <div>
                        <h4 style="font-size:14px; margin-bottom:10px;">Workload Monitoring</h4>
                        <ul style="padding-left:20px; color:var(--text-secondary); line-height:1.6;">
                            <li><strong>Real-time Updates:</strong> Every 30 seconds</li>
                            <li><strong>Alert Threshold:</strong> 70% workload</li>
                            <li><strong>High Priority Weight:</strong> 15 points per task</li>
                            <li><strong>Normal Task Weight:</strong> 10 points per task</li>
                        </ul>
                    </div>
                    <div>
                        <h4 style="font-size:14px; margin-bottom:10px;">Data Retention</h4>
                         <ul style="padding-left:20px; color:var(--text-secondary); line-height:1.6;">
                            <li><strong>Login Logs:</strong> 90 days</li>
                            <li><strong>Activity History:</strong> 60 days</li>
                            <li><strong>Task History:</strong> Unlimited</li>
                            <li><strong>Workload Snapshots:</strong> 30 days</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>

        <div class="data-card" style="margin-top:24px;">
            <div class="data-card-header"><h3>ℹ️ System Information</h3></div>
             <div class="data-card-body" style="padding:24px;">
                <ul style="padding-left:20px; color:var(--text-secondary); line-height:1.6;">
                    <li><strong>Admin Dashboard Version:</strong> 2.0 (JS/HTML/CSS Optimized)</li>
                    <li><strong>Last Updated:</strong> ${timestamp}</li>
                    <li><strong>Backend API:</strong> ${apiBase}</li>
                    <li><strong>Authentication:</strong> JWT Token-based</li>
                    <li><strong>Database:</strong> MongoDB</li>
                </ul>
            </div>
        </div>
    `;
}

// Custom Time Picker Logic
window.toggleCustomTimePicker = () => {
    const picker = document.getElementById('custom-time-picker');
    picker.style.display = picker.style.display === 'none' ? 'block' : 'none';
};

window.applyCustomTime = () => {
    const h = document.getElementById('custom-h').value;
    const m = document.getElementById('custom-m').value;
    const p = document.getElementById('custom-p').value;

    // Calculate end time (1 hour later)
    let endH = parseInt(h);
    let endP = p;

    if (endH === 11) {
        endP = p === 'AM' ? 'PM' : 'AM';
        endH = 12;
    } else if (endH === 12) {
        endH = 1;
    } else {
        endH = endH + 1;
    }

    const timeStr = `${h}:${m} ${p}`;
    const endStr = `${endH}:${m} ${endP}`;
    const fullSlot = `${timeStr} - ${endStr} (Custom)`;

    // Add to select and select it
    const select = document.getElementById('interview-time-select');
    const opt = document.createElement('option');
    opt.value = fullSlot;
    opt.innerText = fullSlot;
    select.add(opt, select.options[0]); // add to top
    select.value = fullSlot;

    document.getElementById('custom-time-picker').style.display = 'none';
    showToast(`Custom slot added: ${timeStr}`, 'success');
};

// Auto-refresh tasks list on JD extract - wraps original handleExtractJD
// Function to handle JD extraction
window.handleExtractJD = async () => {
    const fileInput = document.getElementById('jd-file-input');
    const textInput = document.getElementById('jd-text-input');
    const resultsDiv = document.getElementById('extract-results');

    // Check if files or text
    if (fileInput.files.length > 0) {
        showToast('Start uploading file...', 'info');
        resultsDiv.innerHTML = '<div class="loading-state"><div class="spinner"></div><p>Uploading & Extracting...</p></div>';

        try {
            const formData = new FormData();
            // Backend expects single file under key 'file' usually, or check API.
            // fronten.py uses: files={"file": file}
            // So we send first file.
            formData.append('file', fileInput.files[0]);

            const res = await apiRequest('/api/upload-requirement', 'POST', formData, true); // true for isFormData

            if (res.success || (res.task_id && !res.error)) {
                // Some endpoints return dict without success:true but with task_id
                showToast('Requirement uploaded & Task created!', 'success');
                resultsDiv.innerHTML = `<div class="alert alert-success">
                    <h4>✅ Task Created Successfully (File)</h4>
                    <p><strong>Title:</strong> ${res.extracted_data?.title || res.title || 'Untitled'}</p>
                    <div style="margin-top:10px;">
                        <button class="btn btn-sm btn-outline" onclick="loadRecTasks()">View in Tasks</button>
                    </div>
                </div>`;
                // Clear input
                fileInput.value = '';
                textInput.value = '';

                // Refresh dashboard and tasks
                await loadRecTasks();
                await loadRecDashboard();
            } else {
                const msg = res.error || res.detail || res.message || 'Upload failed';
                resultsDiv.innerHTML = `<div class="alert alert-danger">❌ Error: ${msg}</div>`;
                showToast(msg, 'error');
            }
        } catch (e) {
            resultsDiv.innerHTML = `<div class="alert alert-danger">❌ Error: ${e.message}</div>`;
            showToast(e.message, 'error');
        }
        return;
    }

    // Text extraction fallback
    const text = textInput.value.trim();

    if (!text) {
        showToast('Please paste a Job Description text', 'warning');
        return;
    }

    resultsDiv.innerHTML = '<div class="loading-state"><div class="spinner"></div><p>Extracting requirements & Creating Task...</p></div>';

    const res = await apiRequest('/api/requirements/extract', 'POST', { content: text });

    if (res.success) {
        showToast('Requirement extracted & Task created!', 'success');
        resultsDiv.innerHTML = `<div class="alert alert-success">
            <h4>✅ Task Created Successfully</h4>
            <p><strong>Title:</strong> ${res.extracted_data.title || 'Untitled'}</p>
            <p><strong>Priority:</strong> ${res.extracted_data.priority || 'Medium'}</p>
            <div style="margin-top:10px;">
                <button class="btn btn-sm btn-outline" onclick="loadRecTasks()">View in Tasks</button>
            </div>
        </div>`;

        // Clear input
        textInput.value = '';

        // Refresh dashboard and tasks
        await loadRecTasks();
        await loadRecDashboard();
    } else {
        resultsDiv.innerHTML = `<div class="alert alert-danger">❌ Error: ${res.detail || res.message || 'Extraction failed'}</div>`;
        showToast(res.detail || 'Extraction failed', 'error');
    }
};

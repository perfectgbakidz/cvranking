"""
FastAPI Glink Semantic Matching System with Multi-Role Authentication
Roles: Recruiter (Admin), Employer, Graduate (Applicant)
"""

import os
import re
import asyncio
import hashlib
import secrets
import gc
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, status, BackgroundTasks, Request, Cookie
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, EmailStr, Field, validator
import aiohttp
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiosmtplib

# Password hashing
from passlib.context import CryptContext

# Document processing
import PyPDF2
import io

# Math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Async SQLite
import aiosqlite

# For creating temporary templates directory
import tempfile
import shutil

# Configuration
DATABASE_FILE = "glink_database.db"
MATCH_THRESHOLD = 0.80  # 80% match threshold



# HuggingFace API for embeddings
HF_API_URL = "https://api-inference.huggingface.co/models/0xnbk/nbk-ats-semantic-v1-en"
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "noreply@glink-system.com")

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBasic(auto_error=False)

# Session management
SESSION_SECRET = os.getenv("SESSION_SECRET", secrets.token_urlsafe(32))
active_sessions: Dict[str, Dict[str, Any]] = {}

# Create temporary directory for templates
TEMP_DIR = tempfile.mkdtemp()
TEMPLATES_DIR = os.path.join(TEMP_DIR, "templates")
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# ============== ENUMS ==============

class UserRole(str, Enum):
    RECRUITER = "recruiter"
    EMPLOYER = "employer"
    GRADUATE = "graduate"

class ApplicationStatus(str, Enum):
    PENDING = "pending"
    MATCHED = "matched"
    REVIEWING = "reviewing"
    INTERVIEW = "interview"
    HIRED = "hired"
    REJECTED = "rejected"
    NOTIFIED_NO_MATCH = "notified_no_match"

# ============== PYDANTIC MODELS ==============

class UserSignup(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: str = Field(..., min_length=2)
    role: UserRole
    company_name: Optional[str] = None
    phone: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class JobDescriptionCreate(BaseModel):
    title: str
    description: str
    department: Optional[str] = None
    location: Optional[str] = None
    requirements: Optional[str] = None
    salary_range: Optional[str] = None

class JobDescriptionResponse(BaseModel):
    id: int
    title: str
    description: str
    department: Optional[str]
    location: Optional[str]
    employer_id: int
    employer_name: str
    embedding_created: bool
    created_at: str

class ResumeUploadResponse(BaseModel):
    id: int
    full_name: str
    email: str
    status: str
    message: str

class MatchResult(BaseModel):
    resume_id: int
    job_id: int
    similarity_score: float
    job_title: str
    employer_id: int

class EmployerStats(BaseModel):
    total_jobs: int
    total_applications: int
    matched_candidates: int
    pending_reviews: int

# ============== HTML TEMPLATES ==============

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Glink Semantic Matching System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px;
        }
        .header {
            text-align: center;
            color: white;
            padding: 40px 0;
        }
        .header h1 { font-size: 3em; margin-bottom: 10px; }
        .header p { font-size: 1.2em; opacity: 0.9; }
        .cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 30px;
            margin-top: 40px;
        }
        .card {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            text-align: center;
            transition: transform 0.3s;
        }
        .card:hover { transform: translateY(-10px); }
        .card-icon {
            font-size: 4em;
            margin-bottom: 20px;
        }
        .card h2 { color: #667eea; margin-bottom: 15px; }
        .card p { color: #666; margin-bottom: 30px; line-height: 1.6; }
        .btn {
            display: inline-block;
            padding: 15px 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 30px;
            font-weight: bold;
            transition: transform 0.3s, box-shadow 0.3s;
            border: none;
            cursor: pointer;
            font-size: 1em;
            margin: 5px;
        }
        .btn:hover {
            transform: scale(1.05);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }
        .btn-secondary {
            background: white;
            color: #667eea;
            border: 2px solid #667eea;
        }
        .features {
            margin-top: 60px;
            color: white;
            text-align: center;
        }
        .features h2 { font-size: 2.5em; margin-bottom: 30px; }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        .feature-item {
            background: rgba(255,255,255,0.1);
            padding: 30px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        .feature-item h3 { margin-bottom: 10px; }
        .footer {
            text-align: center;
            color: white;
            padding: 40px 0;
            margin-top: 60px;
            opacity: 0.8;
        }
        .auth-links {
            margin-top: 20px;
        }
        .auth-links a {
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Smart Glink</h1>
            <p>AI-Powered Resume & Job Matching System</p>
        </div>
        
        <div class="cards">
            <div class="card">
                <div class="card-icon">🎓</div>
                <h2>I'm a Graduate</h2>
                <p>Upload your resume and let our AI match you with the perfect job opportunities. Get instant notifications when matches are found.</p>
                <a href="/signup?role=graduate" class="btn">Sign Up</a>
                <a href="/login?role=graduate" class="btn btn-secondary">Login</a>
            </div>
            
            <div class="card">
                <div class="card-icon">🏢</div>
                <h2>I'm an Employer</h2>
                <p>Post job descriptions, review matched candidates, and manage your hiring pipeline with AI-powered semantic analysis.</p>
                <a href="/signup?role=employer" class="btn">Sign Up</a>
                <a href="/login?role=employer" class="btn btn-secondary">Login</a>
            </div>
            
            <div class="card">
                <div class="card-icon">👔</div>
                <h2>I'm a Recruiter</h2>
                <p>Manage the entire platform, oversee employers and applicants, and review all matching activities from the admin dashboard.</p>
                <a href="/login?role=recruiter" class="btn">Recruiter Login</a>
            </div>
        </div>
        
        <div class="features">
            <h2>How It Works</h2>
            <div class="feature-grid">
                <div class="feature-item">
                    <h3>📝 Upload</h3>
                    <p>Graduates upload PDF resumes</p>
                </div>
                <div class="feature-item">
                    <h3>🏢 Post Jobs</h3>
                    <p>Employers post opportunities</p>
                </div>
                <div class="feature-item">
                    <h3>🤖 AI Analysis</h3>
                    <p>Specialized model matching</p>
                </div>
                <div class="feature-item">
                    <h3>🎯 Matching</h3>
                    <p>80% threshold similarity match</p>
                </div>
                <div class="feature-item">
                    <h3>📧 Notification</h3>
                    <p>Automatic email alerts</p>
                </div>
                <div class="feature-item">
                    <h3>📊 Dashboard</h3>
                    <p>Track candidates & jobs</p>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Powered by HuggingFace Inference API</p>
        </div>
    </div>
</body>
</html>
"""

SIGNUP_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign Up - Glink System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .signup-box {
            background: white;
            padding: 50px;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            width: 100%;
            max-width: 450px;
        }
        .signup-box h1 { color: #667eea; margin-bottom: 10px; text-align: center; }
        .signup-box .role-badge {
            text-align: center;
            background: #f0f0f0;
            padding: 8px 20px;
            border-radius: 20px;
            display: inline-block;
            margin: 0 auto 20px;
            font-weight: 600;
            color: #667eea;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 600;
        }
        .form-group input {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        .form-group input:focus {
            outline: none;
            border-color: #667eea;
        }
        .btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.3s;
        }
        .btn:hover { transform: scale(1.02); }
        .back-link {
            display: block;
            text-align: center;
            margin-top: 20px;
            color: #667eea;
            text-decoration: none;
        }
        .error {
            color: #e74c3c;
            margin-bottom: 20px;
            padding: 10px;
            background: #fee;
            border-radius: 5px;
            text-align: center;
        }
        .success {
            color: #28a745;
            margin-bottom: 20px;
            padding: 10px;
            background: #d4edda;
            border-radius: 5px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="signup-box">
        <h1>📝 Sign Up</h1>
        <div style="text-align: center;">
            <span class="role-badge">{{ role.title() }}</span>
        </div>
        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
        {% if success %}
        <div class="success">{{ success }}</div>
        {% endif %}
        <form action="/signup" method="post" id="signupForm">
            <input type="hidden" name="role" value="{{ role }}">
            <div class="form-group">
                <label>Full Name</label>
                <input type="text" name="full_name" required placeholder="John Doe">
            </div>
            <div class="form-group">
                <label>Email Address</label>
                <input type="email" name="email" required placeholder="john@example.com">
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" name="password" required placeholder="Min 6 characters" minlength="6">
            </div>
            {% if role == 'employer' %}
            <div class="form-group">
                <label>Company Name</label>
                <input type="text" name="company_name" required placeholder="Acme Inc.">
            </div>
            {% endif %}
            <div class="form-group">
                <label>Phone (Optional)</label>
                <input type="tel" name="phone" placeholder="+1 234 567 890">
            </div>
            <button type="submit" class="btn">Create Account</button>
        </form>
        <a href="/login?role={{ role }}" class="back-link">Already have an account? Login →</a>
        <br>
        <a href="/" class="back-link">← Back to Home</a>
    </div>
</body>
</html>
"""

LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - Glink System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .login-box {
            background: white;
            padding: 60px;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            width: 100%;
            max-width: 400px;
            text-align: center;
        }
        .login-box h1 { color: #667eea; margin-bottom: 10px; }
        .login-box .role-badge {
            background: #f0f0f0;
            padding: 8px 20px;
            border-radius: 20px;
            display: inline-block;
            margin-bottom: 30px;
            font-weight: 600;
            color: #667eea;
        }
        .form-group {
            margin-bottom: 20px;
            text-align: left;
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 600;
        }
        .form-group input {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        .form-group input:focus {
            outline: none;
            border-color: #667eea;
        }
        .btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.3s;
        }
        .btn:hover { transform: scale(1.02); }
        .back-link {
            display: inline-block;
            margin-top: 20px;
            color: #667eea;
            text-decoration: none;
        }
        .error {
            color: #e74c3c;
            margin-bottom: 20px;
            padding: 10px;
            background: #fee;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="login-box">
        <h1>🔐 Login</h1>
        <span class="role-badge">{{ role.title() }}</span>
        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
        <form action="/login" method="post">
            <input type="hidden" name="role" value="{{ role }}">
            <div class="form-group">
                <label>Email</label>
                <input type="email" name="email" required placeholder="john@example.com">
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" name="password" required placeholder="••••••••">
            </div>
            <button type="submit" class="btn">Sign In</button>
        </form>
        <a href="/signup?role={{ role }}" class="back-link">Don't have an account? Sign up →</a>
        <br>
        <a href="/" class="back-link">← Back to Home</a>
    </div>
</body>
</html>
"""

EMPLOYER_DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employer Dashboard - Glink System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f7fa;
            min-height: 100vh;
        }
        .navbar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .navbar h1 { font-size: 1.5em; }
        .nav-links a {
            color: white;
            text-decoration: none;
            margin-left: 30px;
            opacity: 0.9;
        }
        .nav-links a:hover { opacity: 1; }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 40px;
        }
        .welcome {
            margin-bottom: 30px;
        }
        .welcome h2 { color: #333; margin-bottom: 5px; }
        .welcome p { color: #666; }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .stat-card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        }
        .stat-card h3 {
            color: #666;
            font-size: 0.85em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .stat-card .number {
            font-size: 2.2em;
            color: #667eea;
            font-weight: bold;
        }
        .section {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        }
        .section h2 {
            color: #333;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .btn {
            padding: 10px 25px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.9em;
            text-decoration: none;
            display: inline-block;
        }
        .btn-success { background: #28a745; }
        .btn-info { background: #17a2b8; }
        .btn-sm { padding: 5px 15px; font-size: 0.8em; }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            text-align: left;
            padding: 15px;
            border-bottom: 1px solid #eee;
        }
        th {
            color: #666;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
        }
        tr:hover { background: #f8f9fa; }
        .badge {
            padding: 5px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }
        .badge-matched { background: #d4edda; color: #155724; }
        .badge-pending { background: #fff3cd; color: #856404; }
        .badge-reviewing { background: #cce5ff; color: #004085; }
        .badge-interview { background: #e2d4f0; color: #4a148c; }
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s;
        }
        .candidate-card {
            border: 1px solid #eee;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
        }
        .candidate-card h4 { color: #333; margin-bottom: 5px; }
        .candidate-card p { color: #666; font-size: 0.9em; margin-bottom: 10px; }
        .candidate-actions {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        .modal {
            display: none;
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }
        .modal.active { display: flex; }
        .modal-content {
            background: white;
            padding: 40px;
            border-radius: 20px;
            width: 90%;
            max-width: 700px;
            max-height: 90vh;
            overflow-y: auto;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 600;
        }
        .form-group input,
        .form-group textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-family: inherit;
        }
        .form-group textarea {
            min-height: 150px;
            resize: vertical;
        }
        .close-btn {
            float: right;
            font-size: 1.5em;
            cursor: pointer;
            color: #666;
        }
        .notification {
            background: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
            display: none;
        }
        .notification.show { display: block; }
    </style>
</head>
<body>
    <div class="navbar">
        <h1>🏢 Employer Portal</h1>
        <div class="nav-links">
            <a href="/employer/dashboard">Dashboard</a>
            <a href="#" onclick="openModal()">+ Post Job</a>
            <a href="/logout">Logout</a>
        </div>
    </div>
    
    <div class="container">
        <div class="welcome">
            <h2>Welcome, {{ user.full_name }}</h2>
            <p>{{ user.company_name or 'Your Company' }}</p>
        </div>
        
        <div id="notification" class="notification">
            <strong>🎉 New Match!</strong> A new candidate has matched one of your job postings.
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>Active Jobs</h3>
                <div class="number">{{ stats.total_jobs }}</div>
            </div>
            <div class="stat-card">
                <h3>Total Applications</h3>
                <div class="number">{{ stats.total_applications }}</div>
            </div>
            <div class="stat-card">
                <h3>Matched Candidates</h3>
                <div class="number">{{ stats.matched_candidates }}</div>
            </div>
            <div class="stat-card">
                <h3>Pending Review</h3>
                <div class="number">{{ stats.pending_reviews }}</div>
            </div>
        </div>
        
        <div class="section">
            <h2>
                My Job Postings
                <button class="btn" onclick="openModal()">+ Post New Job</button>
            </h2>
            {% if jobs %}
            <table>
                <thead>
                    <tr>
                        <th>Title</th>
                        <th>Department</th>
                        <th>Location</th>
                        <th>Posted</th>
                        <th>Matches</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {% for job in jobs %}
                    <tr>
                        <td><strong>{{ job.title }}</strong></td>
                        <td>{{ job.department or 'N/A' }}</td>
                        <td>{{ job.location or 'N/A' }}</td>
                        <td>{{ job.created_at }}</td>
                        <td><span class="badge badge-matched">{{ job.match_count }}</span></td>
                        <td>
                            <a href="/employer/jobs/{{ job.id }}/candidates" class="btn btn-info btn-sm">View Candidates</a>
                            <button class="btn btn-danger btn-sm" onclick="deleteJob({{ job.id }})">Delete</button>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% else %}
            <p style="text-align: center; color: #666; padding: 40px;">No jobs posted yet. Click "Post New Job" to get started!</p>
            {% endif %}
        </div>
        
        <div class="section">
            <h2>Recent Matched Candidates</h2>
            {% if recent_matches %}
                {% for match in recent_matches %}
                <div class="candidate-card">
                    <h4>{{ match.full_name }} — {{ match.job_title }}</h4>
                    <p>📧 {{ match.email }} | 🎯 Match Score: {{ "%.1f"|format(match.match_score * 100) }}%</p>
                    <div class="candidate-actions">
                        <a href="/employer/candidates/{{ match.resume_id }}/cv" class="btn btn-info btn-sm" target="_blank">📄 View CV</a>
                        <a href="mailto:{{ match.email }}" class="btn btn-success btn-sm">📧 Email Candidate</a>
                        <button class="btn btn-sm" onclick="updateStatus({{ match.resume_id }}, 'interview')">📅 Schedule Interview</button>
                    </div>
                </div>
                {% endfor %}
            {% else %}
            <p style="text-align: center; color: #666; padding: 40px;">No matched candidates yet. Post jobs to start receiving matches!</p>
            {% endif %}
        </div>
    </div>
    
    <div id="jobModal" class="modal">
        <div class="modal-content">
            <span class="close-btn" onclick="closeModal()">&times;</span>
            <h2>Post New Job</h2>
            <form id="jobForm" onsubmit="submitJob(event)">
                <div class="form-group">
                    <label>Job Title *</label>
                    <input type="text" name="title" required placeholder="e.g., Senior Python Developer">
                </div>
                <div class="form-group">
                    <label>Department</label>
                    <input type="text" name="department" placeholder="e.g., Engineering">
                </div>
                <div class="form-group">
                    <label>Location</label>
                    <input type="text" name="location" placeholder="e.g., Remote, New York, NY">
                </div>
                <div class="form-group">
                    <label>Salary Range</label>
                    <input type="text" name="salary_range" placeholder="e.g., $80k - $120k">
                </div>
                <div class="form-group">
                    <label>Job Description *</label>
                    <textarea name="description" required placeholder="Enter detailed job description..."></textarea>
                </div>
                <div class="form-group">
                    <label>Requirements</label>
                    <textarea name="requirements" placeholder="Enter job requirements, skills needed..."></textarea>
                </div>
                <button type="submit" class="btn" style="width: 100%;">Post Job & Generate Embedding</button>
            </form>
        </div>
    </div>
    
    <script>
        function openModal() {
            document.getElementById('jobModal').classList.add('active');
        }
        function closeModal() {
            document.getElementById('jobModal').classList.remove('active');
        }
        
        // FIXED: Use FormData directly instead of JSON.stringify
        async function submitJob(e) {
            e.preventDefault();
            const formData = new FormData(e.target);
            
            try {
                const response = await fetch('/employer/jobs', {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    alert('Job posted successfully!');
                    location.reload();
                } else if (response.status === 401) {
                    alert('Session expired. Please login again.');
                    window.location.href = '/login?role=employer';
                } else {
                    const err = await response.json();
                    alert('Error: ' + (err.detail || 'Something went wrong'));
                }
            } catch (err) {
                alert('Error: ' + err.message);
            }
        }
        
        async function deleteJob(id) {
            if (!confirm('Delete this job posting?')) return;
            try {
                const response = await fetch(`/employer/jobs/${id}`, { method: 'DELETE' });
                if (response.ok) location.reload();
                else alert('Error deleting job');
            } catch (err) {
                alert('Error: ' + err.message);
            }
        }
        
        async function updateStatus(resumeId, status) {
            try {
                const response = await fetch(`/employer/candidates/${resumeId}/status`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: new URLSearchParams({ status: status, job_id: '1' })
                });
                if (response.ok) {
                    alert('Status updated!');
                    location.reload();
                }
            } catch (err) {
                alert('Error: ' + err.message);
            }
        }
        
        // Check for new matches every 30 seconds
        setInterval(async () => {
            try {
                const response = await fetch('/employer/notifications');
                const data = await response.json();
                if (data.new_matches > 0) {
                    document.getElementById('notification').classList.add('show');
                    setTimeout(() => {
                        document.getElementById('notification').classList.remove('show');
                    }, 5000);
                }
            } catch (e) {}
        }, 30000);
        
        window.onclick = function(e) {
            if (e.target.classList.contains('modal')) closeModal();
        }
    </script>
</body>
</html>
"""

GRADUATE_DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My Dashboard - Glink System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f7fa;
            min-height: 100vh;
        }
        .navbar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .navbar h1 { font-size: 1.5em; }
        .nav-links a {
            color: white;
            text-decoration: none;
            margin-left: 30px;
            opacity: 0.9;
        }
        .nav-links a:hover { opacity: 1; }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px;
        }
        .welcome {
            margin-bottom: 30px;
        }
        .welcome h2 { color: #333; margin-bottom: 5px; }
        .welcome p { color: #666; }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .stat-card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        }
        .stat-card h3 {
            color: #666;
            font-size: 0.85em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .stat-card .number {
            font-size: 2.2em;
            color: #667eea;
            font-weight: bold;
        }
        .section {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        }
        .section h2 {
            color: #333;
            margin-bottom: 20px;
        }
        .btn {
            padding: 12px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.95em;
            text-decoration: none;
            display: inline-block;
            font-weight: 600;
        }
        .match-card {
            border: 1px solid #eee;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 15px;
            transition: transform 0.2s;
        }
        .match-card:hover {
            transform: translateX(5px);
            border-color: #667eea;
        }
        .match-card h3 { color: #333; margin-bottom: 8px; }
        .match-card .company { color: #667eea; font-weight: 600; margin-bottom: 5px; }
        .match-card p { color: #666; font-size: 0.9em; margin-bottom: 10px; }
        .score-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9em;
        }
        .score-high { background: #d4edda; color: #155724; }
        .score-medium { background: #fff3cd; color: #856404; }
        .progress-bar {
            width: 100%;
            height: 10px;
            background: #e0e0e0;
            border-radius: 5px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 5px;
        }
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 600;
        }
        .status-matched { background: #d4edda; color: #155724; }
        .status-interview { background: #cce5ff; color: #004085; }
        .status-hired { background: #d1ecf1; color: #0c5460; }
        .status-rejected { background: #f8d7da; color: #721c24; }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #666;
        }
        .empty-state .icon { font-size: 4em; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="navbar">
        <h1>🎓 Graduate Portal</h1>
        <div class="nav-links">
            <a href="/graduate/dashboard">Dashboard</a>
            <a href="/graduate/upload-resume">Upload Resume</a>
            <a href="/logout">Logout</a>
        </div>
    </div>
    
    <div class="container">
        <div class="welcome">
            <h2>Welcome, {{ user.full_name }}</h2>
            <p>{{ user.email }}</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>Resumes Uploaded</h3>
                <div class="number">{{ stats.total_resumes }}</div>
            </div>
            <div class="stat-card">
                <h3>Job Matches</h3>
                <div class="number">{{ stats.total_matches }}</div>
            </div>
            <div class="stat-card">
                <h3>Interviews</h3>
                <div class="number">{{ stats.interviews }}</div>
            </div>
            <div class="stat-card">
                <h3>Pending</h3>
                <div class="number">{{ stats.pending }}</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📄 Upload New Resume</h2>
            <p style="color: #666; margin-bottom: 20px;">Upload an updated resume to get new job matches</p>
            <a href="/graduate/upload-resume" class="btn">Upload Resume</a>
        </div>
        
        <div class="section">
            <h2>🎯 Your Job Matches</h2>
            {% if matches %}
                {% for match in matches %}
                <div class="match-card">
                    <h3>{{ match.job_title }}</h3>
                    <div class="company">🏢 {{ match.company_name }} — {{ match.department or 'N/A' }}</div>
                    <p>📍 {{ match.location or 'Location not specified' }}</p>
                    <div style="display: flex; align-items: center; gap: 15px; margin: 10px 0;">
                        <span class="score-badge score-high">{{ "%.1f"|format(match.match_score * 100) }}% Match</span>
                        <span class="status-badge status-{{ match.status }}">{{ match.status.title() }}</span>
                    </div>
                    <div class="progress-bar" style="width: 200px;">
                        <div class="progress-fill" style="width: {{ match.match_score * 100 }}%"></div>
                    </div>
                    <p style="margin-top: 10px; font-size: 0.85em; color: #888;">Matched on {{ match.matched_at }}</p>
                </div>
                {% endfor %}
            {% else %}
            <div class="empty-state">
                <div class="icon">🔍</div>
                <h3>No matches yet</h3>
                <p>Upload your resume and we'll match you with the perfect jobs!</p>
            </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

RECRUITER_DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Recruiter Dashboard - Glink System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f7fa;
            min-height: 100vh;
        }
        .navbar {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 20px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .navbar h1 { font-size: 1.5em; }
        .nav-links a {
            color: white;
            text-decoration: none;
            margin-left: 30px;
            opacity: 0.9;
        }
        .nav-links a:hover { opacity: 1; }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 40px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .stat-card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        }
        .stat-card h3 {
            color: #666;
            font-size: 0.85em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .stat-card .number {
            font-size: 2.2em;
            color: #1a1a2e;
            font-weight: bold;
        }
        .section {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        }
        .section h2 {
            color: #333;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .btn {
            padding: 10px 25px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.9em;
            text-decoration: none;
            display: inline-block;
        }
        .btn-danger { background: #e74c3c; }
        .btn-warning { background: #f39c12; }
        .btn-sm { padding: 5px 15px; font-size: 0.8em; }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            text-align: left;
            padding: 15px;
            border-bottom: 1px solid #eee;
        }
        th {
            color: #666;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
        }
        tr:hover { background: #f8f9fa; }
        .badge {
            padding: 5px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }
        .badge-employer { background: #e3f2fd; color: #1565c0; }
        .badge-graduate { background: #f3e5f5; color: #6a1b9a; }
        .badge-recruiter { background: #fff3e0; color: #e65100; }
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border-radius: 8px;
            font-weight: 600;
            color: #666;
        }
        .tab.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
    </style>
</head>
<body>
    <div class="navbar">
        <h1>👔 Recruiter Admin</h1>
        <div class="nav-links">
            <a href="/recruiter/dashboard">Dashboard</a>
            <a href="/logout">Logout</a>
        </div>
    </div>
    
    <div class="container">
        <div class="stats">
            <div class="stat-card">
                <h3>Total Employers</h3>
                <div class="number">{{ stats.total_employers }}</div>
            </div>
            <div class="stat-card">
                <h3>Total Graduates</h3>
                <div class="number">{{ stats.total_graduates }}</div>
            </div>
            <div class="stat-card">
                <h3>Total Jobs</h3>
                <div class="number">{{ stats.total_jobs }}</div>
            </div>
            <div class="stat-card">
                <h3>Total Matches</h3>
                <div class="number">{{ stats.total_matches }}</div>
            </div>
            <div class="stat-card">
                <h3>Pending Reviews</h3>
                <div class="number">{{ stats.pending_reviews }}</div>
            </div>
        </div>
        
        <div class="tabs">
            <div class="tab active" onclick="showTab('employers')">Employers</div>
            <div class="tab" onclick="showTab('graduates')">Graduates</div>
            <div class="tab" onclick="showTab('jobs')">All Jobs</div>
            <div class="tab" onclick="showTab('matches')">All Matches</div>
        </div>
        
        <div id="employers" class="tab-content active">
            <div class="section">
                <h2>Registered Employers</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Email</th>
                            <th>Company</th>
                            <th>Jobs Posted</th>
                            <th>Joined</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for emp in employers %}
                        <tr>
                            <td><strong>{{ emp.full_name }}</strong></td>
                            <td>{{ emp.email }}</td>
                            <td>{{ emp.company_name or 'N/A' }}</td>
                            <td>{{ emp.job_count }}</td>
                            <td>{{ emp.created_at }}</td>
                            <td>
                                <button class="btn btn-danger btn-sm" onclick="deleteUser({{ emp.id }}, 'employer')">Delete</button>
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div id="graduates" class="tab-content">
            <div class="section">
                <h2>Registered Graduates</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Email</th>
                            <th>Resumes</th>
                            <th>Matches</th>
                            <th>Joined</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for grad in graduates %}
                        <tr>
                            <td><strong>{{ grad.full_name }}</strong></td>
                            <td>{{ grad.email }}</td>
                            <td>{{ grad.resume_count }}</td>
                            <td>{{ grad.match_count }}</td>
                            <td>{{ grad.created_at }}</td>
                            <td>
                                <button class="btn btn-danger btn-sm" onclick="deleteUser({{ grad.id }}, 'graduate')">Delete</button>
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div id="jobs" class="tab-content">
            <div class="section">
                <h2>All Job Postings</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Title</th>
                            <th>Employer</th>
                            <th>Department</th>
                            <th>Location</th>
                            <th>Posted</th>
                            <th>Matches</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for job in all_jobs %}
                        <tr>
                            <td><strong>{{ job.title }}</strong></td>
                            <td>{{ job.employer_name }}</td>
                            <td>{{ job.department or 'N/A' }}</td>
                            <td>{{ job.location or 'N/A' }}</td>
                            <td>{{ job.created_at }}</td>
                            <td>{{ job.match_count }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div id="matches" class="tab-content">
            <div class="section">
                <h2>All Matches</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Candidate</th>
                            <th>Job</th>
                            <th>Employer</th>
                            <th>Score</th>
                            <th>Status</th>
                            <th>Date</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for match in all_matches %}
                        <tr>
                            <td>{{ match.candidate_name }}</td>
                            <td>{{ match.job_title }}</td>
                            <td>{{ match.employer_name }}</td>
                            <td>{{ "%.1f"|format(match.match_score * 100) }}%</td>
                            <td><span class="badge badge-{{ match.status }}">{{ match.status }}</span></td>
                            <td>{{ match.matched_at }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script>
        function showTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tabName).classList.add('active');
        }
        
        async function deleteUser(userId, role) {
            if (!confirm(`Delete this ${role}? This cannot be undone.`)) return;
            try {
                const response = await fetch(`/recruiter/users/${userId}?role=${role}`, {
                    method: 'DELETE'
                });
                if (response.ok) location.reload();
                else alert('Error deleting user');
            } catch (err) {
                alert('Error: ' + err.message);
            }
        }
    </script>
</body>
</html>
"""

RESUME_UPLOAD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upload Resume - Glink System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .upload-container {
            background: white;
            padding: 50px;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            width: 100%;
            max-width: 600px;
            text-align: center;
        }
        .upload-container h1 {
            color: #667eea;
            margin-bottom: 10px;
        }
        .upload-container > p {
            color: #666;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 25px;
            text-align: left;
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 600;
        }
        .form-group input[type="text"],
        .form-group input[type="email"] {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        .form-group input:focus {
            outline: none;
            border-color: #667eea;
        }
        .file-input-wrapper {
            position: relative;
            overflow: hidden;
            display: inline-block;
            width: 100%;
        }
        .file-input-wrapper input[type=file] {
            position: absolute;
            left: -9999px;
        }
        .file-input-label {
            display: block;
            padding: 40px 20px;
            border: 3px dashed #ddd;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s;
            text-align: center;
        }
        .file-input-label:hover {
            border-color: #667eea;
            background: #f8f9ff;
        }
        .file-input-label.has-file {
            border-color: #28a745;
            background: #d4edda;
            color: #155724;
        }
        .btn {
            width: 100%;
            padding: 18px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.3s;
        }
        .btn:hover:not(:disabled) {
            transform: scale(1.02);
        }
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .back-link {
            display: inline-block;
            margin-top: 20px;
            color: #667eea;
            text-decoration: none;
        }
        .success-message {
            display: none;
            background: #d4edda;
            color: #155724;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .spinner {
            display: none;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
            vertical-align: middle;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="upload-container">
        <h1>📄 Upload Your Resume</h1>
        <p>Our AI will match you with the perfect job opportunities</p>
        
        <div id="successMsg" class="success-message">
            <strong>✅ Success!</strong><br>
            Your resume has been uploaded. Check your dashboard for matching results!
        </div>
        
        <form id="resumeForm" onsubmit="submitResume(event)">
            <div class="form-group">
                <label>Full Name</label>
                <input type="text" name="full_name" required placeholder="John Doe" value="{{ user.full_name }}">
            </div>
            
            <div class="form-group">
                <label>Email Address</label>
                <input type="email" name="email" required placeholder="john@example.com" value="{{ user.email }}">
            </div>
            
            <div class="form-group">
                <label>Resume (PDF only)</label>
                <div class="file-input-wrapper">
                    <input type="file" id="resume_file" name="resume_file" accept=".pdf" required onchange="updateFileName(this)">
                    <label for="resume_file" class="file-input-label" id="fileLabel">
                        <div>📁 Click to upload PDF</div>
                        <small>Maximum file size: 10MB</small>
                    </label>
                </div>
            </div>
            
            <button type="submit" class="btn" id="submitBtn">
                <span class="spinner" id="spinner"></span>
                <span id="btnText">Upload & Find Matches</span>
            </button>
        </form>
        
        <a href="/graduate/dashboard" class="back-link">← Back to Dashboard</a>
    </div>
    
    <script>
        function updateFileName(input) {
            const label = document.getElementById('fileLabel');
            if (input.files && input.files[0]) {
                label.classList.add('has-file');
                label.innerHTML = `<div>✅ ${input.files[0].name}</div><small>Click to change file</small>`;
            }
        }
        
        async function submitResume(e) {
            e.preventDefault();
            
            const btn = document.getElementById('submitBtn');
            const spinner = document.getElementById('spinner');
            const btnText = document.getElementById('btnText');
            const successMsg = document.getElementById('successMsg');
            
            btn.disabled = true;
            spinner.style.display = 'inline-block';
            btnText.textContent = 'Processing...';
            
            const formData = new FormData(e.target);
            
            try {
                const response = await fetch('/graduate/upload-resume', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    successMsg.style.display = 'block';
                    document.getElementById('resumeForm').reset();
                    document.getElementById('fileLabel').classList.remove('has-file');
                    document.getElementById('fileLabel').innerHTML = '<div>📁 Click to upload PDF</div><small>Maximum file size: 10MB</small>';
                    setTimeout(() => {
                        window.location.href = '/graduate/dashboard';
                    }, 2000);
                } else {
                    alert('Error: ' + (result.detail || 'Something went wrong'));
                }
            } catch (err) {
                alert('Error: ' + err.message);
            } finally {
                btn.disabled = false;
                spinner.style.display = 'none';
                btnText.textContent = 'Upload & Find Matches';
            }
        }
    </script>
</body>
</html>
"""

CANDIDATE_DETAIL_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Candidate Details - Glink System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f7fa;
            min-height: 100vh;
        }
        .navbar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .navbar a { color: white; text-decoration: none; }
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 40px;
        }
        .candidate-header {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        }
        .candidate-header h1 { color: #333; margin-bottom: 10px; }
        .candidate-header p { color: #666; margin-bottom: 5px; }
        .score-display {
            font-size: 3em;
            color: #667eea;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
        }
        .section {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        }
        .section h2 { color: #333; margin-bottom: 15px; }
        .section pre {
            white-space: pre-wrap;
            font-family: inherit;
            color: #555;
            line-height: 1.6;
            max-height: 400px;
            overflow-y: auto;
        }
        .actions {
            display: flex;
            gap: 15px;
            margin-top: 20px;
        }
        .btn {
            padding: 12px 30px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            text-decoration: none;
            display: inline-block;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .btn-success { background: #28a745; color: white; }
        .btn-warning { background: #f39c12; color: white; }
        .btn-danger { background: #e74c3c; color: white; }
        .status-form {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        select {
            padding: 10px;
            border-radius: 8px;
            border: 2px solid #ddd;
            font-size: 1em;
        }
    </style>
</head>
<body>
    <div class="navbar">
        <h1>👤 Candidate Profile</h1>
        <a href="/employer/dashboard">← Back to Dashboard</a>
    </div>
    
    <div class="container">
        <div class="candidate-header">
            <h1>{{ candidate.full_name }}</h1>
            <p>📧 {{ candidate.email }}</p>
            <p>🎯 Applied for: <strong>{{ job.title }}</strong></p>
            <div class="score-display">{{ "%.1f"|format(match.match_score * 100) }}%</div>
            <p style="text-align: center; color: #666;">Match Score</p>
        </div>
        
        <div class="section">
            <h2>Update Status</h2>
            <form action="/employer/candidates/{{ candidate.id }}/status" method="post" class="status-form">
                <input type="hidden" name="job_id" value="{{ job.id }}">
                <select name="status">
                    <option value="matched" {% if match.status == 'matched' %}selected{% endif %}>Matched</option>
                    <option value="reviewing" {% if match.status == 'reviewing' %}selected{% endif %}>Reviewing</option>
                    <option value="interview" {% if match.status == 'interview' %}selected{% endif %}>Interview Scheduled</option>
                    <option value="hired" {% if match.status == 'hired' %}selected{% endif %}>Hired</option>
                    <option value="rejected" {% if match.status == 'rejected' %}selected{% endif %}>Rejected</option>
                </select>
                <button type="submit" class="btn btn-primary">Update Status</button>
            </form>
        </div>
        
        <div class="section">
            <h2>Resume Content</h2>
            <pre>{{ candidate.resume_text }}</pre>
        </div>
        
        <div class="section">
            <h2>Actions</h2>
            <div class="actions">
                <a href="mailto:{{ candidate.email }}?subject=Regarding your application for {{ job.title }}" class="btn btn-success">📧 Email Candidate</a>
                <a href="/employer/dashboard" class="btn btn-primary">← Back to Dashboard</a>
            </div>
        </div>
    </div>
</body>
</html>
"""

# ============== TEMPLATE SETUP ==============

def setup_templates():
    """Write HTML templates to temporary files"""
    templates = {
        "index.html": INDEX_TEMPLATE,
        "signup.html": SIGNUP_TEMPLATE,
        "login.html": LOGIN_TEMPLATE,
        "employer_dashboard.html": EMPLOYER_DASHBOARD_TEMPLATE,
        "graduate_dashboard.html": GRADUATE_DASHBOARD_TEMPLATE,
        "recruiter_dashboard.html": RECRUITER_DASHBOARD_TEMPLATE,
        "resume_upload.html": RESUME_UPLOAD_TEMPLATE,
        "candidate_detail.html": CANDIDATE_DETAIL_TEMPLATE,
    }
    
    for filename, content in templates.items():
        filepath = os.path.join(TEMPLATES_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
    
    return TEMPLATES_DIR

# ============== DATABASE ==============

async def init_database():
    """Initialize SQLite database with required tables"""
    async with aiosqlite.connect(DATABASE_FILE) as db:
        # Users table (unified for all roles)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('recruiter', 'employer', 'graduate')),
                company_name TEXT,
                phone TEXT,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Job descriptions (now linked to employer)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS job_descriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employer_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                department TEXT,
                location TEXT,
                requirements TEXT,
                salary_range TEXT,
                embedding BLOB,
                embedding_created BOOLEAN DEFAULT 0,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (employer_id) REFERENCES users(id)
            )
        """)
        
        # Resumes (linked to graduate user)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS resumes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL,
                resume_text TEXT NOT NULL,
                embedding BLOB,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Matches table (explicit many-to-many with status tracking)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_id INTEGER NOT NULL,
                job_id INTEGER NOT NULL,
                employer_id INTEGER NOT NULL,
                match_score REAL NOT NULL,
                status TEXT DEFAULT 'matched',
                notified BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (resume_id) REFERENCES resumes(id),
                FOREIGN KEY (job_id) REFERENCES job_descriptions(id),
                FOREIGN KEY (employer_id) REFERENCES users(id),
                UNIQUE(resume_id, job_id)
            )
        """)
        
        # Notifications table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                type TEXT NOT NULL,
                message TEXT NOT NULL,
                is_read BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Create indexes
        await db.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_resumes_user ON resumes(user_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_resumes_status ON resumes(status)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_jobs_employer ON job_descriptions(employer_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_matches_resume ON matches(resume_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_matches_job ON matches(job_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_matches_employer ON matches(employer_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications(user_id)")
        
        await db.commit()
        
        # Create default recruiter if not exists
        cursor = await db.execute("SELECT id FROM users WHERE role = 'recruiter' LIMIT 1")
        if not await cursor.fetchone():
            hashed = pwd_context.hash("recruiter123")
            await db.execute("""
                INSERT INTO users (email, password_hash, full_name, role, company_name)
                VALUES (?, ?, ?, ?, ?)
            """, ("recruiter@glink.com", hashed, "System Recruiter", "recruiter", "Glink Platform"))
            await db.commit()
            print("Created default recruiter: recruiter@glink.com / recruiter123")

# ============== AUTH HELPERS ==============

def create_session(user_id: int, role: str) -> str:
    """Create a secure session token"""
    session_id = secrets.token_urlsafe(32)
    active_sessions[session_id] = {
        "user_id": user_id,
        "role": role,
        "created_at": datetime.utcnow()
    }
    return session_id

def get_session(session_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Get session data if valid"""
    if not session_id or session_id not in active_sessions:
        return None
    session = active_sessions[session_id]
    # Check expiry (24 hours)
    if datetime.utcnow() - session["created_at"] > timedelta(hours=24):
        del active_sessions[session_id]
        return None
    return session

async def get_current_user(request: Request, required_role: Optional[UserRole] = None):
    """Get current user from session with optional role check"""
    session_id = request.cookies.get("session_id")
    session = get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    if required_role and session["role"] != required_role.value:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    async with aiosqlite.connect(DATABASE_FILE) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],))
        user = await cursor.fetchone()
        
    if not user or not user["is_active"]:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    
    return dict(user)

async def get_recruiter(request: Request):
    return await get_current_user(request, UserRole.RECRUITER)

async def get_employer(request: Request):
    return await get_current_user(request, UserRole.EMPLOYER)

async def get_graduate(request: Request):
    return await get_current_user(request, UserRole.GRADUATE)

async def get_any_user(request: Request):
    return await get_current_user(request)

# ============== PDF & EMAIL ==============

def extract_text_from_pdf(pdf_file: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

async def send_email(to_email: str, subject: str, body: str, html_body: Optional[str] = None):
    """Send email notification"""
    if not SMTP_USER or not SMTP_PASS:
        print(f"[EMAIL LOG] To: {to_email}\nSubject: {subject}\nBody: {body[:200]}...\n")
        return True
    
    try:
        message = MIMEMultipart("alternative")
        message["From"] = FROM_EMAIL
        message["To"] = to_email
        message["Subject"] = subject
        
        message.attach(MIMEText(body, "plain"))
        if html_body:
            message.attach(MIMEText(html_body, "html"))
        
        await aiosmtplib.send(
            message,
            hostname=SMTP_HOST,
            port=SMTP_PORT,
            start_tls=True,
            username=SMTP_USER,
            password=SMTP_PASS
        )
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

# ============== EMBEDDINGS ==============

async def create_embedding_api(text: str) -> bytes:
    """Create embedding using HuggingFace Inference API"""
    max_chars = 5000
    if len(text) > max_chars:
        text = text[:max_chars]
    
    headers = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                HF_API_URL,
                headers=headers,
                json={"inputs": text}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], list):
                            embedding = np.mean(result, axis=0)
                        else:
                            embedding = np.array(result)
                        return pickle.dumps(embedding)
                    else:
                        raise Exception("Invalid API response format")
                else:
                    error_text = await response.text()
                    raise Exception(f"API error: {response.status}")
    except Exception as e:
        print(f"Primary API failed ({e}), trying fallback...")
        fallback_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    fallback_url,
                    headers=headers,
                    json={"inputs": text}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, list) and len(result) > 0:
                            if isinstance(result[0], list):
                                embedding = np.mean(result, axis=0)
                            else:
                                embedding = np.array(result)
                            return pickle.dumps(embedding)
                        else:
                            raise Exception("Invalid fallback API response format")
                    else:
                        raise Exception(f"Fallback API error: {response.status}")
        except Exception as e2:
            print(f"Fallback API also failed ({e2}), using hash-based embedding")
            words = set(text.lower().split())
            embedding = np.zeros(384)
            for i, word in enumerate(list(words)[:384]):
                hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
                embedding[i] = (hash_val % 1000) / 1000.0
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return pickle.dumps(embedding)

def get_embedding_from_bytes(embedding_blob: bytes) -> np.ndarray:
    """Deserialize embedding from bytes"""
    return pickle.loads(embedding_blob)

def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings"""
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return float(similarity)

# ============== MATCHING LOGIC ==============

async def find_matching_jobs(resume_embedding: np.ndarray, db, exclude_job_ids: List[int] = None) -> List[Dict]:
    """Find all matching jobs for a resume"""
    cursor = await db.execute(
        "SELECT id, employer_id, title, description, embedding FROM job_descriptions WHERE embedding_created = 1 AND is_active = 1"
    )
    jobs = await cursor.fetchall()
    
    matches = []
    for job in jobs:
        if exclude_job_ids and job["id"] in exclude_job_ids:
            continue
            
        if job["embedding"]:
            job_embedding = get_embedding_from_bytes(job["embedding"])
            score = calculate_similarity(resume_embedding, job_embedding)
            
            if score >= MATCH_THRESHOLD:
                matches.append({
                    "job_id": job["id"],
                    "employer_id": job["employer_id"],
                    "title": job["title"],
                    "description": job["description"],
                    "score": score
                })
    
    # Sort by score descending
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches

async def process_resume_matching(resume_id: int, user_id: int):
    """Process resume matching and create match records"""
    
    async with aiosqlite.connect(DATABASE_FILE) as db:
        db.row_factory = aiosqlite.Row
        
        # Get resume
        cursor = await db.execute("SELECT * FROM resumes WHERE id = ?", (resume_id,))
        resume = await cursor.fetchone()
        
        if not resume:
            return
        
        resume_embedding = get_embedding_from_bytes(resume["embedding"])
        
        # Find existing matches to avoid duplicates
        cursor = await db.execute("SELECT job_id FROM matches WHERE resume_id = ?", (resume_id,))
        existing = [row["job_id"] for row in await cursor.fetchall()]
        
        # Find matching jobs
        matches = await find_matching_jobs(resume_embedding, db, existing)
        
        for match in matches:
            # Create match record
            await db.execute("""
                INSERT INTO matches (resume_id, job_id, employer_id, match_score, status)
                VALUES (?, ?, ?, ?, 'matched')
            """, (resume_id, match["job_id"], match["employer_id"], match["score"]))
            
            # Notify employer
            await db.execute("""
                INSERT INTO notifications (user_id, type, message)
                VALUES (?, 'new_match', ?)
            """, (match["employer_id"], f"New candidate match for {match['title']}: {resume['full_name']} ({match['score']*100:.1f}%)"))
            
            # Get employer email
            cursor = await db.execute("SELECT email FROM users WHERE id = ?", (match["employer_id"],))
            employer = await cursor.fetchone()
            
            if employer:
                subject = f"🎯 New Candidate Match: {match['title']}"
                body = f"""
Dear Employer,

A new candidate has matched your job posting!

Position: {match['title']}
Candidate: {resume['full_name']}
Match Score: {match['score']*100:.1f}%

View the candidate in your dashboard: /employer/dashboard

Best regards,
Glink Recruitment Team
"""
                await send_email(employer["email"], subject, body)
        
        # Update resume status
        if matches:
            await db.execute("""
                UPDATE resumes SET status = 'matched', updated_at = CURRENT_TIMESTAMP WHERE id = ?
            """, (resume_id,))
        else:
            await db.execute("""
                UPDATE resumes SET status = 'notified_no_match', updated_at = CURRENT_TIMESTAMP WHERE id = ?
            """, (resume_id,))
            
            # Notify graduate of no match
            subject = "Application Received - Updates Coming Soon"
            body = f"""
Dear {resume['full_name']},

Thank you for submitting your resume. While we don't currently have an opening that matches your qualifications at the 80% threshold, we are continuously adding new positions.

You will receive an automatic notification as soon as a matching position becomes available.

Best regards,
Glink Recruitment Team
"""
            await send_email(resume["email"], subject, body)
        
        await db.commit()

async def reprocess_pending_resumes(new_job_id: int, employer_id: int):
    """Reprocess pending resumes when new job is added"""
    
    async with aiosqlite.connect(DATABASE_FILE) as db:
        db.row_factory = aiosqlite.Row
        
        # Get new job embedding
        cursor = await db.execute(
            "SELECT embedding, title, description FROM job_descriptions WHERE id = ? AND employer_id = ?", 
            (new_job_id, employer_id)
        )
        job = await cursor.fetchone()
        
        if not job or not job["embedding"]:
            return
        
        job_embedding = get_embedding_from_bytes(job["embedding"])
        
        # Get all pending resumes not already matched to this job
        cursor = await db.execute("""
            SELECT r.* FROM resumes r
            WHERE r.status = 'pending'
            AND r.id NOT IN (SELECT resume_id FROM matches WHERE job_id = ?)
        """, (new_job_id,))
        pending_resumes = await cursor.fetchall()
        
        new_matches = 0
        for resume in pending_resumes:
            if resume["embedding"]:
                resume_embedding = get_embedding_from_bytes(resume["embedding"])
                score = calculate_similarity(job_embedding, resume_embedding)
                
                if score >= MATCH_THRESHOLD:
                    # Create match
                    await db.execute("""
                        INSERT INTO matches (resume_id, job_id, employer_id, match_score, status)
                        VALUES (?, ?, ?, ?, 'matched')
                    """, (resume["id"], new_job_id, employer_id, score))
                    
                    # Notify employer
                    await db.execute("""
                        INSERT INTO notifications (user_id, type, message)
                        VALUES (?, 'new_match', ?)
                    """, (employer_id, f"New candidate match for {job['title']}: {resume['full_name']} ({score*100:.1f}%)"))
                    
                    new_matches += 1
        
        await db.commit()
        
        if new_matches > 0:
            # Get employer email
            cursor = await db.execute("SELECT email FROM users WHERE id = ?", (employer_id,))
            employer = await cursor.fetchone()
            if employer:
                subject = f"🎯 {new_matches} New Candidate(s) Matched!"
                body = f"""
Your new job posting has received {new_matches} candidate match(es)!

View them in your employer dashboard.

Best regards,
Glink Recruitment Team
"""
                await send_email(employer["email"], subject, body)

# ============== FASTAPI APP ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    setup_templates()
    print("Initializing database...")
    await init_database()
    print("Multi-role Glink system initialized")
    yield
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    print("Shutting down...")

app = FastAPI(
    title="Glink Semantic Matching API - Multi-Role",
    description="AI-powered Glink with Recruiter, Employer, and Graduate roles",
    version="3.0.0",
    lifespan=lifespan
)

templates = Jinja2Templates(directory=TEMPLATES_DIR)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== PUBLIC ROUTES ==============

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request, role: str = "graduate", error: Optional[str] = None, success: Optional[str] = None):
    """Signup page"""
    if role not in ["employer", "graduate"]:
        role = "graduate"
    return templates.TemplateResponse("signup.html", {
        "request": request, 
        "role": role,
        "error": error,
        "success": success
    })

@app.post("/signup")
async def signup(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    full_name: str = Form(...),
    role: str = Form(...),
    company_name: Optional[str] = Form(None),
    phone: Optional[str] = Form(None)
):
    """Handle signup"""
    if role not in ["employer", "graduate"]:
        return templates.TemplateResponse("signup.html", {
            "request": request,
            "role": role,
            "error": "Invalid role selected"
        })
    
    async with aiosqlite.connect(DATABASE_FILE) as db:
        # Check if email exists
        cursor = await db.execute("SELECT id FROM users WHERE email = ?", (email,))
        if await cursor.fetchone():
            return templates.TemplateResponse("signup.html", {
                "request": request,
                "role": role,
                "error": "Email already registered"
            })
        
        # Hash password
        password_hash = pwd_context.hash(password)
        
        await db.execute("""
            INSERT INTO users (email, password_hash, full_name, role, company_name, phone)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (email, password_hash, full_name, role, company_name, phone))
        await db.commit()
    
    return templates.TemplateResponse("signup.html", {
        "request": request,
        "role": role,
        "success": "Account created successfully! Please login."
    })

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, role: str = "graduate", error: Optional[str] = None):
    """Login page"""
    return templates.TemplateResponse("login.html", {
        "request": request,
        "role": role,
        "error": error
    })

@app.post("/login")
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...)
):
    """Handle login"""
    async with aiosqlite.connect(DATABASE_FILE) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM users WHERE email = ? AND role = ?", 
            (email, role)
        )
        user = await cursor.fetchone()
        
        if not user or not pwd_context.verify(password, user["password_hash"]):
            return templates.TemplateResponse("login.html", {
                "request": request,
                "role": role,
                "error": "Invalid email or password"
            })
        
        if not user["is_active"]:
            return templates.TemplateResponse("login.html", {
                "request": request,
                "role": role,
                "error": "Account is deactivated"
            })
        
        # Create session
        session_id = create_session(user["id"], user["role"])
        
        # Redirect based on role
        if role == "recruiter":
            redirect_url = "/recruiter/dashboard"
        elif role == "employer":
            redirect_url = "/employer/dashboard"
        else:
            redirect_url = "/graduate/dashboard"
        
        response = RedirectResponse(url=redirect_url, status_code=302)
        response.set_cookie(
            key="session_id", 
            value=session_id, 
            httponly=True, 
            max_age=86400,
            samesite="lax"
        )
        return response

@app.get("/logout")
async def logout():
    """Logout"""
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie(key="session_id")
    return response

# ============== GRADUATE ROUTES ==============

@app.get("/graduate/dashboard", response_class=HTMLResponse)
async def graduate_dashboard(request: Request, user: Dict = Depends(get_graduate)):
    """Graduate dashboard"""
    async with aiosqlite.connect(DATABASE_FILE) as db:
        db.row_factory = aiosqlite.Row
        
        # Stats
        cursor = await db.execute("SELECT COUNT(*) as count FROM resumes WHERE user_id = ?", (user["id"],))
        total_resumes = (await cursor.fetchone())["count"]
        
        cursor = await db.execute("""
            SELECT COUNT(*) as count FROM matches m
            JOIN resumes r ON m.resume_id = r.id
            WHERE r.user_id = ?
        """, (user["id"],))
        total_matches = (await cursor.fetchone())["count"]
        
        cursor = await db.execute("""
            SELECT COUNT(*) as count FROM matches m
            JOIN resumes r ON m.resume_id = r.id
            WHERE r.user_id = ? AND m.status = 'interview'
        """, (user["id"],))
        interviews = (await cursor.fetchone())["count"]
        
        cursor = await db.execute("""
            SELECT COUNT(*) as count FROM resumes WHERE user_id = ? AND status = 'pending'
        """, (user["id"],))
        pending = (await cursor.fetchone())["count"]
        
        # Matches with job details
        cursor = await db.execute("""
            SELECT m.*, j.title as job_title, j.department, j.location, 
                   u.company_name, m.created_at as matched_at
            FROM matches m
            JOIN job_descriptions j ON m.job_id = j.id
            JOIN users u ON j.employer_id = u.id
            JOIN resumes r ON m.resume_id = r.id
            WHERE r.user_id = ?
            ORDER BY m.match_score DESC
        """, (user["id"],))
        matches = await cursor.fetchall()
    
    return templates.TemplateResponse("graduate_dashboard.html", {
        "request": request,
        "user": user,
        "stats": {
            "total_resumes": total_resumes,
            "total_matches": total_matches,
            "interviews": interviews,
            "pending": pending
        },
        "matches": matches
    })

@app.get("/graduate/upload-resume", response_class=HTMLResponse)
async def graduate_upload_page(request: Request, user: Dict = Depends(get_graduate)):
    """Resume upload page for graduates"""
    return templates.TemplateResponse("resume_upload.html", {
        "request": request,
        "user": user
    })

@app.post("/graduate/upload-resume")
async def graduate_upload_resume(
    request: Request,
    background_tasks: BackgroundTasks,
    full_name: str = Form(...),
    email: EmailStr = Form(...),
    resume_file: UploadFile = File(...),
    user: Dict = Depends(get_graduate)
):
    """Upload resume for graduate"""
    if not resume_file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    contents = await resume_file.read()
    resume_text = extract_text_from_pdf(contents)
    
    if not resume_text or len(resume_text) < 100:
        raise HTTPException(status_code=400, detail="Could not extract sufficient text from PDF")
    
    resume_embedding = await create_embedding_api(resume_text)
    
    async with aiosqlite.connect(DATABASE_FILE) as db:
        await db.execute("""
            INSERT INTO resumes (user_id, full_name, email, resume_text, embedding, status)
            VALUES (?, ?, ?, ?, ?, 'pending')
        """, (user["id"], full_name, email, resume_text, resume_embedding))
        await db.commit()
        
        cursor = await db.execute("SELECT last_insert_rowid()")
        resume_id = (await cursor.fetchone())[0]
    
    # Process matching in background
    background_tasks.add_task(process_resume_matching, resume_id, user["id"])
    
    return {
        "id": resume_id,
        "full_name": full_name,
        "email": email,
        "status": "processing",
        "message": "Resume uploaded successfully. Matching in progress!"
    }

# ============== EMPLOYER ROUTES ==============

@app.get("/employer/dashboard", response_class=HTMLResponse)
async def employer_dashboard(request: Request, user: Dict = Depends(get_employer)):
    """Employer dashboard"""
    async with aiosqlite.connect(DATABASE_FILE) as db:
        db.row_factory = aiosqlite.Row
        
        # Stats
        cursor = await db.execute(
            "SELECT COUNT(*) as count FROM job_descriptions WHERE employer_id = ? AND is_active = 1", 
            (user["id"],)
        )
        total_jobs = (await cursor.fetchone())["count"]
        
        cursor = await db.execute("""
            SELECT COUNT(*) as count FROM matches WHERE employer_id = ?
        """, (user["id"],))
        total_applications = (await cursor.fetchone())["count"]
        
        cursor = await db.execute("""
            SELECT COUNT(*) as count FROM matches WHERE employer_id = ? AND status = 'matched'
        """, (user["id"],))
        matched_candidates = (await cursor.fetchone())["count"]
        
        cursor = await db.execute("""
            SELECT COUNT(*) as count FROM matches WHERE employer_id = ? AND status = 'reviewing'
        """, (user["id"],))
        pending_reviews = (await cursor.fetchone())["count"]
        
        # Jobs with match counts
        cursor = await db.execute("""
            SELECT j.*, COUNT(m.id) as match_count
            FROM job_descriptions j
            LEFT JOIN matches m ON j.id = m.job_id
            WHERE j.employer_id = ? AND j.is_active = 1
            GROUP BY j.id
            ORDER BY j.created_at DESC
        """, (user["id"],))
        jobs = await cursor.fetchall()
        
        # Recent matches
        cursor = await db.execute("""
            SELECT m.*, r.full_name, r.email, j.title as job_title
            FROM matches m
            JOIN resumes r ON m.resume_id = r.id
            JOIN job_descriptions j ON m.job_id = j.id
            WHERE m.employer_id = ?
            ORDER BY m.created_at DESC
            LIMIT 10
        """, (user["id"],))
        recent_matches = await cursor.fetchall()
    
    return templates.TemplateResponse("employer_dashboard.html", {
        "request": request,
        "user": user,
        "stats": {
            "total_jobs": total_jobs,
            "total_applications": total_applications,
            "matched_candidates": matched_candidates,
            "pending_reviews": pending_reviews
        },
        "jobs": jobs,
        "recent_matches": recent_matches
    })

@app.post("/employer/jobs")
async def employer_create_job(
    request: Request,
    background_tasks: BackgroundTasks,
    title: str = Form(...),
    description: str = Form(...),
    department: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    requirements: Optional[str] = Form(None),
    salary_range: Optional[str] = Form(None),
    user: Dict = Depends(get_employer)
):
    """Employer creates a job posting"""
    embedding = await create_embedding_api(description)
    
    async with aiosqlite.connect(DATABASE_FILE) as db:
        await db.execute("""
            INSERT INTO job_descriptions 
            (employer_id, title, description, department, location, requirements, salary_range, embedding, embedding_created)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (user["id"], title, description, department, location, requirements, salary_range, embedding))
        await db.commit()
        
        cursor = await db.execute("SELECT last_insert_rowid()")
        job_id = (await cursor.fetchone())[0]
    
    # Reprocess pending resumes in background
    background_tasks.add_task(reprocess_pending_resumes, job_id, user["id"])
    
    return {"id": job_id, "message": "Job posted successfully! Matching candidates..."}

@app.delete("/employer/jobs/{job_id}")
async def employer_delete_job(job_id: int, user: Dict = Depends(get_employer)):
    """Delete employer's job"""
    async with aiosqlite.connect(DATABASE_FILE) as db:
        cursor = await db.execute(
            "DELETE FROM job_descriptions WHERE id = ? AND employer_id = ?", 
            (job_id, user["id"])
        )
        await db.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Job not found")
    
    return {"message": "Job deleted successfully"}

@app.get("/employer/jobs/{job_id}/candidates", response_class=HTMLResponse)
async def employer_job_candidates(request: Request, job_id: int, user: Dict = Depends(get_employer)):
    """View candidates for a specific job"""
    async with aiosqlite.connect(DATABASE_FILE) as db:
        db.row_factory = aiosqlite.Row
        
        # Verify job belongs to employer
        cursor = await db.execute(
            "SELECT * FROM job_descriptions WHERE id = ? AND employer_id = ?", 
            (job_id, user["id"])
        )
        job = await cursor.fetchone()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Get candidates
        cursor = await db.execute("""
            SELECT m.*, r.full_name, r.email, r.resume_text
            FROM matches m
            JOIN resumes r ON m.resume_id = r.id
            WHERE m.job_id = ?
            ORDER BY m.match_score DESC
        """, (job_id,))
        candidates = await cursor.fetchall()
    
    # Simple HTML for candidate list
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Candidates - {job['title']}</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background: #f5f7fa; margin: 0; }}
            .navbar {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px 40px; }}
            .container {{ max-width: 1200px; margin: 0 auto; padding: 40px; }}
            .job-header {{ background: white; padding: 30px; border-radius: 15px; margin-bottom: 30px; }}
            .candidate-card {{ background: white; padding: 25px; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); }}
            .score {{ font-size: 2em; color: #667eea; font-weight: bold; }}
            .btn {{ padding: 10px 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 20px; cursor: pointer; text-decoration: none; display: inline-block; }}
            .btn-success {{ background: #28a745; }}
        </style>
    </head>
    <body>
        <div class="navbar">
            <h1>🎯 Candidates for {job['title']}</h1>
        </div>
        <div class="container">
            <div class="job-header">
                <h2>{job['title']}</h2>
                <p>{job['department'] or ''} | {job['location'] or ''}</p>
                <a href="/employer/dashboard" class="btn">← Back to Dashboard</a>
            </div>
    """
    
    for candidate in candidates:
        html += f"""
            <div class="candidate-card">
                <h3>{candidate['full_name']}</h3>
                <p>📧 {candidate['email']}</p>
                <div class="score">{candidate['match_score']*100:.1f}%</div>
                <p>Status: <strong>{candidate['status']}</strong></p>
                <div style="margin-top: 15px;">
                    <a href="/employer/candidates/{candidate['resume_id']}/cv" class="btn" target="_blank">View CV</a>
                    <a href="mailto:{candidate['email']}" class="btn btn-success">Email</a>
                </div>
            </div>
        """
    
    if not candidates:
        html += '<p style="text-align: center; color: #666; padding: 40px;">No candidates yet.</p>'
    
    html += """
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html)

@app.get("/employer/candidates/{resume_id}/cv")
async def employer_view_cv(resume_id: int, user: Dict = Depends(get_employer)):
    """View candidate CV text"""
    async with aiosqlite.connect(DATABASE_FILE) as db:
        db.row_factory = aiosqlite.Row
        
        # Verify employer has a match with this resume
        cursor = await db.execute("""
            SELECT r.* FROM resumes r
            JOIN matches m ON r.id = m.resume_id
            WHERE r.id = ? AND m.employer_id = ?
        """, (resume_id, user["id"]))
        resume = await cursor.fetchone()
        
        if not resume:
            raise HTTPException(status_code=404, detail="CV not found")
    
    return {
        "candidate_name": resume["full_name"],
        "email": resume["email"],
        "resume_text": resume["resume_text"],
        "uploaded_at": resume["created_at"]
    }

@app.post("/employer/candidates/{resume_id}/status")
async def employer_update_status(
    resume_id: int,
    status: str = Form(...),
    job_id: int = Form(...),
    user: Dict = Depends(get_employer)
):
    """Update candidate status"""
    if status not in ["matched", "reviewing", "interview", "hired", "rejected"]:
        raise HTTPException(status_code=400, detail="Invalid status")
    
    async with aiosqlite.connect(DATABASE_FILE) as db:
        cursor = await db.execute("""
            UPDATE matches SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE resume_id = ? AND job_id = ? AND employer_id = ?
        """, (status, resume_id, job_id, user["id"]))
        await db.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Match not found")
        
        # Get candidate email for notification
        cursor = await db.execute("""
            SELECT r.email, r.full_name, j.title 
            FROM resumes r
            JOIN matches m ON r.id = m.resume_id
            JOIN job_descriptions j ON m.job_id = j.id
            WHERE r.id = ? AND m.job_id = ?
        """, (resume_id, job_id))
        result = await cursor.fetchone()
        
        if result:
            subject = f"Application Update: {result['title']}"
            body = f"""
Dear {result['full_name']},

Your application status for "{result['title']}" has been updated to: {status.upper()}

Best regards,
Hiring Team
"""
            await send_email(result["email"], subject, body)
    
    return {"message": "Status updated successfully"}

@app.get("/employer/notifications")
async def employer_notifications(user: Dict = Depends(get_employer)):
    """Get unread notifications for employer"""
    async with aiosqlite.connect(DATABASE_FILE) as db:
        db.row_factory = aiosqlite.Row
        
        cursor = await db.execute("""
            SELECT COUNT(*) as count FROM notifications 
            WHERE user_id = ? AND is_read = 0
        """, (user["id"],))
        count = (await cursor.fetchone())["count"]
        
        # Mark as read
        await db.execute("""
            UPDATE notifications SET is_read = 1 WHERE user_id = ?
        """, (user["id"],))
        await db.commit()
    
    return {"new_matches": count}

# ============== RECRUITER ROUTES ==============

@app.get("/recruiter/dashboard", response_class=HTMLResponse)
async def recruiter_dashboard(request: Request, user: Dict = Depends(get_recruiter)):
    """Recruiter admin dashboard"""
    async with aiosqlite.connect(DATABASE_FILE) as db:
        db.row_factory = aiosqlite.Row
        
        # Stats
        cursor = await db.execute("SELECT COUNT(*) as count FROM users WHERE role = 'employer' AND is_active = 1")
        total_employers = (await cursor.fetchone())["count"]
        
        cursor = await db.execute("SELECT COUNT(*) as count FROM users WHERE role = 'graduate' AND is_active = 1")
        total_graduates = (await cursor.fetchone())["count"]
        
        cursor = await db.execute("SELECT COUNT(*) as count FROM job_descriptions WHERE is_active = 1")
        total_jobs = (await cursor.fetchone())["count"]
        
        cursor = await db.execute("SELECT COUNT(*) as count FROM matches")
        total_matches = (await cursor.fetchone())["count"]
        
        cursor = await db.execute("""
            SELECT COUNT(*) as count FROM matches WHERE status = 'reviewing'
        """)
        pending_reviews = (await cursor.fetchone())["count"]
        
        # Employers with job counts
        cursor = await db.execute("""
            SELECT u.*, COUNT(j.id) as job_count
            FROM users u
            LEFT JOIN job_descriptions j ON u.id = j.employer_id AND j.is_active = 1
            WHERE u.role = 'employer' AND u.is_active = 1
            GROUP BY u.id
            ORDER BY u.created_at DESC
        """)
        employers = await cursor.fetchall()
        
        # Graduates with resume/match counts
        cursor = await db.execute("""
            SELECT u.*, 
                   COUNT(DISTINCT r.id) as resume_count,
                   COUNT(DISTINCT m.id) as match_count
            FROM users u
            LEFT JOIN resumes r ON u.id = r.user_id
            LEFT JOIN matches m ON r.id = m.resume_id
            WHERE u.role = 'graduate' AND u.is_active = 1
            GROUP BY u.id
            ORDER BY u.created_at DESC
        """)
        graduates = await cursor.fetchall()
        
        # All jobs
        cursor = await db.execute("""
            SELECT j.*, u.full_name as employer_name,
                   COUNT(m.id) as match_count
            FROM job_descriptions j
            JOIN users u ON j.employer_id = u.id
            LEFT JOIN matches m ON j.id = m.job_id
            WHERE j.is_active = 1
            GROUP BY j.id
            ORDER BY j.created_at DESC
        """)
        all_jobs = await cursor.fetchall()
        
        # All matches
        cursor = await db.execute("""
            SELECT m.*, r.full_name as candidate_name, j.title as job_title,
                   u.full_name as employer_name, m.created_at as matched_at
            FROM matches m
            JOIN resumes r ON m.resume_id = r.id
            JOIN job_descriptions j ON m.job_id = j.id
            JOIN users u ON j.employer_id = u.id
            ORDER BY m.created_at DESC
            LIMIT 100
        """)
        all_matches = await cursor.fetchall()
    
    return templates.TemplateResponse("recruiter_dashboard.html", {
        "request": request,
        "user": user,
        "stats": {
            "total_employers": total_employers,
            "total_graduates": total_graduates,
            "total_jobs": total_jobs,
            "total_matches": total_matches,
            "pending_reviews": pending_reviews
        },
        "employers": employers,
        "graduates": graduates,
        "all_jobs": all_jobs,
        "all_matches": all_matches
    })

@app.delete("/recruiter/users/{user_id}")
async def recruiter_delete_user(user_id: int, role: str, user: Dict = Depends(get_recruiter)):
    """Recruiter can delete employer or graduate accounts"""
    if role not in ["employer", "graduate"]:
        raise HTTPException(status_code=400, detail="Can only delete employer or graduate accounts")
    
    async with aiosqlite.connect(DATABASE_FILE) as db:
        # Soft delete
        await db.execute("""
            UPDATE users SET is_active = 0, updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND role = ?
        """, (user_id, role))
        await db.commit()
    
    return {"message": f"{role} account deactivated"}

# ============== HEALTH CHECK ==============

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "3.0.0",
        "roles": ["recruiter", "employer", "graduate"],
        "embedding_source": "HuggingFace Inference API",
        "database": "connected"
    }

# ============== RUN ==============

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
FastAPI ATS Semantic Matching System with Embedded Frontend
Single-file implementation with SQLite database and Jinja2 templates
Optimized for Render.com with all-MiniLM-L6-v2 model
"""

import os
import re
import sqlite3
import asyncio
import hashlib
import secrets
import gc
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, status, BackgroundTasks, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
import aiohttp
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiosmtplib

# Document processing
import PyPDF2
import io

# ML
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# For creating temporary templates directory
import tempfile
import shutil

# Configuration
DATABASE_FILE = "ats_database.db"
MODEL_NAME = 'all-MiniLM-L6-v2'  # Changed to smaller model (~20MB)
MATCH_THRESHOLD = 0.80  # 80% match threshold
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "noreply@ats-system.com")

# Admin credentials (in production, use hashed passwords)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# Security
security = HTTPBasic()

# Initialize model globally - lazy loading
model = None
model_lock = asyncio.Lock()

# Create temporary directory for templates
TEMP_DIR = tempfile.mkdtemp()
TEMPLATES_DIR = os.path.join(TEMP_DIR, "templates")
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Create cache directory for embeddings
CACHE_DIR = "/tmp/ats_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Pydantic Models
class JobDescriptionCreate(BaseModel):
    title: str
    description: str
    department: Optional[str] = None
    location: Optional[str] = None

class JobDescriptionResponse(BaseModel):
    id: int
    title: str
    description: str
    department: Optional[str]
    location: Optional[str]
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

# HTML Templates
INDEX_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ATS Semantic Matching System</title>
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
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
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
        }
        .btn:hover {
            transform: scale(1.05);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
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
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Smart ATS</h1>
            <p>AI-Powered Resume & Job Matching System</p>
        </div>
        
        <div class="cards">
            <div class="card">
                <div class="card-icon">👨‍💼</div>
                <h2>I'm a Candidate</h2>
                <p>Upload your resume and let our AI match you with the perfect job opportunities. Get instant notifications when matches are found.</p>
                <a href="/submit-resume" class="btn">Upload Resume</a>
            </div>
            
            <div class="card">
                <div class="card-icon">🏢</div>
                <h2>I'm an Admin</h2>
                <p>Post job descriptions, manage applications, and review AI-powered candidate matches with semantic analysis.</p>
                <a href="/admin/login" class="btn">Admin Portal</a>
            </div>
        </div>
        
        <div class="features">
            <h2>How It Works</h2>
            <div class="feature-grid">
                <div class="feature-item">
                    <h3>📝 Upload</h3>
                    <p>Candidates upload PDF resumes</p>
                </div>
                <div class="feature-item">
                    <h3>🤖 AI Analysis</h3>
                    <p>Semantic embedding generation</p>
                </div>
                <div class="feature-item">
                    <h3>🎯 Matching</h3>
                    <p>80% threshold similarity match</p>
                </div>
                <div class="feature-item">
                    <h3>📧 Notification</h3>
                    <p>Automatic email alerts</p>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Powered by Sentence Transformers & FastAPI</p>
        </div>
    </div>
</body>
</html>
"""

ADMIN_LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Login - ATS System</title>
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
        .login-box p { color: #666; margin-bottom: 30px; }
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
        <h1>🔐 Admin Login</h1>
        <p>Enter your credentials to access the dashboard</p>
        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
        <form action="/admin/login" method="post">
            <div class="form-group">
                <label>Username</label>
                <input type="text" name="username" required placeholder="admin">
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" name="password" required placeholder="••••••••">
            </div>
            <button type="submit" class="btn">Sign In</button>
        </form>
        <a href="/" class="back-link">← Back to Home</a>
    </div>
</body>
</html>
"""

ADMIN_DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Dashboard - ATS System</title>
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
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .stat-card {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .stat-card h3 {
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .stat-card .number {
            font-size: 2.5em;
            color: #667eea;
            font-weight: bold;
        }
        .section {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
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
        .btn-danger {
            background: #e74c3c;
        }
        .btn-sm {
            padding: 5px 15px;
            font-size: 0.8em;
        }
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
        .badge-notified { background: #d1ecf1; color: #0c5460; }
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
            max-width: 600px;
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
            min-height: 200px;
            resize: vertical;
        }
        .close-btn {
            float: right;
            font-size: 1.5em;
            cursor: pointer;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="navbar">
        <h1>🎯 Smart ATS Admin</h1>
        <div class="nav-links">
            <a href="/admin/dashboard">Dashboard</a>
            <a href="#" onclick="openModal()">+ New Job</a>
            <a href="/">Logout</a>
        </div>
    </div>
    
    <div class="container">
        <div class="stats">
            <div class="stat-card">
                <h3>Total Jobs</h3>
                <div class="number">{{ stats.total_jobs }}</div>
            </div>
            <div class="stat-card">
                <h3>Total Resumes</h3>
                <div class="number">{{ stats.total_resumes }}</div>
            </div>
            <div class="stat-card">
                <h3>Matched</h3>
                <div class="number">{{ stats.matched }}</div>
            </div>
            <div class="stat-card">
                <h3>Pending</h3>
                <div class="number">{{ stats.pending }}</div>
            </div>
        </div>
        
        <div class="section">
            <h2>
                Job Descriptions
                <button class="btn" onclick="openModal()">+ Add New Job</button>
            </h2>
            <table>
                <thead>
                    <tr>
                        <th>Title</th>
                        <th>Department</th>
                        <th>Location</th>
                        <th>Posted</th>
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
                        <td>
                            <button class="btn btn-danger btn-sm" onclick="deleteJob({{ job.id }})">Delete</button>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>Recent Applications</h2>
            <table>
                <thead>
                    <tr>
                        <th>Candidate</th>
                        <th>Email</th>
                        <th>Status</th>
                        <th>Match Score</th>
                        <th>Matched Job</th>
                    </tr>
                </thead>
                <tbody>
                    {% for resume in resumes %}
                    <tr>
                        <td><strong>{{ resume.full_name }}</strong></td>
                        <td>{{ resume.email }}</td>
                        <td>
                            <span class="badge badge-{{ resume.status }}">
                                {{ resume.status }}
                            </span>
                        </td>
                        <td>
                            {% if resume.match_score %}
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <span>{{ "%.1f"|format(resume.match_score * 100) }}%</span>
                                <div class="progress-bar" style="width: 100px;">
                                    <div class="progress-fill" style="width: {{ resume.match_score * 100 }}%"></div>
                                </div>
                            </div>
                            {% else %}
                            -
                            {% endif %}
                        </td>
                        <td>{{ resume.job_title or '-' }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
    
    <div id="jobModal" class="modal">
        <div class="modal-content">
            <span class="close-btn" onclick="closeModal()">&times;</span>
            <h2>Create New Job Description</h2>
            <form id="jobForm" onsubmit="submitJob(event)">
                <div class="form-group">
                    <label>Job Title</label>
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
                    <label>Job Description</label>
                    <textarea name="description" required placeholder="Enter detailed job description..."></textarea>
                </div>
                <button type="submit" class="btn" style="width: 100%;">Create Job & Generate Embedding</button>
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
        
        async function submitJob(e) {
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData);
            
            try {
                const response = await fetch('/admin/jobs', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': 'Basic ' + btoa('admin:admin123')
                    },
                    body: JSON.stringify(data)
                });
                
                if (response.ok) {
                    alert('Job created successfully! Embedding generated.');
                    location.reload();
                } else {
                    alert('Error creating job');
                }
            } catch (err) {
                alert('Error: ' + err.message);
            }
        }
        
        async function deleteJob(id) {
            if (!confirm('Are you sure you want to delete this job?')) return;
            
            try {
                const response = await fetch(`/admin/jobs/${id}`, {
                    method: 'DELETE',
                    headers: {
                        'Authorization': 'Basic ' + btoa('admin:admin123')
                    }
                });
                
                if (response.ok) {
                    location.reload();
                } else {
                    alert('Error deleting job');
                }
            } catch (err) {
                alert('Error: ' + err.message);
            }
        }
        
        // Close modal on outside click
        window.onclick = function(e) {
            if (e.target.classList.contains('modal')) {
                closeModal();
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
    <title>Upload Resume - ATS System</title>
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
            Your resume has been uploaded. Check your email for matching results!
        </div>
        
        <form id="resumeForm" onsubmit="submitResume(event)">
            <div class="form-group">
                <label>Full Name</label>
                <input type="text" name="full_name" required placeholder="John Doe">
            </div>
            
            <div class="form-group">
                <label>Email Address</label>
                <input type="email" name="email" required placeholder="john@example.com">
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
        
        <a href="/" class="back-link">← Back to Home</a>
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
                const response = await fetch('/upload-resume', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    successMsg.style.display = 'block';
                    document.getElementById('resumeForm').reset();
                    document.getElementById('fileLabel').classList.remove('has-file');
                    document.getElementById('fileLabel').innerHTML = '<div>📁 Click to upload PDF</div><small>Maximum file size: 10MB</small>';
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

# Write templates to files
def setup_templates():
    """Write HTML templates to temporary files"""
    templates = {
        "index.html": INDEX_TEMPLATE,
        "admin_login.html": ADMIN_LOGIN_TEMPLATE,
        "admin_dashboard.html": ADMIN_DASHBOARD_TEMPLATE,
        "resume_upload.html": RESUME_UPLOAD_TEMPLATE
    }
    
    for filename, content in templates.items():
        filepath = os.path.join(TEMPLATES_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
    
    return TEMPLATES_DIR

# Database setup
def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # Job descriptions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS job_descriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            department TEXT,
            location TEXT,
            embedding BLOB,
            embedding_created BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Resumes table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL,
            resume_text TEXT NOT NULL,
            embedding BLOB,
            status TEXT DEFAULT 'pending',
            matched_job_id INTEGER,
            match_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (matched_job_id) REFERENCES job_descriptions(id)
        )
    """)
    
    # Create index for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_resumes_status ON resumes(status)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_resumes_email ON resumes(email)
    """)
    
    conn.commit()
    conn.close()

def get_db():
    """Database dependency"""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# Authentication
def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify admin credentials"""
    if credentials.username != ADMIN_USERNAME or credentials.password != ADMIN_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# PDF Processing
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

# Email Service
async def send_email(to_email: str, subject: str, body: str, html_body: Optional[str] = None):
    """Send email notification"""
    if not SMTP_USER or not SMTP_PASS:
        # Log email instead of sending if SMTP not configured
        print(f"[EMAIL LOG] To: {to_email}\nSubject: {subject}\nBody: {body}\n")
        return True
    
    try:
        message = MIMEMultipart("alternative")
        message["From"] = FROM_EMAIL
        message["To"] = to_email
        message["Subject"] = subject
        
        # Add plain text part
        message.attach(MIMEText(body, "plain"))
        
        # Add HTML part if provided
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

# Model Management with Lazy Loading and Memory Optimization
async def get_model():
    """Get or load the model with thread safety"""
    global model
    async with model_lock:
        if model is None:
            print(f"Loading model: {MODEL_NAME}")
            # Use CPU and disable parallelism to save memory
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            model = SentenceTransformer(MODEL_NAME, device='cpu')
            print("Model loaded successfully")
    return model

def unload_model():
    """Unload model to free memory"""
    global model
    if model is not None:
        del model
        model = None
        gc.collect()
        print("Model unloaded to free memory")

def create_embedding(text: str) -> bytes:
    """Create embedding for text using the model with memory management"""
    # Truncate text to avoid memory issues (model max is 256 tokens for MiniLM)
    max_chars = 10000
    if len(text) > max_chars:
        text = text[:max_chars]
    
    # Get model (lazy load)
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        current_model = loop.run_until_complete(get_model())
    except:
        # If no event loop, load synchronously
        current_model = SentenceTransformer(MODEL_NAME, device='cpu')
    
    # Create embedding
    embedding = current_model.encode(text, show_progress_bar=False, convert_to_numpy=True)
    
    # Clear from memory after use in background tasks
    if os.getenv('RENDER', ''):
        unload_model()
    
    return pickle.dumps(embedding)

def get_embedding_from_bytes(embedding_blob: bytes) -> np.ndarray:
    """Deserialize embedding from bytes"""
    return pickle.loads(embedding_blob)

def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings"""
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return float(similarity)

# Matching Logic
async def find_best_job_match(resume_embedding: np.ndarray, db: sqlite3.Connection) -> Optional[Dict]:
    """Find the best matching job for a resume"""
    cursor = db.cursor()
    cursor.execute(
        "SELECT id, title, description, embedding FROM job_descriptions WHERE embedding_created = 1"
    )
    jobs = cursor.fetchall()
    
    if not jobs:
        return None
    
    best_match = None
    best_score = 0.0
    
    for job in jobs:
        if job["embedding"]:
            job_embedding = get_embedding_from_bytes(job["embedding"])
            score = calculate_similarity(resume_embedding, job_embedding)
            
            if score > best_score:
                best_score = score
                best_match = {
                    "job_id": job["id"],
                    "title": job["title"],
                    "description": job["description"],
                    "score": score
                }
    
    return best_match if best_match and best_score >= MATCH_THRESHOLD else None

async def process_resume_matching(resume_id: int, background_tasks: BackgroundTasks):
    """Process resume matching and send notification"""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Get resume details
        cursor.execute("SELECT * FROM resumes WHERE id = ?", (resume_id,))
        resume = cursor.fetchone()
        
        if not resume or resume["status"] != "pending":
            return
        
        resume_embedding = get_embedding_from_bytes(resume["embedding"])
        
        # Find best match
        match = await find_best_job_match(resume_embedding, conn)
        
        if match:
            # Update resume with match
            cursor.execute("""
                UPDATE resumes 
                SET status = 'matched', 
                    matched_job_id = ?, 
                    match_score = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (match["job_id"], match["score"], resume_id))
            
            # Send match notification
            subject = f"Great News! Job Match Found - {match['title']}"
            body = f"""
Dear {resume["full_name"]},

We found a job that matches your profile!

Position: {match['title']}
Match Score: {match['score']*100:.1f}%

Job Description:
{match['description'][:500]}...

Our team will contact you shortly with next steps.

Best regards,
ATS Recruitment Team
"""
            html_body = f"""
<html>
<body>
    <h2>Great News! Job Match Found</h2>
    <p>Dear {resume["full_name"]},</p>
    <p>We found a job that matches your profile!</p>
    <h3>Position: {match['title']}</h3>
    <p><strong>Match Score: {match['score']*100:.1f}%</strong></p>
    <p>{match['description'][:500]}...</p>
    <p>Our team will contact you shortly with next steps.</p>
    <br>
    <p>Best regards,<br>ATS Recruitment Team</p>
</body>
</html>
"""
            await send_email(resume["email"], subject, body, html_body)
            
        else:
            # No match found
            cursor.execute("""
                UPDATE resumes 
                SET status = 'notified_no_match',
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (resume_id,))
            
            # Send no-match notification
            subject = "Application Received - Updates Coming Soon"
            body = f"""
Dear {resume["full_name"]},

Thank you for submitting your resume to our system.

We've reviewed your profile and while we don't currently have an opening that matches your qualifications at the 80% threshold, we are continuously adding new positions.

You will receive an automatic notification as soon as a matching position becomes available.

Best regards,
ATS Recruitment Team
"""
            await send_email(resume["email"], subject, body)
        
        conn.commit()
        
    except Exception as e:
        print(f"Error processing resume {resume_id}: {e}")
    finally:
        conn.close()

async def reprocess_pending_resumes(new_job_id: int, background_tasks: BackgroundTasks):
    """Reprocess pending resumes when new job is added"""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Get new job embedding
        cursor.execute("SELECT embedding FROM job_descriptions WHERE id = ?", (new_job_id,))
        job = cursor.fetchone()
        
        if not job or not job["embedding"]:
            return
        
        job_embedding = get_embedding_from_bytes(job["embedding"])
        
        # Get all pending resumes
        cursor.execute("SELECT * FROM resumes WHERE status = 'pending'")
        pending_resumes = cursor.fetchall()
        
        for resume in pending_resumes:
            if resume["embedding"]:
                resume_embedding = get_embedding_from_bytes(resume["embedding"])
                score = calculate_similarity(job_embedding, resume_embedding)
                
                if score >= MATCH_THRESHOLD:
                    # Update and notify
                    cursor.execute("""
                        UPDATE resumes 
                        SET status = 'matched', 
                            matched_job_id = ?, 
                            match_score = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (new_job_id, score, resume["id"]))
                    
                    # Get job details for email
                    cursor.execute("SELECT title, description FROM job_descriptions WHERE id = ?", (new_job_id,))
                    job_details = cursor.fetchone()
                    
                    # Send notification
                    subject = f"New Job Match Found! - {job_details['title']}"
                    body = f"""
Dear {resume["full_name"]},

Great news! A new position matching your profile has been posted.

Position: {job_details['title']}
Match Score: {score*100:.1f}%

{job_details['description'][:500]}...

Our team will contact you shortly.

Best regards,
ATS Recruitment Team
"""
                    await send_email(resume["email"], subject, body)
        
        conn.commit()
        
    except Exception as e:
        print(f"Error reprocessing resumes: {e}")
    finally:
        conn.close()

# FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    global model
    
    # Setup templates
    setup_templates()
    
    # Startup
    print("Initializing database...")
    init_database()
    
    # Pre-download model in build step, don't load yet
    print(f"Model {MODEL_NAME} will be loaded on first use...")
    
    yield
    
    # Cleanup
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    unload_model()
    print("Shutting down...")

app = FastAPI(
    title="ATS Semantic Matching API",
    description="AI-powered Applicant Tracking System with semantic resume-job matching",
    version="1.0.0",
    lifespan=lifespan
)

# Setup Jinja2 templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frontend Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/submit-resume", response_class=HTMLResponse)
async def submit_resume_page(request: Request):
    """Resume upload page"""
    return templates.TemplateResponse("resume_upload.html", {"request": request})

@app.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page(request: Request, error: Optional[str] = None):
    """Admin login page"""
    return templates.TemplateResponse("admin_login.html", {"request": request, "error": error})

@app.post("/admin/login")
async def admin_login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Process admin login"""
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        # Redirect to dashboard with basic auth header
        response = RedirectResponse(url="/admin/dashboard", status_code=302)
        return response
    else:
        return templates.TemplateResponse(
            "admin_login.html", 
            {"request": request, "error": "Invalid username or password"}
        )

@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request, admin: str = Depends(verify_admin), db: sqlite3.Connection = Depends(get_db)):
    """Admin dashboard"""
    cursor = db.cursor()
    
    # Get stats
    cursor.execute("SELECT COUNT(*) as count FROM job_descriptions")
    total_jobs = cursor.fetchone()["count"]
    
    cursor.execute("SELECT COUNT(*) as count FROM resumes")
    total_resumes = cursor.fetchone()["count"]
    
    cursor.execute("SELECT COUNT(*) as count FROM resumes WHERE status = 'matched'")
    matched = cursor.fetchone()["count"]
    
    cursor.execute("SELECT COUNT(*) as count FROM resumes WHERE status = 'pending'")
    pending = cursor.fetchone()["count"]
    
    # Get jobs
    cursor.execute("""
        SELECT id, title, description, department, location, created_at 
        FROM job_descriptions 
        ORDER BY created_at DESC
    """)
    jobs = cursor.fetchall()
    
    # Get resumes with job titles
    cursor.execute("""
        SELECT r.*, j.title as job_title 
        FROM resumes r
        LEFT JOIN job_descriptions j ON r.matched_job_id = j.id
        ORDER BY r.created_at DESC
        LIMIT 50
    """)
    resumes = cursor.fetchall()
    
    return templates.TemplateResponse("admin_dashboard.html", {
        "request": request,
        "stats": {
            "total_jobs": total_jobs,
            "total_resumes": total_resumes,
            "matched": matched,
            "pending": pending
        },
        "jobs": jobs,
        "resumes": resumes
    })

# API Endpoints
@app.post("/admin/jobs", response_model=JobDescriptionResponse)
async def create_job_description(
    job: JobDescriptionCreate,
    background_tasks: BackgroundTasks,
    admin: str = Depends(verify_admin),
    db: sqlite3.Connection = Depends(get_db)
):
    """
    Admin endpoint to upload job description.
    Automatically creates embedding and reprocesses pending resumes.
    """
    cursor = db.cursor()
    
    # Create embedding immediately
    embedding = create_embedding(job.description)
    
    # Insert job
    cursor.execute("""
        INSERT INTO job_descriptions (title, description, department, location, embedding, embedding_created)
        VALUES (?, ?, ?, ?, ?, 1)
    """, (job.title, job.description, job.department, job.location, embedding))
    
    db.commit()
    job_id = cursor.lastrowid
    
    # Reprocess pending resumes in background
    background_tasks.add_task(reprocess_pending_resumes, job_id, background_tasks)
    
    return {
        "id": job_id,
        "title": job.title,
        "description": job.description,
        "department": job.department,
        "location": job.location,
        "embedding_created": True,
        "created_at": datetime.now().isoformat()
    }

@app.get("/admin/jobs", response_model=List[JobDescriptionResponse])
async def list_jobs(
    admin: str = Depends(verify_admin),
    db: sqlite3.Connection = Depends(get_db)
):
    """List all job descriptions"""
    cursor = db.cursor()
    cursor.execute("""
        SELECT id, title, description, department, location, embedding_created, created_at 
        FROM job_descriptions 
        ORDER BY created_at DESC
    """)
    jobs = cursor.fetchall()
    
    return [
        {
            "id": job["id"],
            "title": job["title"],
            "description": job["description"],
            "department": job["department"],
            "location": job["location"],
            "embedding_created": bool(job["embedding_created"]),
            "created_at": job["created_at"]
        }
        for job in jobs
    ]

@app.delete("/admin/jobs/{job_id}")
async def delete_job(
    job_id: int,
    admin: str = Depends(verify_admin),
    db: sqlite3.Connection = Depends(get_db)
):
    """Delete a job description"""
    cursor = db.cursor()
    cursor.execute("DELETE FROM job_descriptions WHERE id = ?", (job_id,))
    
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Job not found")
    
    db.commit()
    return {"message": "Job deleted successfully"}

@app.get("/admin/resumes")
async def list_resumes(
    admin: str = Depends(verify_admin),
    db: sqlite3.Connection = Depends(get_db)
):
    """List all resumes with their match status"""
    cursor = db.cursor()
    cursor.execute("""
        SELECT r.*, j.title as job_title 
        FROM resumes r
        LEFT JOIN job_descriptions j ON r.matched_job_id = j.id
        ORDER BY r.created_at DESC
    """)
    resumes = cursor.fetchall()
    
    return [
        {
            "id": resume["id"],
            "full_name": resume["full_name"],
            "email": resume["email"],
            "status": resume["status"],
            "match_score": resume["match_score"],
            "matched_job": resume["job_title"],
            "created_at": resume["created_at"]
        }
        for resume in resumes
    ]

# Public Endpoints
@app.post("/upload-resume", response_model=ResumeUploadResponse)
async def upload_resume(
    background_tasks: BackgroundTasks,
    full_name: str = Form(...),
    email: EmailStr = Form(...),
    resume_file: UploadFile = File(...),
    db: sqlite3.Connection = Depends(get_db)
):
    """
    Upload resume in PDF format with full name and email.
    Automatically matches against existing jobs and sends notification.
    """
    # Validate file type
    if not resume_file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    # Read and extract text from PDF
    contents = await resume_file.read()
    resume_text = extract_text_from_pdf(contents)
    
    if not resume_text or len(resume_text) < 100:
        raise HTTPException(status_code=400, detail="Could not extract sufficient text from PDF")
    
    # Create embedding
    resume_embedding = create_embedding(resume_text)
    
    # Save to database
    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO resumes (full_name, email, resume_text, embedding, status)
        VALUES (?, ?, ?, ?, 'pending')
    """, (full_name, email, resume_text, resume_embedding))
    
    db.commit()
    resume_id = cursor.lastrowid
    
    # Process matching in background
    background_tasks.add_task(process_resume_matching, resume_id, background_tasks)
    
    return {
        "id": resume_id,
        "full_name": full_name,
        "email": email,
        "status": "processing",
        "message": "Resume uploaded successfully. You will receive an email notification shortly."
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "database": "connected"
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

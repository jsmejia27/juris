import os
import sys
import json
import time
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from server import app

client = TestClient(app)

def test_feedback_page_html():
    response = client.get("/feedback")
    assert response.status_code == 200
    assert "Submit Platform Feedback" in response.text
    assert "feedbackEmail" in response.text
    assert "feedbackPhone" in response.text
    assert "captchaAnswerInput" in response.text

def test_feedback_captcha_generation():
    response = client.get("/api/feedback/captcha")
    assert response.status_code == 200
    data = response.json()
    assert "question" in data
    assert "token" in data
    assert "?" in data["question"]

def test_feedback_submission_success():
    # 1. Fetch challenge
    captcha_res = client.get("/api/feedback/captcha")
    assert captcha_res.status_code == 200
    c_data = captcha_res.json()
    q_str = c_data["question"]
    token = c_data["token"]

    # Calculate answer from "What is A + B?" or "What is A - B?"
    parts = q_str.replace("What is", "").replace("?", "").strip().split()
    a = int(parts[0])
    op = parts[1]
    b = int(parts[2])
    ans = a + b if op == "+" else a - b

    # 2. Submit valid feedback with mandatory email and optional phone
    res = client.post("/api/feedback", json={
        "email": "counsel@example.ph",
        "phone": "+63 917 123 4567",
        "category": "Legal Accuracy / Citation Issue",
        "subject": "Citation verification report for RA 9262",
        "message": "Observed accurate citing of Sec 5(i) requisites in recent jurisprudence analysis.",
        "captcha_token": token,
        "captcha_answer": str(ans)
    })
    assert res.status_code == 200
    res_data = res.json()
    assert res_data["status"] == "success"

def test_feedback_submission_invalid_captcha():
    captcha_res = client.get("/api/feedback/captcha")
    c_data = captcha_res.json()
    token = c_data["token"]

    # Submit wrong answer
    res = client.post("/api/feedback", json={
        "email": "counsel@example.ph",
        "phone": None,
        "category": "General Feedback",
        "subject": "Testing captcha failure",
        "message": "Testing that invalid captcha is rejected.",
        "captcha_token": token,
        "captcha_answer": "999999"
    })
    assert res.status_code == 400
    assert "Incorrect CAPTCHA" in res.json()["error"]

def test_feedback_submission_missing_or_invalid_email():
    captcha_res = client.get("/api/feedback/captcha")
    c_data = captcha_res.json()
    token = c_data["token"]

    # Invalid email
    res = client.post("/api/feedback", json={
        "email": "notanemail",
        "phone": None,
        "category": "General Feedback",
        "subject": "Testing bad email",
        "message": "Testing that bad email is rejected.",
        "captcha_token": token,
        "captcha_answer": "42"
    })
    assert res.status_code == 400

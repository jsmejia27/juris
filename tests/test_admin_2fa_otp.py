import os
import sys
import time
import json
import base64
import struct
import hmac
import hashlib
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from server import app, get_or_create_otp_secret, generate_totp, verify_totp, ADMIN_USERNAME, ADMIN_PASSWORD

def test_otp_secret_and_generation():
    secret = get_or_create_otp_secret()
    assert secret is not None
    assert len(secret) >= 16
    otp = generate_totp(secret)
    assert len(otp) == 6
    assert otp.isdigit()
    assert verify_totp(otp, secret) is True
    assert verify_totp("000000", secret) is False

def test_otp_setup_endpoint():
    client = TestClient(app)
    response = client.get("/api/manage/otp-setup")
    assert response.status_code == 200
    data = response.json()
    assert "secret" in data
    assert "otpauth_uri" in data
    assert data["username"] == ADMIN_USERNAME

def test_admin_2fa_login_failure():
    client = TestClient(app)
    # Invalid password
    res1 = client.post("/api/manage/login", json={"username": ADMIN_USERNAME, "password": "WrongPassword", "otp": "123456"})
    assert res1.status_code == 401

    # Invalid OTP
    res2 = client.post("/api/manage/login", json={"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD, "otp": "000000"})
    assert res2.status_code == 401

def test_admin_2fa_login_success_and_session():
    client = TestClient(app)
    secret = get_or_create_otp_secret()
    current_otp = generate_totp(secret)

    # Good login
    res = client.post("/api/manage/login", json={
        "username": ADMIN_USERNAME,
        "password": ADMIN_PASSWORD,
        "otp": current_otp
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "authenticated"
    assert "token" in data

    token = data["token"]

    # Access protected stats with session cookie
    stats_res = client.get("/api/manage/stats")
    assert stats_res.status_code == 200
    stats_data = stats_res.json()
    assert "total_vector_points" in stats_data
    assert "datasets" in stats_data

    # Access protected stats with Bearer token on fresh client
    fresh_client = TestClient(app)
    stats_res_bearer = fresh_client.get("/api/manage/stats", headers={"Authorization": f"Bearer {token}"})
    assert stats_res_bearer.status_code == 200

def test_admin_unauthorized_access():
    client = TestClient(app)
    res = client.get("/api/manage/stats")
    assert res.status_code == 401

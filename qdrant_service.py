# qdrant_service.py
import os
import time
import logging
import subprocess
import requests

logger = logging.getLogger(__name__)

DEFAULT_QDRANT_HOST = "127.0.0.1"
DEFAULT_QDRANT_HTTP_PORT = 6333
DEFAULT_QDRANT_GRPC_PORT = 6334
DEFAULT_QDRANT_URL = f"http://{DEFAULT_QDRANT_HOST}:{DEFAULT_QDRANT_HTTP_PORT}"
DEFAULT_STORAGE_DIR = os.path.abspath("./qdrant_server_data")
QDRANT_BIN_PATH = os.path.abspath("./bin/qdrant.exe")

_QDRANT_PROCESS = None

def is_qdrant_healthy(timeout: float = 1.0) -> bool:
    try:
        r = requests.get(f"{DEFAULT_QDRANT_URL}/healthz", timeout=timeout)
        return r.status_code == 200
    except Exception:
        try:
            r = requests.get(f"{DEFAULT_QDRANT_URL}/", timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

def ensure_qdrant_running(max_wait_seconds: int = 8) -> bool:
    global _QDRANT_PROCESS

    if is_qdrant_healthy():
        logger.info(f"Native Qdrant server is already running and healthy at {DEFAULT_QDRANT_URL}")
        return True

    if not os.path.exists(QDRANT_BIN_PATH):
        logger.error(f"Qdrant executable not found at {QDRANT_BIN_PATH}")
        return False

    os.makedirs(DEFAULT_STORAGE_DIR, exist_ok=True)

    logger.info(f"Starting native Qdrant server from {QDRANT_BIN_PATH} (storage: {DEFAULT_STORAGE_DIR})...")
    
    env = os.environ.copy()
    env["QDRANT__STORAGE__STORAGE_PATH"] = DEFAULT_STORAGE_DIR
    env["QDRANT__SERVICE__HTTP_PORT"] = str(DEFAULT_QDRANT_HTTP_PORT)
    env["QDRANT__SERVICE__GRPC_PORT"] = str(DEFAULT_QDRANT_GRPC_PORT)
    env["QDRANT__SERVICE__ENABLE_STATIC_CONTENT"] = "true"

    try:
        # Launch detached background process on Windows
        creation_flags = 0
        if os.name == "nt":
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

        cmd = [QDRANT_BIN_PATH]
        config_path = os.path.abspath("./config/config.yaml")
        if os.path.exists(config_path):
            cmd.extend(["--config-path", config_path])

        _QDRANT_PROCESS = subprocess.Popen(
            cmd,
            env=env,
            cwd=os.path.abspath("."),
            creationflags=creation_flags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        t_start = time.time()
        while time.time() - t_start < max_wait_seconds:
            time.sleep(0.4)
            if is_qdrant_healthy():
                logger.info(f"✅ Native Qdrant server started successfully! Web UI available at {DEFAULT_QDRANT_URL}/dashboard")
                return True

        logger.error("Timed out waiting for Qdrant server to become healthy.")
        return False
    except Exception as e:
        logger.error(f"Failed to start Qdrant server process: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = ensure_qdrant_running()
    print("Qdrant Running Status:", success)

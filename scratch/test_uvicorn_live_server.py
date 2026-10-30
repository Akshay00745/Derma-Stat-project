"""
Live Uvicorn Server Startup & Endpoint Verification Script.
Runs uvicorn main:app from D:\HAM10000\backend, probes live endpoints, and verifies clean startup.
"""

import os
import sys
import time
import subprocess
import urllib.request
import urllib.parse
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PYTHON_EXE = sys.executable
BACKEND_DIR = r"D:\HAM10000\backend"
BASE_URL = "http://127.0.0.1:8000"

def main():
    print("=== STARTING LIVE UVICORN SERVER TEST ===")
    print(f"Working Directory: {BACKEND_DIR}")
    print(f"Command: {PYTHON_EXE} -m uvicorn main:app --port 8000")

    # Launch Uvicorn process inside D:\HAM10000\backend
    proc = subprocess.Popen(
        [PYTHON_EXE, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=BACKEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    try:
        # Wait up to 15 seconds for server to start
        print("Waiting for server to listen on port 8000...")
        started = False
        start_time = time.time()
        
        while time.time() - start_time < 20:
            try:
                with urllib.request.urlopen(f"{BASE_URL}/api/health", timeout=2) as resp:
                    if resp.status == 200:
                        started = True
                        break
            except Exception:
                time.sleep(0.5)

        if not started:
            print("ERROR: Server failed to start in time. Subprocess output:")
            proc.terminate()
            stdout, _ = proc.communicate(timeout=5)
            print(stdout)
            sys.exit(1)

        print("[TEST 1/3] GET /api/health succeeded!")
        with urllib.request.urlopen(f"{BASE_URL}/api/health") as resp:
            health = json.loads(resp.read().decode("utf-8"))
            print("  Status:", health.get("status"))
            print("  Model Loaded:", health.get("model_loaded"))
            print("  Device:", health.get("device"))
            assert health.get("model_loaded") is True, "Model is not loaded!"

        print("\n[TEST 2/3] POST /api/predict/sample/mel succeeded!")
        req = urllib.request.Request(
            f"{BASE_URL}/api/predict/sample/mel",
            data=b"",
            headers={"Content-Length": "0"},
            method="POST"
        )
        with urllib.request.urlopen(req) as resp:
            pred = json.loads(resp.read().decode("utf-8"))
            print("  Predicted Category:", pred.get("predicted_name"))
            print("  Confidence:", pred.get("confidence_percentage"), "%")
            print("  Uncertainty Status:", pred.get("uncertainty", {}).get("status"))
            assert "predicted_code" in pred

        print("\n[TEST 3/3] POST /api/predict/upload (Multipart Image Upload)...")
        # Build multipart/form-data payload
        sample_path = r"D:\HAM10000\samples\mel.jpg"
        with open(sample_path, "rb") as f:
            file_bytes = f.read()

        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="mel.jpg"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

        upload_req = urllib.request.Request(
            f"{BASE_URL}/api/predict/upload",
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body))
            },
            method="POST"
        )

        with urllib.request.urlopen(upload_req) as resp:
            upload_pred = json.loads(resp.read().decode("utf-8"))
            print("  Upload Result Predicted:", upload_pred.get("predicted_name"))
            print("  Upload Result Confidence:", upload_pred.get("confidence_percentage"), "%")
            print("  Dimensions:", upload_pred.get("original_dimensions"))
            assert upload_pred.get("predicted_code") is not None

        print("\n=== ALL LIVE UVICORN SERVER TESTS PASSED PERFECTLY! ===")

    finally:
        print("\nTerminating Uvicorn subprocess...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Server shutdown complete.")

if __name__ == "__main__":
    main()

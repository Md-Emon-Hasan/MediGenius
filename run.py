import os, sys, time, subprocess, urllib.request
from threading import Thread

VENV_DIR = ".venv"
IS_WINDOWS = sys.platform == "win32"
PYTHON_EXE = os.path.join(VENV_DIR, "Scripts", "python.exe") if IS_WINDOWS else os.path.join(VENV_DIR, "bin", "python")

def setup():
    # Create virtual environment if it doesn't exist
    if not os.path.exists(VENV_DIR):
        print(f"Creating virtual environment in {VENV_DIR}...")
        subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)

    # Auto-install dependencies
    if os.path.exists("backend/requirements.txt"):
        print("Checking backend deps...")
        # Use the venv's python to install requirements
        subprocess.run([PYTHON_EXE, "-m", "pip", "install", "-r", "backend/requirements.txt"])
    
    if not os.path.exists("frontend/node_modules"):
        print("Installing frontend deps...")
        subprocess.run(["npm", "install"], cwd="frontend", shell=True)

def wait_for_api():
    print("Waiting for backend...")
    for _ in range(60):
        try:
            if urllib.request.urlopen("http://localhost:8000/api/v1/health").status == 200:
                print("Backend ready!")
                return True
        except: pass
        time.sleep(2)

if __name__ == "__main__":
    setup()
    
    # Start backend in background using venv's python
    env = {**os.environ, "PYTHONPATH": os.path.abspath("backend")}
    Thread(target=lambda: subprocess.run(
        [PYTHON_EXE, "-m", "uvicorn", "app.main:app", "--port", "8000"], 
        cwd="backend", env=env
    ), daemon=True).start()

    # Wait and launch frontend
    if wait_for_api():
        subprocess.run(["npm", "run", "dev"], cwd="frontend", shell=True)

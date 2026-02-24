import os, sys, time, subprocess, urllib.request
from threading import Thread

def setup():
    # Auto-install dependencies
    if os.path.exists("backend/requirements.txt"):
        print("Checking backend deps...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "backend/requirements.txt"])
    
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
    
    # Start backend in background
    env = {**os.environ, "PYTHONPATH": os.path.abspath("backend")}
    Thread(target=lambda: subprocess.run(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8000"], 
        cwd="backend", env=env
    ), daemon=True).start()

    # Wait and launch frontend
    if wait_for_api():
        subprocess.run(["npm", "run", "dev"], cwd="frontend", shell=True)

import subprocess
import sys
import os
import signal
import time
from threading import Thread

def run_backend():
    print("Starting Backend...")
    # Add backend/app to Python path context
    env = os.environ.copy()
    backend_dir = os.path.join(os.getcwd(), 'backend')
    env['PYTHONPATH'] = backend_dir
    
    # Run uvicorn directly
    subprocess.run([sys.executable, "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000"], cwd=backend_dir, env=env)

def run_frontend():
    print("Starting Frontend...")
    frontend_dir = os.path.join(os.getcwd(), 'frontend')
    # Use npm run dev
    subprocess.run(["npm", "run", "dev"], cwd=frontend_dir, shell=True)

if __name__ == "__main__":
    try:
        # Start backend in a separate thread/process
        backend_thread = Thread(target=run_backend)
        backend_thread.daemon = True
        backend_thread.start()
        
        # Give backend a moment to initialize
        time.sleep(2)
        
        # Start frontend in this process (blocks until interrupted)
        run_frontend()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)

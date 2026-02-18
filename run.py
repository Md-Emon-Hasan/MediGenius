import subprocess
import sys
import os
import time
import urllib.request
import urllib.error
from threading import Thread


def run_backend():
    print("Starting Backend...")
    env = os.environ.copy()
    backend_dir = os.path.join(os.getcwd(), 'backend')
    env['PYTHONPATH'] = backend_dir
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000"],
        cwd=backend_dir,
        env=env
    )


def wait_for_backend(url="http://127.0.0.1:8000/api/v1/health", timeout=120, interval=2):
    """Poll the backend health endpoint until it responds or timeout is reached."""
    print(f"Waiting for backend to be ready at {url} ...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    print("Backend is ready!")
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(interval)
    print("WARNING: Backend did not become ready within timeout. Starting frontend anyway.")
    return False


def run_frontend():
    print("Starting Frontend...")
    frontend_dir = os.path.join(os.getcwd(), 'frontend')
    subprocess.run(["npm", "run", "dev"], cwd=frontend_dir, shell=True)


if __name__ == "__main__":
    try:
        # Start backend in a background thread
        backend_thread = Thread(target=run_backend)
        backend_thread.daemon = True
        backend_thread.start()

        # Wait until backend health endpoint responds (up to 2 minutes)
        wait_for_backend()

        # Now start the frontend (blocks until interrupted)
        run_frontend()

    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)

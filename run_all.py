"""
AuditHub - Unified App Runner
===============================

Launches the FastAPI backend and Streamlit dashboard concurrently, logging output from both.
"""

import sys
import time
import subprocess
import signal
from pathlib import Path

# Target ports from constants/settings
FASTAPI_CMD = [
    sys.executable, "-m", "uvicorn", "src.api.main:app",
    "--host", "0.0.0.0", "--port", "8000"
]

STREAMLIT_CMD = [
    sys.executable, "-m", "streamlit", "run", "app/main.py",
    "--server.address", "0.0.0.0", "--server.port", "8501"
]


def main():
    print("=============================================================================")
    print("                  Starting AuditHub Unified App Runner                       ")
    print("=============================================================================")
    
    processes = []
    
    try:
        # 1. Launch FastAPI Backend
        print("[FastAPI] Launching Backend on http://localhost:8000 ...")
        fastapi_proc = subprocess.Popen(
            FASTAPI_CMD,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        processes.append(fastapi_proc)
        
        # 2. Launch Streamlit Dashboard
        print("[Streamlit] Launching Dashboard on http://localhost:8501 ...")
        streamlit_proc = subprocess.Popen(
            STREAMLIT_CMD,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        processes.append(streamlit_proc)
        
        print("\nPress Ctrl+C to terminate both servers concurrently.\n")
        
        # Monitor outputs in non-blocking fashion
        # Simple loop to print lines
        import select
        
        # Since select.select is not always reliable on Windows for pipes,
        # we can do simple non-blocking reads or direct prints from stdout using threads.
        # But for Windows, a simple poll/sleep loop is safest and most portable without threads.
        import os
        if os.name == 'nt':
            # Set pipes to non-blocking or just read line-by-line using threads,
            # but standard print monitoring using short poll checks is extremely robust.
            import threading
            
            def log_reader(pipe, prefix):
                try:
                    for line in iter(pipe.readline, ''):
                        if line:
                            print(f"[{prefix}] {line.strip()}")
                except Exception:
                    pass
                    
            t1 = threading.Thread(target=log_reader, args=(fastapi_proc.stdout, "API"), daemon=True)
            t2 = threading.Thread(target=log_reader, args=(streamlit_proc.stdout, "UI"), daemon=True)
            t1.start()
            t2.start()
            
            while True:
                # Check if any process exited
                if fastapi_proc.poll() is not None:
                    print(f"FastAPI process terminated with exit code {fastapi_proc.returncode}")
                    break
                if streamlit_proc.poll() is not None:
                    print(f"Streamlit process terminated with exit code {streamlit_proc.returncode}")
                    break
                time.sleep(1)
        else:
            # Linux/Mac select loop
            while True:
                reads = [fastapi_proc.stdout, streamlit_proc.stdout]
                readable, _, _ = select.select(reads, [], [], 1)
                
                for src in readable:
                    line = src.readline().strip()
                    if line:
                        prefix = "API" if src == fastapi_proc.stdout else "UI"
                        print(f"[{prefix}] {line}")
                        
                if fastapi_proc.poll() is not None or streamlit_proc.poll() is not None:
                    break
                    
    except KeyboardInterrupt:
        print("\nShutdown signal received. Terminating servers...")
    finally:
        # Clean termination of all subprocesses
        for proc in processes:
            if proc.poll() is None:
                print(f"Killing process {proc.pid}...")
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    
        print("AuditHub runner terminated cleanly.")


if __name__ == "__main__":
    main()

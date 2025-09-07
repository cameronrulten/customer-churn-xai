import subprocess
import sys
from pathlib import Path

def main():
    # Launch Streamlit with your dashboard file
    dash = Path(__file__).with_name("dashboard.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(dash),
           "--server.address", "127.0.0.1", "--server.port", "8501"]
    # inherit stdout/stderr so you see logs
    raise SystemExit(subprocess.call(cmd))

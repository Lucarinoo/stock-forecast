import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

cmd = [
    sys.executable,
    "-m",
    "streamlit",
    "run",
    "scripts/dashboard.py",
    "--server.port=8501",
    "--server.headless=true",
]

subprocess.run(cmd, cwd=str(ROOT))

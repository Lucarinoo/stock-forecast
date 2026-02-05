import subprocess
import datetime
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "outputs", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join(LOG_DIR, f"retrain_{now}.log")

cmd = [
    sys.executable,
    os.path.join(ROOT, "scripts", "train_model.py"),
]

with open(log_file, "w", encoding="utf-8") as f:
    f.write(f"Retrain started: {now}\n\n")
    result = subprocess.run(cmd, stdout=f, stderr=f)
    f.write("\nRetrain finished.\n")
    f.write(f"Return code: {result.returncode}\n")

print(f"✅ Retrain done. Log: {log_file}")

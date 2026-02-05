import subprocess
import sys
from pathlib import Path
from datetime import datetime
import os
os.environ["PYTHONIOENCODING"] = "utf-8"



ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "outputs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = LOG_DIR / f"scheduler_run_daily_{ts}.log"

def run(cmd, f):
    f.write("\n>> " + " ".join(map(str, cmd)) + "\n")
    f.flush()
    r = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=f, text=True)
    f.write(f"\n[returncode={r.returncode}]\n")
    f.flush()
    if r.returncode != 0:
        raise SystemExit(r.returncode)

if __name__ == "__main__":
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Started: {ts}\n")
        f.write(f"Python: {sys.executable}\n")
        f.write(f"ROOT: {ROOT}\n")

        run([sys.executable, "scripts/run_forecast.py", "--bands"], f)
        run([sys.executable, "scripts/evaluate_forecasts.py"], f)

        f.write("\n✅ Daily run finished.\n")

    print(f"Log written: {log_path}")

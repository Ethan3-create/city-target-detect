"""Launch YOLOv9c training as a detached background process.

Usage: python launch_training.py
The training process will run independently of this script.
Output is redirected to a log file.
"""
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

PROJECT = Path(__file__).parent
PYTHON = sys.executable  # Use the same Python that runs this script
LOG_DIR = PROJECT / "outputs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"train_v9c_{timestamp}.log"
pid_file = LOG_DIR / "train_v9c.pid"

# Build command
cmd = [
    PYTHON, "scripts/02_train.py",
    "--config", "configs/train_gpu_v9c.yaml",
    "--resume", "weights/checkpoints_v9/last.pt",
]

# On Windows, use DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP to create
# a truly independent process that survives parent termination
creationflags = 0
if sys.platform == "win32":
    creationflags = (
        subprocess.DETACHED_PROCESS      # 0x00000008
        | subprocess.CREATE_NEW_PROCESS_GROUP  # 0x00000200
        | subprocess.CREATE_NO_WINDOW    # 0x08000000
    )

print(f"Starting YOLOv9c training...")
print(f"  Command: {' '.join(cmd)}")
print(f"  WorkDir: {PROJECT}")
print(f"  LogFile: {log_file}")
print(f"  StartTime: {datetime.now()}")

with open(log_file, "w", encoding="utf-8") as log:
    log.write(f"Training started at {datetime.now()}\n")
    log.write(f"Command: {' '.join(cmd)}\n")
    log.write(f"Working dir: {PROJECT}\n")
    log.flush()

    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT),
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

# Write PID file
pid_file.write_text(str(proc.pid))
print(f"  PID: {proc.pid}")
print(f"  PID file: {pid_file}")
print(f"\nTraining launched as detached process. Check log: {log_file}")
print(f"To check progress: tail -f \"{log_file}\"")
print(f"To stop: taskkill /PID {proc.pid} /F")

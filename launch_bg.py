import subprocess, os, sys

# Launch resume_v2.py as a fully detached background process
# This survives even if this terminal crashes

work_dir = r"C:\Users\Manas Kushwaha\Desktop\Cyberbullying Detector plus Safe Reply Generator"
python_exe = os.path.join(work_dir, "venv", "Scripts", "python.exe")
script = os.path.join(work_dir, "resume_v2.py")
log_file = os.path.join(work_dir, "resume_bg.log")

with open(log_file, "w") as fout:
    proc = subprocess.Popen(
        [python_exe, "-u", script],
        stdout=fout,
        stderr=subprocess.STDOUT,
        cwd=work_dir,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    )
    print(f"Launched background process PID: {proc.pid}")
    print(f"Monitor progress with: Get-Content '{log_file}' -Wait")

import os
import subprocess
import socket
import sys
import psutil  # pip install psutil
from core.constants import DB_PATH

class ChromaLauncher:
    """
    A reusable launcher for ChromaDB local server.
    Handles:
    - checking if Chroma is already running
    - starting the server safely
    - cross-platform execution
    - storing PID to prevent multiple starts
    """

    PID_FILE = "chroma.pid"

    def __init__(self, data_path=DB_PATH, port=8000):
        self.data_path = data_path
        self.port = port

    # ------------------------------
    # UTIL: Check if port already used
    # ------------------------------
    def is_port_in_use(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", self.port)) == 0

    # ------------------------------
    # UTIL: Check PID file and running process
    # ------------------------------
    def is_already_running(self) -> bool:
        if os.path.exists(self.PID_FILE):
            try:
                pid = int(open(self.PID_FILE).read())
                if psutil.pid_exists(pid):
                    print(f"✔ ChromaDB already running (PID {pid})")
                    return True
            except Exception:
                pass
        # fallback to port check
        return self.is_port_in_use()

    # ------------------------------
    # Start Chroma server
    # ------------------------------
    def start(self):
        print("=== ChromaLauncher ===")

        if self.is_already_running():
            return False  # already running

        # Ensure folder exists
        os.makedirs(self.data_path, exist_ok=True)

        # Build command
        cmd = [
            "chroma",
            "run",
            "--path", self.data_path,
            "--host", "0.0.0.0",
            "--port", str(self.port)
        ]

        print(f"▶ Starting ChromaDB at {self.data_path} on port {self.port}")
        print("Command:", " ".join(cmd))

        try:
            process = subprocess.Popen(cmd)
            # Save PID to file
            with open(self.PID_FILE, "w") as f:
                f.write(str(process.pid))
            print(f"✔ ChromaDB started (PID {process.pid})")
            return True
        except FileNotFoundError:
            print("\n❌ ERROR: 'chroma' CLI not found.")
            print("Install with: pip install chromadb")
            sys.exit(1)
        except Exception as e:
            print("❌ Failed to launch ChromaDB:", e)
            sys.exit(1)

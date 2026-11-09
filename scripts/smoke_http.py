"""Start a temporary localhost server, verify HTTP, and always stop it."""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    with Path("reports/uvicorn.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "fraud_detection.api.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8765",
            ],
            stdout=log,
            stderr=log,
        )
        try:
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError("Server stopped; inspect reports/uvicorn.log")
                try:
                    with urllib.request.urlopen(
                        "http://127.0.0.1:8765/health", timeout=1
                    ) as response:
                        result = json.load(response)
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.1)
            else:
                raise RuntimeError("Server did not become ready within ten seconds")
            assert result == {"status": "ready", "model_loaded": True}
            Path("reports/http_smoke.json").write_text(json.dumps(result) + "\n")
            print("Local HTTP readiness smoke passed")
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()

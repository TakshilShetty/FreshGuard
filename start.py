"""One command to run everything: MongoDB check, Streamlit, FastAPI backend, Caddy, and ESP32 AI Pipeline.

Usage:
    python start.py                    (auto-detects ESP32 on COM port, starts UI + API + Pipeline)
    python start.py --port COM5        (explicit port)
    python start.py --no-ingest        (start frontend & backend without serial hardware)

Endpoints:
    Dashboard:   http://localhost:5000  (or http://freshguard.local)
    Backend API: http://localhost:8000  (API Docs: http://localhost:8000/docs)
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from zeroconf import IPVersion, ServiceInfo, Zeroconf

from src.config import DEFAULT_SPECIMEN, MDNS_NAME, PORT
from src.sensor_reader import list_ports

NAME = MDNS_NAME
HERE = Path(__file__).parent


def wifi_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("10.255.255.255", 1))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"


def up(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def detect_esp32_port() -> str | None:
    ports = list_ports()
    if not ports:
        return None
    for p in ports:
        dev = p.split()[0]
        desc = p.lower()
        if "cp210" in desc or "ch340" in desc or "esp32" in desc or "usb-to-uart" in desc:
            return dev
    return ports[0].split()[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Start FreshGuard Full Stack (Frontend, Backend API, and Hardware Pipeline)")
    ap.add_argument("--port", help="ESP32 serial COM port (e.g. COM5)")
    ap.add_argument("--specimen", default=DEFAULT_SPECIMEN, help=f"Specimen label (default: {DEFAULT_SPECIMEN})")
    ap.add_argument("--no-ingest", action="store_true", help="Do not start serial ingestion pipeline")
    ap.add_argument("--no-api", action="store_true", help="Do not start FastAPI backend server")
    a = ap.parse_args()

    if not up(27017):
        sys.exit("MongoDB is not running on port 27017. Please start MongoDB service (net start MongoDB).")

    procs: list[subprocess.Popen] = []

    # 1. Start Streamlit Frontend (bound to 0.0.0.0 for LAN access)
    procs.append(
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app.py",
                "--server.port",
                str(PORT),
                "--server.address",
                "0.0.0.0",
                "--server.enableCORS",
                "false",
                "--server.enableXsrfProtection",
                "false",
                "--server.headless",
                "true",
            ],
            cwd=HERE,
        )
    )

    # 2. Start FastAPI Backend REST API Server (bound to 0.0.0.0 for LAN access)
    if not a.no_api:
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "src.server:app",
                    "--port",
                    "8000",
                    "--host",
                    "0.0.0.0",
                ],
                cwd=HERE,
            )
        )

    # 3. Start Caddy reverse proxy (optional port-80 router for LAN clients)
    caddy_proc: subprocess.Popen | None = None
    if shutil.which("caddy"):
        # Terminate any orphaned caddy instances to release port 80 & 2019
        try:
            subprocess.run(["taskkill", "/F", "/IM", "caddy.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.4)
        except Exception:
            pass

        caddy_config = f"""{{
    admin off
}}

:80 {{
    handle /api/* {{
        reverse_proxy 127.0.0.1:8000
    }}
    handle /docs* {{
        reverse_proxy 127.0.0.1:8000
    }}
    handle /openapi.json {{
        reverse_proxy 127.0.0.1:8000
    }}
    handle /ws/* {{
        reverse_proxy 127.0.0.1:8000
    }}
    handle {{
        reverse_proxy 127.0.0.1:{PORT}
    }}
}}
"""
        caddyfile = Path(tempfile.gettempdir()) / "freshguard.Caddyfile"
        caddyfile.write_text(caddy_config, encoding="utf-8")
        try:
            caddy_proc = subprocess.Popen(["caddy", "run", "--config", str(caddyfile), "--adapter", "caddyfile"])
            procs.append(caddy_proc)
        except Exception:
            pass

    # 4. Start Hardware Ingestion Pipeline
    if not a.no_ingest:
        port = a.port or detect_esp32_port()
        if port:
            ingest_script = HERE / "scripts" / "ingest.py"
            if not ingest_script.exists():
                ingest_script = HERE / "ingest.py"
            print(f"[*] Attaching ESP32 on {port} -> Live AI Ingestion Pipeline")
            procs.append(
                subprocess.Popen(
                    [sys.executable, str(ingest_script), "--port", port, "--specimen", a.specimen],
                    cwd=HERE,
                )
            )
        else:
            print("[!] No ESP32 COM port found. Running without live hardware ingestion.")

    # 5. mDNS Zeroconf registration
    ip = wifi_ip()
    zc = Zeroconf(ip_version=IPVersion.V4Only)
    try:
        zc.register_service(
            ServiceInfo(
                "_http._tcp.local.",
                f"{NAME}._http._tcp.local.",
                addresses=[socket.inet_aton(ip)],
                port=80,
                server=f"{NAME}.local.",
            )
        )
    except Exception:
        pass

    print("\n=======================================================")
    print(f" FreshGuard LAN Server Running Successfully! [ONLINE]")
    print(f"=======================================================")
    print(f" Local Machine:       http://localhost:{PORT}")
    print(f" LAN Wi-Fi Access:    http://{ip}  (Port 80) or http://{ip}:{PORT}")
    print(f" mDNS Hostname:       http://{NAME}.local")
    if not a.no_api:
        print(f" REST API (LAN):      http://{ip}/api/status  |  http://{ip}:8000/docs")
    print(f" Press Ctrl+C to terminate all services.\n")

    # Only treat core services as fatal if they terminate; caddy is optional
    core_procs = [p for p in procs if p is not caddy_proc]
    try:
        while all(p.poll() is None for p in core_procs):
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping FreshGuard services...")
    finally:
        zc.unregister_all_services()
        zc.close()
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()

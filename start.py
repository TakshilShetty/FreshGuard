"""One command to run everything: MongoDB check, Streamlit, Caddy reverse proxy, mDNS name.

    python start.py        (Ctrl+C stops what it started)
Dashboard: http://freshguard.local  (or http://<wifi-ip> for devices without mDNS).
"""
import os, socket, subprocess, sys, tempfile, time
from pathlib import Path

from zeroconf import IPVersion, ServiceInfo, Zeroconf

NAME, PORT = "freshguard", 5000
HERE = Path(__file__).parent


def wifi_ip() -> str:
    # ponytail: default-route interface; assumes the hotspot is the active route
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]


def up(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main() -> None:
    if not up(27017):
        sys.exit("MongoDB is not running on 27017 - start the 'MongoDB' Windows service (admin: net start MongoDB).")
    ip = wifi_ip()
    caddyfile = Path(tempfile.gettempdir()) / "freshguard.Caddyfile"
    caddyfile.write_text(f":80 {{\n\treverse_proxy 127.0.0.1:{PORT}\n}}\n")
    procs = [
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(PORT),
                          "--server.address", "127.0.0.1", "--server.headless", "true"], cwd=HERE),
        subprocess.Popen(["caddy", "run", "--config", str(caddyfile), "--adapter", "caddyfile"]),
    ]
    zc = Zeroconf(ip_version=IPVersion.V4Only)
    zc.register_service(ServiceInfo("_http._tcp.local.", f"{NAME}._http._tcp.local.", addresses=[socket.inet_aton(ip)],
                                    port=80, server=f"{NAME}.local."))
    print(f"\nFreshGuard up:  http://{NAME}.local   |   http://{ip}   (Ctrl+C to stop)\n")
    try:
        while all(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        zc.unregister_all_services(); zc.close()
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()

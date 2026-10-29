"""Backward-compatible alias for scripts/ingest.py."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parent / "scripts" / "ingest.py"), run_name="__main__")

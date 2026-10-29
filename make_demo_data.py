"""Backward-compatible alias for scripts/make_demo_data.py."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parent / "scripts" / "make_demo_data.py"), run_name="__main__")

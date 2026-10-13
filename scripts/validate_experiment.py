"""Run the complete Experiment 0 workflow in a single GPU allocation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.validation import main

if __name__ == "__main__":
    raise SystemExit(main())

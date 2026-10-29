"""Retired pilot-selection command; historical evidence is never deleted here.

Experiments 2 and 3 now use predeclared settings. Both Experiment 2 learning
rates remain in the reported grid. Old pilot files and receipts, if present,
are historical evidence and cannot authorize or alter the current campaign.
"""
from __future__ import annotations

import argparse
from pathlib import Path

RETIRED = ("Pilot selection is retired. Experiment 2 evaluates both predeclared learning rates; "
           "Experiment 3 uses fixed methods and the primary-rate baseline. "
           "Existing pilot evidence is preserved and is not used to choose settings.")


def selection_path(design: str) -> Path:
    """Locate an old receipt for explicit historical inspection only."""
    from src.utils import paths
    if design in {"cpt", "retention_pilot"}:
        return paths.experiment_dir(2) / "selection" / "learning_rate.json"
    if design == "retention":
        return paths.experiment_dir(3) / "selection" / "adapters.json"
    raise ValueError(f"Unknown historical selection design: {design}")


def load_selection(*args, **kwargs):
    """Fail closed even if a historical receipt exists."""
    raise ValueError(RETIRED)


def freeze(*args, **kwargs):
    raise ValueError(RETIRED)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("learning-rate", "adapters"))
    parser.parse_args(argv)
    print(RETIRED)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

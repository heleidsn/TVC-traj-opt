#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TVC rocket trajectory optimization — project main entry (GUI).

This module only launches the UI. Optimization methods live under ``scripts/`` and are
invoked from the GUI (Methods 1–7):

- Method 1: ``scripts/tvc_traj_opt.py`` (Crocoddyl custom calcDiff)
- Methods 2–3: ``scripts/tvc_traj_opt_pinocchio.py`` (FDDP / BoxFDDP)
- Methods 4–5–7: ``scripts/tvc_traj_opt_acados.py`` (Acados: tracking / min-time / free-tf)
- Method 6: ``scripts/tvc_traj_opt_acados_min_time.py`` (Spannagl FFTF)

Usage::

    python run_tvc_traj_opt.py

After installation::

    tvc-traj-opt
"""

from __future__ import annotations

import os
import sys


def _ensure_scripts_on_path() -> str:
    """Add ``scripts/`` to ``sys.path`` so imports match ``python scripts/...``."""
    root = os.path.dirname(os.path.abspath(__file__))
    scripts = os.path.join(root, "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return scripts


def main(argv: list[str] | None = None) -> int:
    """
    Launch the trajectory optimization GUI.

    Parameters
    ----------
    argv : optional
        Qt command-line arguments; defaults to ``sys.argv``.

    Returns
    -------
    int
        Qt exit code.
    """
    if argv is not None:
        sys.argv = list(argv)
    _ensure_scripts_on_path()
    from tvc_traj_opt_gui import run_gui

    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())

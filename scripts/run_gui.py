#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Launch script for TVC Trajectory Optimization GUI

Usage:
    python run_gui.py
"""

import sys
import os

# Add scripts directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from tvc_traj_opt_gui import main

if __name__ == '__main__':
    main()

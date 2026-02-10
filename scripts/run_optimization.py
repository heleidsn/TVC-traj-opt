#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Launch script for TVC Trajectory Optimization (Command-line)

Usage:
    python -u run_optimization.py
"""

import sys
import os

# Add scripts directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from tvc_traj_opt import solve_once, plot_trajectory
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Run optimization
    xs, us, logger = solve_once(dt=0.02, N=100, max_iter=100)
    print("\n" + "="*50)
    print("Solved. N =", len(us))
    print("x0 =", xs[0])
    print("xN =", xs[-1])
    
    # Plot results
    print("\nPlotting trajectory...")
    x_goal = np.array([0., 0., 10.])  # Target position
    fig = plot_trajectory(xs, us, dt=0.02, logger=logger, x_goal=x_goal)
    plt.show()

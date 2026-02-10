#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for TVC Rocket Trajectory Optimization
"""

from setuptools import setup, find_packages

setup(
    name="tvc-rocket-trajectory-optimization",
    version="1.0.0",
    description="TVC Rocket Trajectory Optimization using Crocoddyl",
    author="Lei He",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "crocoddyl>=1.0.0",
        "PyQt5>=5.15.0",
        "pyyaml>=5.4.0",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "tvc-opt-gui=scripts.tvc_traj_opt_gui:main",
            "tvc-opt=scripts.tvc_traj_opt:solve_once",
        ],
    },
)

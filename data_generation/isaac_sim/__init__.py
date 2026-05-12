"""
Copyright (c) 2026 Style Machine LLC. All rights reserved.

Isaac Sim Data Generation Module

Tools for generating physics simulation training data from Isaac Sim
and PyBullet robotics simulations.
"""

try:
    from .generate_robotics_hdf5 import RoboticsGenerator, generate_schema_hdf5
    __all__ = ['RoboticsGenerator', 'generate_schema_hdf5']
except ImportError:
    # pybullet not installed — helpers (state_extraction, hdf5_utils) remain importable
    __all__ = []

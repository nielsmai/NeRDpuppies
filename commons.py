# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import numpy as np

DATASET_MODES = ['transition', 'trajectory']

CONTACT_FREE_DEPTH = 10000.
CONTACT_DEPTH_LOWER_RATIO = -2.
CONTACT_DEPTH_UPPER_RATIO = 4.
MIN_CONTACT_EVENT_THRESHOLD = 0.12

# --- Joint Position Limits ---
JOINT_Q_MIN = {
    'Cartpole': np.array([-3.5, -np.pi]),
    'Pendulum': -np.pi,
    'Franka': -10.,
    'Ant': np.array([-10., 1.0, -10., -1., -1., -1., -1., np.deg2rad(-40), np.deg2rad(30), np.deg2rad(-40), np.deg2rad(-100), np.deg2rad(-40), np.deg2rad(-100), np.deg2rad(-40), np.deg2rad(30)]),
    'CubeToss': np.array([-20.0, -20.0, 0.94, -1., -1., -1., -1.]),
    'AnyMAL': np.array([-10., 0.8, -10., -1., -1., -1., -1., -0.49, -np.pi, -np.pi, -0.72, -np.pi, -np.pi, -0.49, -np.pi, -np.pi, -0.72, -np.pi, -np.pi]),
    # Reordered to match Warp Topology: RH, LH, RF, LF
    'Pupper': np.array([
        -0.78, -1.57, -2.62,  # RH
        -0.78, -1.57, -2.62,  # LH
        -0.78, -1.57, -2.62,  # RF
        -0.78, -1.57, -2.62   # LF
    ], dtype=np.float32)
}

JOINT_Q_MAX = {
    'Cartpole': np.array([3.5, np.pi]),
    'Pendulum': np.pi,
    'Franka': 10.,
    'Ant': np.array([10., 1.3, 10., 1., 1., 1., 1., np.deg2rad(40), np.deg2rad(100), np.deg2rad(40), np.deg2rad(-30), np.deg2rad(40), np.deg2rad(-30), np.deg2rad(40), np.deg2rad(100)]),
    'CubeToss': np.array([20.0, 20.0, 5.0, 1., 1., 1., 1.]),
    'AnyMAL': np.array([10., 1.0, 10., 1., 1., 1., 1., 0.72, -np.pi, -np.pi, 0.49, -np.pi, -np.pi, 0.72, -np.pi, -np.pi, 0.49, -np.pi, -np.pi]),
    # Reordered to match Warp Topology: RH, LH, RF, LF
    'Pupper': np.array([
        0.78, 1.57, 2.62,     # RH
        0.78, 1.57, 2.62,     # LH
        0.78, 1.57, 2.62,     # RF
        0.78, 1.57, 2.62      # LF
    ], dtype=np.float32)
}

# --- Joint Velocity Limits ---
JOINT_QD_MIN = {
    'Cartpole': -10.0,
    'Pendulum': np.array([-np.pi * 2., -np.pi * 4.]),
    'Franka': -np.pi,
    'Ant': -np.array([np.pi, np.pi, np.pi, 1., 1., 1., np.pi * 2., np.pi * 2., np.pi * 2., np.pi * 2., np.pi * 2., np.pi * 2., np.pi * 2., np.pi * 2.]),
    'CubeToss': np.array([-2.0, -2.0, -2.0, -3.5, -3.5, -3.5]),
    'AnyMAL': -np.array([np.pi, np.pi, np.pi, 0.25, 0.25, 0.25, np.pi * 2., np.pi * 2., np.pi * 2., np.pi * 2., np.pi * 2., np.pi * 2., np.pi * 2., np.pi * 2., np.pi * 2., np.pi * 2., np.pi * 2., np.pi * 2.]),
    'Pupper': -np.array([
        7.0, 8.0, 8.0,  # RH
        7.0, 8.0, 8.0,  # LH
        7.0, 8.0, 8.0,  # RF
        7.0, 8.0, 8.0   # LF
    ], dtype=np.float32)
}

JOINT_QD_MAX = {
    'Cartpole': 10.0,
    'Pendulum': np.array([np.pi * 2., np.pi * 4.]),
    'Franka': np.pi,
    'Ant': np.array([np.pi, np.pi, np.pi, 1., 1., 1., np.pi * 2., np.pi * 2., np.pi * 2., np.pi * 2., np.pi * 2., np.pi * 2., np.pi * 2., np.pi * 2.]),
    'CubeToss': np.array([2.0, 2.0, 2.0, 3.5, 3.5, 3.5]),
    'AnyMAL': np.array([np.pi, np.pi, np.pi, 0.25, 0.25, 0.25, np.pi * 2., np.pi * 2., np.pi * 2., np.pi * 2., np.pi * 2., np.pi * 2., np.pi * 2., np.pi * 2., np.pi * 2., np.pi * 2., np.pi * 2., np.pi * 2.]),
    'Pupper': np.array([
        7.0, 8.0, 8.0,  # RH
        7.0, 8.0, 8.0,  # LH
        7.0, 8.0, 8.0,  # RF
        7.0, 8.0, 8.0   # LF
    ], dtype=np.float32)
}

# --- Torque / Action Scaling ---
JOINT_ACT_SCALE = {
    'Cartpole': np.array([1500., 0.]),
    'Pendulum': 1500.,
    'Franka': np.array([100., 100., 100., 100., 20., 20., 20.]),
    'Ant': 400.,
    'AnyMAL': 1.5 * np.array([50.0, 40.0, 8.0, 50.0, 40.0, 8.0, 50.0, 40.0, 8.0, 50.0, 40.0, 8.0]),
    # Increased to 20.0 to prevent collapse during impact
    'Pupper': np.array([3.0] * 12)
}

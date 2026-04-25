"""
NeRD PPO Policy Deployment - Pupper Robot (Fixed for Real Hardware)
====================================================================
Now uses env.reset() to stand up, then runs the policy with correct observations.
"""

import os
import sys
import time
import numpy as np
import pickle
import argparse
import msgpack

# Add StanfordQuadruped to path
sys.path.insert(0, '/home/pi/puppersim_deploy/StanfordQuadruped')
from djipupper.HardwareInterface import HardwareInterface

# ===== Environment factory (still needed for env.reset() on real robot) =====
import puppersim
import gin
from pybullet_envs.minitaur.envs_v2 import env_loader
from pybullet import COV_ENABLE_GUI
import puppersim.data as pd

def create_pupper_env(run_on_robot=True, render=False):
    CONFIG_DIR = puppersim.getPupperSimPath()
    if run_on_robot:
        _CONFIG_FILE = os.path.join(CONFIG_DIR, "config", "pupper_pmtg_robot.gin")
    else:
        _CONFIG_FILE = os.path.join(CONFIG_DIR, "config", "pupper_pmtg.gin")
    gin.bind_parameter("scene_base.SceneBase.data_root", pd.getDataPath() + "/")
    gin.parse_config_file(_CONFIG_FILE)
    gin.bind_parameter("SimulationParameters.enable_rendering", render)
    env = env_loader.load()
    env._pybullet_client.configureDebugVisualizer(COV_ENABLE_GUI, 0)
    return env

# ==================== Constants ====================
DEFAULT_JOINT_Q = np.array([0.0, 0.9, -1.8] * 4, dtype=np.float32)
ACTION_SCALE = 0.3
SIT_POSE = np.array([0.0, 1.2, -2.4] * 4, dtype=np.float32)
MAX_TILT_RAD = np.deg2rad(10.0)

# Action filter: 0 = no change, 1 = instantaneous
ACTION_FILTER_ALPHA = 0.8   # was 0.2 -> increased for less lag

# ==================== Helper Functions ====================
def load_policy(path):
    """Load NeRD policy from .npz."""
    d = np.load(path)
    obs_mean = d["obs_mean"].astype(np.float32)
    obs_std = np.sqrt(d["obs_var"].astype(np.float32) + 1e-8)
    layers = [
        (d["w0"].astype(np.float32), d["b0"].astype(np.float32)),
        (d["w2"].astype(np.float32), d["b2"].astype(np.float32)),
        (d["w4"].astype(np.float32), d["b4"].astype(np.float32)),
    ]
    mu_w = d["mu_w"].astype(np.float32)
    mu_b = d["mu_b"].astype(np.float32)
    return dict(obs_mean=obs_mean, obs_std=obs_std, layers=layers, mu=(mu_w, mu_b))

def run_policy(net, obs):
    x = np.clip((obs - net["obs_mean"]) / net["obs_std"], -5.0, 5.0)
    for (w, b) in net["layers"]:
        x = np.maximum(0.0, np.dot(x, w.T) + b)
    mu_w, mu_b = net["mu"]
    return np.dot(x, mu_w.T) + mu_b

def read_state(hw):
    """Read the most recent state from the hardware interface."""
    state = None
    while True:
        data = hw.reader.chew()
        if not data:
            break
        try:
            state = msgpack.unpackb(data)
        except Exception:
            pass
    return state

def wait_for_valid_state(hw, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = read_state(hw)
        if state is not None:
            return state
        time.sleep(0.01)
    raise TimeoutError("No state received from robot.")

def quat_to_projected_gravity(qx, qy, qz, qw):
    rx = 2.0 * (qx * qz + qw * qy)
    ry = 2.0 * (qy * qz - qw * qx)
    rz = 1.0 - 2.0 * (qx * qx + qy * qy)
    return np.array([-rx, -ry, -rz], dtype=np.float32)

def build_obs(state, prev_action):
    """
    Build observation exactly as in training.
    IMPORTANT: Now uses FULL previous action (12 dims) instead of only [:4].
    """
    joint_q = np.array(state['pos'], dtype=np.float32)
    joint_qd = np.array(state['vel'], dtype=np.float32)

    roll = float(state['roll'])
    pitch = float(state['pitch'])
    yaw = float(state['yaw'])

    # Convert euler to quaternion for gravity projection
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    proj_grav = quat_to_projected_gravity(qx, qy, qz, qw)

    # Linear velocity: if your training used an estimator, implement one.
    # For now, keeping zeros is safe only if the policy was trained with zeros.
    # Better: set from IMU integration or joint kinematics. We'll leave configurable.
    lin_vel = np.zeros(3, dtype=np.float32)

    ang_vel = np.array([state['roll_rate'], state['pitch_rate'], state['yaw_rate']], dtype=np.float32)

    # ----- FIX: use full previous action (12 dims) -----
    extra = prev_action   # now full 12

    return np.concatenate([proj_grav, lin_vel, ang_vel, joint_q, joint_qd, extra])

def action_to_targets(action):
    targets = DEFAULT_JOINT_Q + ACTION_SCALE * np.clip(action, -1.0, 1.0)
    return np.clip(targets, -1.5, 1.5).astype(np.float32)

def interpolate_to_pose(hw, target, steps=80, dt=0.02, start_pos=None, desc=""):
    """Smoothly interpolate from current to target joint positions."""
    if start_pos is not None:
        curr = start_pos.copy().astype(np.float32)
    else:
        state = read_state(hw)
        if state is not None:
            curr = np.array(state['pos'], dtype=np.float32)
        else:
            curr = target.copy().astype(np.float32)

    target = target.astype(np.float32)
    for i in range(steps):
        alpha = (i + 1.0) / steps
        cmd = curr + alpha * (target - curr)
        hw.set_actuator_postions(cmd.reshape(3, 4))
        time.sleep(dt)
    if desc:
        print(desc)

def stand_up_and_verify(hw, env):
    """Use env.reset() to stand up, then verify robot is upright."""
    print("Calling env.reset() to stand up...")
    env.reset()
    print("env.reset() finished – robot should be standing.")
    time.sleep(1.5)   # wait for motion to settle

    # Read actual joint positions after standing
    try:
        state = wait_for_valid_state(hw, timeout=3.0)
        actual_joint_pos = np.array(state['pos'], dtype=np.float32)
        roll = state['roll']
        pitch = state['pitch']
        print(f"After stand-up: roll={np.rad2deg(roll):.1f}°, pitch={np.rad2deg(pitch):.1f}°")
        print(f"Joint pose (first 3 joints): {actual_joint_pos[:3]}")
        # Check tilt
        if abs(roll) > MAX_TILT_RAD or abs(pitch) > MAX_TILT_RAD:
            print("ERROR: Robot not level after stand-up! Aborting.")
            interpolate_to_pose(hw, SIT_POSE, steps=60, dt=0.02)
            sys.exit(1)
        # Compare joint positions to DEFAULT_JOINT_Q (optional warning)
        deviation = np.max(np.abs(actual_joint_pos - DEFAULT_JOINT_Q))
        if deviation > 0.2:
            print(f"WARNING: Joint deviation from default is {deviation:.2f} rad.")
    except TimeoutError:
        print("ERROR: Could not read state after stand-up. Aborting.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to pupper_weights.npz')
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--control_freq', type=float, default=50.0)
    parser.add_argument('--log_to_file', action='store_true')
    parser.add_argument('--alpha', type=float, default=ACTION_FILTER_ALPHA,
                        help='Action smoothing factor (0..1)')
    args = parser.parse_args()

    print("=== NeRD PPO Policy Deployment (FIXED) ===")
    print(f"Model: {args.model_path}")
    print(f"Max steps: {args.max_steps}")
    print(f"Control freq: {args.control_freq} Hz")
    print(f"Action filter alpha: {args.alpha}")
    print(f"Logging: {args.log_to_file}")

    net = load_policy(args.model_path)
    # Print expected observation size (first layer input dimension)
    expected_obs_dim = net['layers'][0][0].shape[1]
    print(f"Policy expects observation dim = {expected_obs_dim}")

    # ----- Stand up using env.reset() -----
    env = create_pupper_env(run_on_robot=True, render=False)
    hw = HardwareInterface('/dev/ttyACM0')
    time.sleep(2.0)

    stand_up_and_verify(hw, env)

    # ----- Ready for policy loop -----
    print("Standing stable. Starting policy now – KEEP HANDS CLEAR!")
    time.sleep(0.5)

    dt = 1.0 / args.control_freq
    prev_action = np.zeros(12, dtype=np.float32)
    last_commanded_targets = None   # will be set after first observation

    overruns = 0
    skipped = 0

    log_dict = {'t': [], 'MotorAngle': [], 'IMU': [], 'action': [], 'obs': [], 'raw_action': []} if args.log_to_file else None

    try:
        for step in range(args.max_steps):
            t0 = time.time()

            state = read_state(hw)
            if state is None:
                skipped += 1
                time.sleep(dt)
                continue

            # Build observation and run policy
            obs = build_obs(state, prev_action)
            # Check observation size matches policy input
            if obs.shape[0] != expected_obs_dim:
                print(f"ERROR: Obs dim {obs.shape[0]} != expected {expected_obs_dim}. Aborting.")
                break

            raw_action = run_policy(net, obs)
            raw_targets = action_to_targets(raw_action)

            # Apply smoothing (if last_commanded_targets exists)
            if last_commanded_targets is None:
                smoothed_targets = raw_targets
            else:
                smoothed_targets = args.alpha * raw_targets + (1.0 - args.alpha) * last_commanded_targets

            # Send command
            hw.set_actuator_postions(smoothed_targets.reshape(3, 4))
            last_commanded_targets = smoothed_targets.copy()
            prev_action = raw_action.copy()   # store raw action for next obs (or use smoothed? Use raw as in training)

            # Logging
            if args.log_to_file and log_dict:
                log_dict['t'].append(time.time())
                log_dict['MotorAngle'].append(np.array(state['pos']))
                log_dict['IMU'].append(np.array([state['roll'], state['pitch'], state['yaw'], state['yaw_rate']]))
                log_dict['action'].append(smoothed_targets.copy())
                log_dict['raw_action'].append(raw_action.copy())
                log_dict['obs'].append(obs.copy())

            # Timing
            elapsed = time.time() - t0
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                overruns += 1

            if step % 10 == 0:
                print(f"Step {step}  rpy=({state['roll']:.2f},{state['pitch']:.2f},{state['yaw']:.2f})  dt={elapsed*1e3:.1f}ms  overruns={overruns}  skipped={skipped}")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        print("Moving to sit pose...")
        interpolate_to_pose(hw, SIT_POSE, steps=60, dt=0.02)
        if args.log_to_file and log_dict:
            log_filename = f"nerd_log_fixed_{int(time.time())}.pkl"
            print(f"Saving log to {log_filename}")
            with open(log_filename, "wb") as f:
                pickle.dump(log_dict, f)
        print(f"Done. Overruns={overruns} Skipped={skipped}")
        env.close()

if __name__ == "__main__":
    main()
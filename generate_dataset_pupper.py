"""
Generate dataset for Stanford Pupper
"""
import sys, os
base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.append(base_dir)

import argparse
import h5py
import numpy as np
import torch

from generate.trajectory_sampler_pupper import TrajectorySamplerPupper
from utils.python_utils import set_random_seed
from envs.neural_environment import NeuralEnvironment
from utils.commons import (
    JOINT_Q_MIN, JOINT_Q_MAX, 
    JOINT_QD_MIN, JOINT_QD_MAX, 
    JOINT_ACT_SCALE
)

def collect_dataset(
    env_name, 
    num_envs, 
    initial_states_source,
    num_transitions, 
    trajectory_length, 
    dataset_path, 
    passive = False,
    seed = 0, 
    render = False,
    export_video = False,
    export_video_path = None
):
    data_writer = h5py.File(dataset_path, 'w')
    data_grp = data_writer.create_group('data')
    data_grp.attrs['env'] = env_name
    data_grp.attrs['mode'] = "trajectory"

    warp_env_cfg = {
        "seed": seed,
        "task": "dataset",

    }
    # Initialize the Environment
    # Initialize the Environment
    # Initialize the Environment
    env = NeuralEnvironment(
        env_name = env_name,
        num_envs = num_envs,
        neural_integrator_cfg = {}, # Keep this empty
        neural_model = None,
        default_env_mode = "ground-truth", # Ensure this is ground-truth
        warp_env_cfg = warp_env_cfg,
        render = render
    )
    robot_name = env.robot_name

    # --- 1. Prepare Padded Limits (Fixing the AssertionError) ---
    # Pupper in Warp has 19 Q DOFs (7 base + 12 joints) and 18 QD DOFs (6 base + 12 joints)
    j_q_min = JOINT_Q_MIN[robot_name]
    j_q_max = JOINT_Q_MAX[robot_name]
    j_qd_min = JOINT_QD_MIN[robot_name]
    j_qd_max = JOINT_QD_MAX[robot_name]

    # Pad Q: [pos(3), quat(4), joints(12)] = 19
    q_min_full = np.concatenate([
        np.array([-1.0, -1.0, 0.15]),       # x, y, z min height
        np.array([-1.0, -1.0, -1.0, -1.0]), # quat min
        j_q_min
    ])
    q_max_full = np.concatenate([
        np.array([1.0, 1.0, 0.45]),        # x, y, z max height
        np.array([1.0, 1.0, 1.0, 1.0]),    # quat max
        j_q_max
    ])

    # Pad QD: [lin_vel(3), ang_vel(3), joints(12)] = 18
    qd_min_full = np.concatenate([
        np.array([-1.0, -1.0, -1.0]),      # linear velocity min
        np.array([-3.14, -3.14, -3.14]),   # angular velocity min
        j_qd_min
    ])
    qd_max_full = np.concatenate([
        np.array([1.0, 1.0, 1.0]),         # linear velocity max
        np.array([3.14, 3.14, 3.14]),      # angular velocity max
        j_qd_max
    ])

    # --- 2. Initialize Sampler ---
    # --- 2. Initialize Sampler ---
    simulation_sampler = TrajectorySamplerPupper(
                            env,
                            joint_q_min = q_min_full,
                            joint_q_max = q_max_full,
                            joint_qd_min = qd_min_full,
                            joint_qd_max = qd_max_full,
                            joint_act_scale = 0.5 # Hardcode to 0.5
                        )
                        
    
    # --- 3. Run Rollouts ---
    rollouts = simulation_sampler.sample_trajectories_action_mode(
            num_transitions, 
            trajectory_length, 
            passive=passive,
            initial_states_source=initial_states_source,
            render=render,
            export_video=export_video,
            export_video_path=export_video_path,
        )

    # --- 4. Save to HDF5 ---
    data_grp.attrs['total_trajectories'] = rollouts['states'].shape[1]
    data_grp.attrs['total_transitions'] = rollouts['states'].shape[0] * rollouts['states'].shape[1]

    datasets_to_save = {
        'gravity_dir': rollouts['gravity_dir'],
        'root_body_q': rollouts['root_body_q'],
        'states': rollouts['states'],
        'contact_normals': rollouts['contacts']['contact_normals'],
        'contact_depths': rollouts['contacts']['contact_depths'],
        'contact_points_0': rollouts['contacts']['contact_points_0'],
        'contact_points_1': rollouts['contacts']['contact_points_1'],
        'contact_thicknesses': rollouts['contacts']['contact_thicknesses'],
        'joint_acts': rollouts['joint_acts'],
        'next_states': rollouts['next_states']
    }

    if 'actions' in rollouts:
        datasets_to_save['actions'] = rollouts['actions']

    for name, data in datasets_to_save.items():
        data_grp.create_dataset(name=name, data=data.detach().cpu().numpy())

    # Save Attributes
    data_grp.attrs['state_dim'] = rollouts['states'].shape[-1]
    data_grp.attrs['contact_prob'] = 0.0
    data_grp.attrs['num_contacts_per_env'] = rollouts['contacts']['contact_depths'].shape[-1]
    if 'actions' in rollouts:
        data_grp.attrs['action_dim'] = rollouts['actions'].shape[-1]
    data_grp.attrs['joint_act_dim'] = rollouts['joint_acts'].shape[-1]
    data_grp.attrs['next_state_dim'] = rollouts['next_states'].shape[-1]

    data_writer.flush()
    data_writer.close()
    print(f"Successfully saved dataset to {dataset_path}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, default='../../data/datasets/')
    parser.add_argument('--env-name', type=str, default='Pupper', choices=['Pupper'])
    parser.add_argument('--initial-states-source', type=str, default='env', choices = ['sample', 'env'])
    parser.add_argument('--num-transitions', type=int, default=1000000)
    parser.add_argument('--trajectory-length', type=int, default=100)
    parser.add_argument('--passive', action='store_true')
    parser.add_argument('--dataset-name', type=str, default='dataset.hdf5')
    parser.add_argument('--num-envs', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--export-video', action = 'store_true')
    parser.add_argument('--export-video-path', type = str, default = 'video.gif')
    
    args = parser.parse_args()
    set_random_seed(args.seed)

    dataset_path = os.path.join(args.dataset_dir, args.env_name, args.dataset_name)
    if os.path.exists(dataset_path):
        answer = input(f'Dataset exists in {dataset_path}, overwrite? [y/n]')
        if answer.lower() != 'y': exit()
    
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    collect_dataset(
        env_name=args.env_name, 
        num_envs=args.num_envs,
        initial_states_source=args.initial_states_source,
        num_transitions=args.num_transitions, 
        trajectory_length=args.trajectory_length,
        dataset_path=dataset_path,
        passive=args.passive,
        seed=args.seed,
        render=args.render,
        export_video=args.export_video,
        export_video_path=args.export_video_path
    )
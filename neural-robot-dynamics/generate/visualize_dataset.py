# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys, os
base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.append(base_dir)

import argparse
import h5py
import numpy as np
import shutil
import torch

from envs.warp_sim_envs import RenderMode
from envs.neural_environment import NeuralEnvironment
from utils import warp_utils
from utils.python_utils import print_warning

def visualize_dataset(dataset_path, num_envs, num_transitions, export_video=False, export_video_path='dataset_replay.gif', save_usd=False, save_usd_path='dataset_replay.usd', render_mode=None):
    dataset = h5py.File(dataset_path, 'r', swmr=True, libver='latest')
    env_name = dataset['data'].attrs['env']
    total_transitions = dataset['data'].attrs['total_transitions']
    print('total transitions = ', total_transitions)

    # create neural env
    warp_env_cfg = {}
    if render_mode is not None:
        warp_env_cfg["render_mode"] = render_mode
    elif save_usd or export_video:
        warp_env_cfg["render_mode"] = RenderMode.USD

    if export_video and warp_env_cfg.get("render_mode") != RenderMode.OPENGL:
        print_warning(
            'Video export requires pixel capture support. ' \
            'Running with current renderer may disable mp4/gif export. ' \
            'Use --render-mode opengl on a system with an available display or EGL support.'
        )

    env = NeuralEnvironment(
        env_name=env_name,
        num_envs=num_envs,
        warp_env_cfg=warp_env_cfg,
        neural_integrator_cfg={},
        neural_model=None,
        default_env_mode="ground-truth",
        render=True,
    )
    if export_video:
        env.start_video_export(export_video_path)

    # load datase
    states = dataset['data']['states'][()].astype('float32').transpose(1, 0, 2)
    states = states.reshape(-1, states.shape[-1])
    next_states = dataset['data']['next_states'][()].astype('float32').transpose(1, 0, 2)
    next_states = next_states.reshape(-1, next_states.shape[-1])

    # visualize
    # reshape states and next_states to be (num_envs, total_transitions / num_envs, dofs)
    total_transitions = total_transitions // num_envs * num_envs
    states = torch.tensor(
                states[:total_transitions, :].reshape(num_envs, -1, states.shape[-1]), 
                device = warp_utils.device_to_torch(env.device))
    next_states = torch.tensor(
                    next_states[:total_transitions, :].reshape(num_envs, -1, next_states.shape[-1]), 
                    device = warp_utils.device_to_torch(env.device))

    min_vel, max_vel = np.inf, -np.inf
    for step in range(min(num_transitions, states.shape[1])):
        env.reset(initial_states = states[:, step, :])
        env.render()
    if export_video:
        env.end_video_export()
    if save_usd:
        env.save_usd()
        usd_path = getattr(env.env.renderer, 'filename', None)
        if usd_path is None:
            usd_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), '..', 'envs', 'outputs', env.env.sim_name + '.usd'
            ))
        else:
            usd_path = os.path.abspath(usd_path)
        if os.path.abspath(save_usd_path) != usd_path:
            os.makedirs(os.path.dirname(save_usd_path), exist_ok=True)
            shutil.copy2(usd_path, save_usd_path)
            print(f"Copied USD to {save_usd_path}")
        else:
            print(f"USD saved to {usd_path}")
    print('Min vel = {}, Max vel = {}'.format(min_vel, max_vel))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', 
                        type=str,
                        required=True,
                        help='Path to the dataset.')
    parser.add_argument('--num-transitions',
                        type=int,
                        default=1000,
                        help='Number of transitions to be visualized.')
    parser.add_argument('--num-envs',
                        type=int,
                        default=10,
                        help='The number of parallel environments.')
    parser.add_argument('--render-mode',
                        type=str,
                        choices=['usd', 'opengl', 'none'],
                        default='usd',
                        help='Render mode to use for visualizing the dataset.')
    parser.add_argument('--export-video',
                        action='store_true',
                        help='Export a replay video from the rendered dataset.')
    parser.add_argument('--export-video-path',
                        type=str,
                        default='dataset_replay.gif',
                        help='Path to write the replay video.')
    parser.add_argument('--save-usd',
                        action='store_true',
                        help='Save a USD replay of the dataset.')
    parser.add_argument('--save-usd-path',
                        type=str,
                        default='dataset_replay.usd',
                        help='Path to write the USD file.')
    
    args = parser.parse_args()

    render_mode = RenderMode(args.render_mode) if args.render_mode is not None else None

    visualize_dataset(
        dataset_path = args.dataset_path,
        num_envs = args.num_envs,
        num_transitions = args.num_transitions,
        export_video = args.export_video,
        export_video_path = args.export_video_path,
        save_usd = args.save_usd,
        save_usd_path = args.save_usd_path,
        render_mode = render_mode
    )
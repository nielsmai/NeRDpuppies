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

import warp as wp
import numpy as np
from envs.warp_sim_envs.env_pupper import PupperEnvironment


@wp.kernel
def apply_pupper_joint_position_pd_control(
    actions: wp.array(dtype=wp.float32, ndim=1),
    action_scale: wp.float32,
    baseline_joint_q: wp.array(dtype=wp.float32),
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    Kp: wp.float32,
    Kd: wp.float32,
    joint_q_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_axis_dim: wp.array(dtype=wp.int32, ndim=2),
    joint_axis_start: wp.array(dtype=wp.int32),
    # outputs
    target_joint_q: wp.array(dtype=wp.float32),
    joint_act: wp.array(dtype=wp.float32)
):
    joint_id = wp.tid()

    ai = joint_axis_start[joint_id]
    qi = joint_q_start[joint_id]
    qdi = joint_qd_start[joint_id]
    dim = joint_axis_dim[joint_id, 0] + joint_axis_dim[joint_id, 1]

    for j in range(dim):
        qj = qi + j
        qdj = qdi + j
        aj = ai + j

        q = joint_q[qj]
        qd = joint_qd[qdj]

        # target position = default pose + scaled action offset
        tq = actions[aj] * action_scale + baseline_joint_q[qj]
        target_joint_q[aj] = tq

        # PD torque
        tq = Kp * (tq - q) - Kd * qd
        joint_act[aj] = tq


class PupperJointPositionControlEnvironment(PupperEnvironment):
    """
    Pupper environment with joint position PD control.
    
    The policy outputs target joint position offsets from the default pose.
    A PD controller converts these to torques that are passed to the simulator.
    
    This mirrors how real Pupper Dynamixel XL430 servos work — they are
    position-controlled, not torque-controlled.
    
    Kp and Kd are tuned for Dynamixel XL430 servos (much weaker than Anymal).
    Anymal uses Kp=85, Kd=2. Pupper uses much lower values.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Store default joint positions as the baseline for PD control
        # Policy outputs offsets around this pose
        self.default_joint_q = self.model.joint_q

        # Action scale: how large the position offsets can be (radians)
        # 0.3 rad ~ 17 degrees, reasonable for a walking gait
        self.action_scale = 0.3

        # PD gains tuned for Pupper Dynamixel XL430 servos
        # Stall torque ~1.5 Nm, so gains must be much lower than Anymal (85/2)
        # Kp: position gain (torque per radian of error)
        # Kd: velocity damping (torque per rad/s)
        self.Kp = 10.0
        self.Kd = 0.5

        # Buffer for target joint positions (for logging/debugging)
        self.target_joint_q = wp.empty(
            (self.num_envs * self.control_dim),
            dtype=wp.float32,
            device=self.device
        )

    def assign_control(
        self,
        actions: wp.array,
        control: wp.sim.Control,
        state: wp.sim.State
    ):
        if self.task == "dataset":
            # Randomize Kp and Kd during dataset generation for robustness
            # Range chosen to be physically plausible for Pupper servos
            self.Kp = float(np.random.uniform(low=5.0, high=30.0))
            self.Kd = float(np.random.uniform(low=0.0, high=1.0))

        wp.launch(
            kernel=apply_pupper_joint_position_pd_control,
            dim=self.model.joint_count,
            inputs=[
                wp.from_torch(wp.to_torch(actions).reshape(-1)),
                self.action_scale,
                self.default_joint_q,
                state.joint_q,
                state.joint_qd,
                self.Kp,
                self.Kd,
                self.model.joint_q_start,
                self.model.joint_qd_start,
                self.model.joint_axis_dim,
                self.model.joint_axis_start,
            ],
            outputs=[
                self.target_joint_q,
                control.joint_act
            ],
            device=self.model.device
        )
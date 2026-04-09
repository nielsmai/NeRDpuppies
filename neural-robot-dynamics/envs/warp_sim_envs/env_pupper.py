# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import math
import warp as wp
import warp.sim
import numpy as np

from envs.warp_sim_envs import Environment, IntegratorType, RenderMode

# --- PUPPER SPECIFIC CONSTANTS ---
PUPPER_URDF_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../urdf/standford_pupper_clean.urdf")
PUPPER_DEFAULT_HEIGHT = 0.3
PUPPER_NUM_CONTACTS = 4

@wp.kernel(enable_backward=False)
def reset_pupper_dataset(
    reset: wp.array(dtype=wp.bool),
    seed: int,
    random_reset: bool,
    dof_q_per_env: int,
    dof_qd_per_env: int,
    default_joint_q_init: wp.array(dtype=wp.float32),
    default_joint_qd_init: wp.array(dtype=wp.float32),
    # outputs
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()

    if reset[env_id]:
        for i in range(dof_q_per_env):
            joint_q[env_id * dof_q_per_env + i] = default_joint_q_init[env_id * dof_q_per_env + i]
        for i in range(dof_qd_per_env):
            joint_qd[env_id * dof_qd_per_env + i] = default_joint_qd_init[env_id * dof_qd_per_env + i]

        # Force safe base height and upright orientation
        joint_q[env_id * dof_q_per_env + 2] = 0.3
        joint_q[env_id * dof_q_per_env + 3] = 0.0
        joint_q[env_id * dof_q_per_env + 4] = 0.0
        joint_q[env_id * dof_q_per_env + 5] = 0.0
        joint_q[env_id * dof_q_per_env + 6] = 1.0

        if random_reset:
            random_state = wp.rand_init(seed, env_id)

            joint_q[env_id * dof_q_per_env + 0] += wp.randf(random_state, -0.05, 0.05)
            joint_q[env_id * dof_q_per_env + 1] += wp.randf(random_state, -0.05, 0.05)
            joint_q[env_id * dof_q_per_env + 2] += wp.randf(random_state, -0.05, 0.05)

            angle = wp.randf(random_state, -1.0, 1.0) * 0.1
            axis = wp.vec3(
                wp.randf(random_state, -1.0, 1.0),
                wp.randf(random_state, -1.0, 1.0),
                wp.randf(random_state, -1.0, 1.0),
            )
            axis = wp.normalize(axis)

            curr_quat = wp.quat(
                joint_q[env_id * dof_q_per_env + 3],
                joint_q[env_id * dof_q_per_env + 4],
                joint_q[env_id * dof_q_per_env + 5],
                joint_q[env_id * dof_q_per_env + 6],
            )
            delta_quat = wp.quat_from_axis_angle(axis, angle)
            new_quat = curr_quat * delta_quat

            for i in range(4):
                joint_q[env_id * dof_q_per_env + 3 + i] = new_quat[i]

            for i in range(12):
                joint_q[env_id * dof_q_per_env + 7 + i] += wp.randf(random_state, -0.05, 0.05)

            for i in range(6):
                joint_qd[env_id * dof_qd_per_env + i] = 0.1 * wp.randf(random_state, -1., 1.)
            for i in range(12):
                joint_qd[env_id * dof_qd_per_env + 7 + i] = 0.1 * wp.randf(random_state, -1., 1.)


class PupperEnvironment(Environment):
    robot_name = "Pupper"
    sim_name = "env_pupper"

    up_axis = 'z'
    gravity = (0.0, 0.0, -9.81)

    env_offset = (1.5, 0.0, 1.5)
    opengl_render_settings = dict(scaling=0.5)
    usd_render_settings = dict(scaling=0.5)

    sim_substeps_euler = 32
    sim_substeps_featherstone = 32 
    sim_substeps_xpbd = 8

    # Mirror Anymal exactly — proven working config in this codebase
    xpbd_settings = dict(iterations=2)
    joint_attach_ke: float = 100000.0
    joint_attach_kd: float = 10.0

    # Use Featherstone like Anymal — XPBD contact handling is broken in this framework
    integrator_type = IntegratorType.FEATHERSTONE

    # This is what makes Anymal work — separate ground contact handling
    separate_ground_contacts = True
    handle_collisions_once_per_step = True

    use_graph_capture = False
    num_envs = 1
    activate_ground_plane = True

    action_strength = 4.0
    controllable_dofs = np.arange(12)
    control_gains = np.array([400.0] * 12)
    control_limits = [(-1.0, 1.0)] * 12

    show_rigid_contact_points = True
    contact_points_radius = 0.02

    def __init__(
        self,
        seed=42,
        random_reset=True,
        task="dataset",
        obs_type="dflex",
        camera_tracking=False,
        render_mode=RenderMode.USD,
        **kwargs
    ):
        self.seed = seed
        self.random_reset = random_reset
        self.obs_type = obs_type
        self.camera_tracking = camera_tracking
        self.task = task

        if 'render_mode' not in kwargs:
            kwargs['render_mode'] = render_mode

        super().__init__(**kwargs)
        self.after_init()

    def create_articulation(self, builder):
        urdf_filename = PUPPER_URDF_PATH

        if not os.path.exists(urdf_filename):
            raise FileNotFoundError(f"Pupper URDF not found at {urdf_filename}")

        # Mirror Anymal's parse_urdf settings as closely as possible
        wp.sim.parse_urdf(
            urdf_filename,
            builder,
            floating=True,
            stiffness=10.0,       # same as Anymal
            damping=1.0,          # same as Anymal
            armature=0.06,        # same as Anymal
            contact_ke=5.0e3,     # same as Anymal
            contact_kd=1.0e3,     # same as Anymal
            contact_kf=1.0e2,
            contact_mu=0.75,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )

        # Pupper is Z-up, standing pose
        builder.joint_q[:7] = [
            0.0, 0.0, PUPPER_DEFAULT_HEIGHT,
            0.0, 0.0, 0.0, 1.0,  # identity quaternion — upright in Z-up
        ]

        builder.joint_q[7:] = [
            0.0,  0.8, -1.6,  # FL
            0.0,  0.8, -1.6,  # FR
            0.0,  0.8, -1.6,  # RL
            0.0,  0.8, -1.6,  # RR
        ]

        for i in range(builder.joint_axis_count):
            builder.joint_axis_mode[i] = wp.sim.JOINT_MODE_FORCE

        self.sim_time_wp = wp.zeros(1, dtype=wp.float32, device=self.device)
        self.torques = wp.zeros(
            self.num_envs * builder.joint_axis_count,
            dtype=wp.float32,
            device=self.device,
        )

        builder.separate_ground_contacts = self.separate_ground_contacts

    def after_init(self):
        # Read only — mirror Anymal's after_init exactly
        self.start_torso_pos = wp.array(
            self.model.joint_q.numpy().reshape(self.num_envs, -1)[:, 0:3].reshape(-1).copy()
        )
        self.start_rot = wp.quat(self.model.joint_q.numpy()[3:7])
        self.inv_start_rot = wp.quat_inverse(self.start_rot)
        self.basis_vec0 = wp.vec3(1., 0., 0.)
        self.basis_vec1 = wp.vec3(0., 0., 1.)

    def reset_envs(self, env_ids: wp.array = None):
        if env_ids is None:
            reset_mask = wp.ones(self.num_envs, dtype=wp.bool, device=self.device)
        else:
            reset_mask = env_ids

        wp.launch(
            reset_pupper_dataset,
            dim=self.num_envs,
            inputs=[
                reset_mask,
                self.seed,
                self.random_reset,
                self.dof_q_per_env,
                self.dof_qd_per_env,
                self.model.joint_q,
                self.model.joint_qd,
            ],
            outputs=[
                self.state.joint_q,
                self.state.joint_qd,
            ],
            device=self.device,
        )
        self.seed += self.num_envs

        # Mirror Anymal's reset — just FK, no manual collide() call needed
        # with separate_ground_contacts=True the framework handles it
        wp.sim.eval_fk(
            self.model,
            self.state.joint_q,
            self.state.joint_qd,
            None,
            self.state,
        )

    def compute_cost_termination(self, state, control, step, traj_length, cost, terminated):
        if not self.uses_generalized_coordinates:
            wp.sim.eval_ik(self.model, state, state.joint_q, state.joint_qd)
        pass

    @property
    def observation_dim(self):
        if self.obs_type == "dflex":
            return 37
        return self.dof_q_per_env + self.dof_qd_per_env

    def compute_observations(self, state, control, observations, step, horizon_length):
        if not self.uses_generalized_coordinates:
            wp.sim.eval_ik(self.model, state, state.joint_q, state.joint_qd)

        from envs.warp_sim_envs.env_anymal import compute_observations_anymal_dflex

        wp.launch(
            compute_observations_anymal_dflex,
            dim=self.num_envs,
            inputs=[
                state.joint_q,
                state.joint_qd,
                self.basis_vec0,
                self.basis_vec1,
                self.dof_q_per_env,
                self.dof_qd_per_env,
            ],
            outputs=[observations],
            device=self.device,
        )

    def assign_control(self, actions, control, state):
        super().assign_control(actions, control, state)
        self.raw_joint_act = wp.from_torch(wp.to_torch(control.joint_act).clone())
        self.apply_pd_control(
            control_out=control.joint_act,
            joint_q=state.joint_q,
            joint_qd=state.joint_qd,
            body_q=state.body_q,
        )

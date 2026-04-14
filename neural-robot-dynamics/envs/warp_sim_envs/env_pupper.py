# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import math
import warp as wp
import warp.sim
import numpy as np

from envs.warp_sim_envs import Environment, IntegratorType, RenderMode

# --- PUPPER SPECIFIC CONSTANTS ---
PUPPER_URDF_PATH = "/teamspace/studios/this_studio/NeRDpuppies/urdf/standford_pupper_clean.urdf"
PUPPER_DEFAULT_HEIGHT = 0.23
PUPPER_NUM_CONTACTS = 4

@wp.kernel
def pupperv2_forward_cost(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    contact_depths: wp.array(dtype=wp.float32),
    joint_act: wp.array(dtype=wp.float32),
    dof_q: int,
    dof_qd: int,
    num_contacts: int,
    # outputs
    cost: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.bool)
):
    env_id = wp.tid()

    # Z-up: position is (x, y, z), z is height
    torso_pos = wp.vec3(
        joint_q[dof_q * env_id + 0],
        joint_q[dof_q * env_id + 1],
        joint_q[dof_q * env_id + 2]
    )

    # In Warp's Featherstone, spatial velocity is [ang_vel(0,1,2), lin_vel(3,4,5)]
    # Z-up: lin_vel[0]=x-fwd, lin_vel[1]=y-lateral, lin_vel[2]=z-up
    lin_vel = wp.vec3(
        joint_qd[dof_qd * env_id + 3],
        joint_qd[dof_qd * env_id + 4],
        joint_qd[dof_qd * env_id + 5]
    )
    ang_vel = wp.vec3(
        joint_qd[dof_qd * env_id + 0],
        joint_qd[dof_qd * env_id + 1],
        joint_qd[dof_qd * env_id + 2]
    )

    # Convert twist to CoM velocity
    lin_vel = lin_vel - wp.cross(torso_pos, ang_vel)

    # Forward = +X, lateral = Y, yaw = Z axis rotation
    target_x_vel  = 0.5   # m/s forward — Pupper is small, 0.5 is a brisk walk
    target_y_vel  = 0.0   # no lateral drift
    target_yaw_vel = 0.0  # no turning

    # lin_vel[0] = forward (X), lin_vel[1] = lateral (Y)
    lin_vel_error = (lin_vel[0] - target_x_vel) ** 2. + (lin_vel[1] - target_y_vel) ** 2.
    # ang_vel[2] = yaw (Z-up)
    ang_vel_error = (ang_vel[2] - target_yaw_vel) ** 2.

    rew_lin_coef    = 1.0
    rew_ang_coef    = 0.5
    rew_torque_coef = 2.5e-5

    rew_lin_vel = wp.exp(-lin_vel_error) * rew_lin_coef
    rew_ang_vel = wp.exp(-ang_vel_error) * rew_ang_coef

    # 12 actuated joints: 3 per leg (abduction, upper, lower) x 4 legs
    # order from URDF: leftFront(abd,upper,lower), leftRear(abd,upper,lower),
    #                  rightFront(abd,upper,lower), rightRear(abd,upper,lower)
    rew_torque = 0.0
    for i in range(12):
        rew_torque -= (joint_act[env_id * 12 + i]) ** 2.0 * rew_torque_coef

    c = -rew_lin_vel - rew_ang_vel - rew_torque
    if c > 0.:
        c = 0.
    wp.atomic_add(cost, env_id, c)

    if terminated:
        # Pupper only has 4 toe contacts (PUPPER_NUM_CONTACTS = 4)
        # indices 0-3: leftFrontToe, leftRearToe, rightFrontToe, rightRearToe
        # Toe sphere radius = 0.0095m — negative depth means penetration/collision
        # No knee contacts unlike ANYmal — toes are the only contact bodies

        # Height termination — standing z=0.3, collapse below ~half
        if torso_pos[2] < 0.15:
            terminated[env_id] = True

        # Roll/pitch termination via upright check
        # quaternion is at joint_q[3:7] — extract z-component of up vector
        # up vector in world = R * [0,0,1], z component = 1 - 2*(qx^2 + qy^2)
        qx = joint_q[dof_q * env_id + 3]
        qy = joint_q[dof_q * env_id + 4]
        upright_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        if upright_z < 0.5:   # >60 deg tilt
            terminated[env_id] = True

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
    random_state = wp.rand_init(seed, env_id)
    if reset[env_id]:
        for i in range(dof_q_per_env):
            joint_q[env_id * dof_q_per_env + i] = default_joint_q_init[env_id * dof_q_per_env + i]
        for i in range(12):
            joint_qd[env_id * dof_qd_per_env + 6 + i] = 0.1 * wp.randf(random_state, -1., 1.)
        # Force safe base height and upright orientation
        joint_q[env_id * dof_q_per_env + 2] = 0.3
        joint_q[env_id * dof_q_per_env + 3] = 0.0
        joint_q[env_id * dof_q_per_env + 4] = 0.0
        joint_q[env_id * dof_q_per_env + 5] = 0.0
        joint_q[env_id * dof_q_per_env + 6] = 1.0

        
        if random_reset:
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

    action_strength = 1.0
    controllable_dofs = np.arange(12)
    control_gains = np.array([5.0] * 12)
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
            collapse_fixed_joints=False,
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
            
        num_contacts = self.num_rigid_contacts_per_env if self.num_rigid_contacts_per_env is not None else 0
        if self.task == "forward":
            wp.launch(
                pupperv2_forward_cost,
                dim=self.num_envs,
                inputs=[
                    state.joint_q, 
                    state.joint_qd,
                    self.model.rigid_contact_depth,   # from AbstractContactEnvironment
                    control.joint_act,
                    self.dof_q_per_env,
                    self.dof_qd_per_env,
                    num_contacts,
                ],
                outputs=[cost, terminated],
                device=self.device,
            )
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

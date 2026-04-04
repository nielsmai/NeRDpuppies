# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import math
import warp as wp
import warp.sim
import numpy as np

from envs.warp_sim_envs import Environment, IntegratorType, RenderMode

# --- PUPPER SPECIFIC CONSTANTS ---
PUPPER_URDF_PATH = "/teamspace/studios/this_studio/urdf/standford_pupper_clean.urdf"
PUPPER_DEFAULT_HEIGHT = 0.55 # Adjust based on your URDF leg length
PUPPER_NUM_CONTACTS = 9  # 4 Feet

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
        # 1. Copy defaults first
        for i in range(dof_q_per_env):
            joint_q[env_id * dof_q_per_env + i] = default_joint_q_init[env_id * dof_q_per_env + i]
        for i in range(dof_qd_per_env):
            joint_qd[env_id * dof_qd_per_env + i] = default_joint_qd_init[env_id * dof_qd_per_env + i]

        # ---------------------------------------------------------
        # CRITICAL FIX: FORCE BASE HEIGHT AND ORIENTATION
        # Do this BEFORE randomization to ensure a safe baseline
        # ---------------------------------------------------------
        
        # Force Base Z (Index 2) to 0.50 (Safe height)
        joint_q[env_id * dof_q_per_env + 2] = 0.45
        
        # Force Upright Quaternion (Indices 3,4,5,6)
        joint_q[env_id * dof_q_per_env + 3] = 0.0
        joint_q[env_id * dof_q_per_env + 4] = 0.0
        joint_q[env_id * dof_q_per_env + 5] = 0.0
        joint_q[env_id * dof_q_per_env + 6] = 1.0

        if random_reset:
            random_state = wp.rand_init(seed, env_id)

            # Add SMALL perturbation to the SAFE baseline
            # X/Y noise ±0.05m
            joint_q[env_id * dof_q_per_env + 0] += wp.randf(random_state, -0.05, 0.05)
            joint_q[env_id * dof_q_per_env + 1] += wp.randf(random_state, -0.05, 0.05)
            
            # Z noise ±0.05m (Range will be 0.45 to 0.55)
            joint_q[env_id * dof_q_per_env + 2] += wp.randf(random_state, -0.05, 0.05)
            
            # Orientation noise (±0.1 rad)
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

            # Small joint noise
            for i in range(12):
                joint_q[env_id * dof_q_per_env + 7 + i] += wp.randf(random_state, -0.05, 0.05)

            # Tiny velocity noise
            for i in range(6): 
                joint_qd[env_id * dof_qd_per_env + i] = 0.1 * wp.randf(random_state, -1., 1.)
            for i in range(12): 
                joint_qd[env_id * dof_qd_per_env + 7 + i] = 0.1 * wp.randf(random_state, -1., 1.)
class PupperEnvironment(Environment):
    robot_name = "Pupper"
    sim_name = "env_pupper"

      # --- CRITICAL FIX: Match your stable script ---
    up_axis = 'z'       # Force Z-Up (Was likely 'y' by default)
    gravity = (0.0, 0.0, -9.81) # Gravity points down -Z
    
    # Fix Visibility: Smaller offset and scaling for a small robot
    env_offset = (1.5, 0.0, 1.5) 
    opengl_render_settings = dict(scaling=0.5) # Zoom in closer
    usd_render_settings = dict(scaling=0.5)    # Zoom in closer

    # Fix Stability: High iterations for small contacts
    sim_substeps_euler = 32
    sim_substeps_featherstone = 10
    sim_substeps_xpbd = 60  # Increased steps
    handle_collisions_once_per_step = False 
    xpbd_settings = dict(iterations=100) # High iterations to prevent sinking/exploding

    joint_attach_ke: float = 1000.0
    joint_attach_kd: float = 100.0
    integrator_type = IntegratorType.FEATHERSTONE
    #integrator_type = IntegratorType.XPBD

    separate_ground_contacts = True
    use_graph_capture = False
    num_envs = 1
    activate_ground_plane = True

    # Fix Explosion: Moderate Gains (Match your working script's stiffness roughly)
    # Your script had ke=1000. We use 200-400 here for RL stability.
    action_strength = 4.0 
    controllable_dofs = np.arange(12)
    
    # Uniform gains for simplicity. If unstable, lower to 100.0
    control_gains = np.array([10.0] * 12) 
    #control_gains = np.array([200.0] * 12) 
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
        render_mode=RenderMode.USD, # Default to USD for headless
        **kwargs
    ):
        self.seed = seed
        self.random_reset = random_reset
        self.obs_type = obs_type
        self.camera_tracking = camera_tracking
        self.task = task
        # Force render_mode if not passed
        if 'render_mode' not in kwargs:
            kwargs['render_mode'] = render_mode
            
        super().__init__(**kwargs)
        self.after_init()

    def create_articulation(self, builder):
        urdf_filename = PUPPER_URDF_PATH
            
        if not os.path.exists(urdf_filename):
            raise FileNotFoundError(f"Pupper URDF not found at {urdf_filename}")
        builder.default_shape_margin = 0.03
        print(f"DEBUG: Set builder.default_shape_margin to {builder.default_shape_margin}")
        
        # Parse URDF
        # CRITICAL: Do NOT collapse fixed joints if it hides legs. 
        # But usually collapse_fixed_joints=True is fine for performance.
        wp.sim.parse_urdf(
            urdf_filename,
            builder,
            floating=True,
            stiffness=10.0, # Default PD stiffness matching control_gains
            damping=1.0,     # Damping to prevent jitter
            armature=0.1,   
            contact_ke=2.0e3, # Stiffer contacts to prevent sinking
            contact_kd=1.0e3,
            contact_kf=1.0e2,
            contact_mu=0.9, 
            limit_ke=1.0e4,
            limit_kd=1.0e2,
            enable_self_collisions=False, 
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False
        )
        print(f"DEBUG: After parse_urdf, builder.default_shape_margin is {builder.default_shape_margin}")
        
        
        # FIX ORIENTATION: Set Initial Pose Explicitly Upright
        # Base: x, y, z, qx, qy, qz, qw
        # Quaternion (0,0,0,1) is Upright (No rotation)
        builder.joint_q[:7] = [
            0.0,
            0.0,
            PUPPER_DEFAULT_HEIGHT,
            0.0, 0.0, 0.0, 1.0, # IDENTITY QUATERNION (Upright)
        ]

        # Default Joint Angles (Standing Pose)
        # Ensure these match your URDF's "home" position
        builder.joint_q[7:] = [
            0.0,  0.8, -1.6,  # FL
            0.0,  0.8, -1.6,  # FR
            0.0,  0.8, -1.6,  # RL
            0.0,  0.8, -1.6   # RR
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
        self.start_torso_pos = wp.array(
            self.model.joint_q.numpy().reshape(self.num_envs, -1)[:, 0:3].reshape(-1).copy()
        )
        self.start_rot = wp.quat(self.model.joint_q.numpy()[3:7])
        self.inv_start_rot = wp.quat_inverse(self.start_rot)
        self.basis_vec0 = wp.vec3(1., 0., 0.)
        self.basis_vec1 = wp.vec3(0., 0., 1.)
    
    def reset_envs(self, env_ids: wp.array = None):
        """Reset using Pupper-specific kernel (Upright)."""
        
        # Create mask
        if env_ids is None:
            reset_mask = wp.ones(self.num_envs, dtype=wp.bool, device=self.device)
        else:
            reset_mask = env_ids

        # 1. Launch Reset Kernel (Sets joint_q)
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
        
        # ---------------------------------------------------------
        # STEP 3 FIX: WARM START THE PHYSICS
        # ---------------------------------------------------------
        
        # A. Update Body Positions from Joint Angles (FK)
        wp.sim.eval_fk(
            self.model, 
            self.state.joint_q, 
            self.state.joint_qd, 
            None, 
            self.state
        )
        
        # B. Clear Forces
        self.state.clear_forces()
        
        # C. Run Collision Detection IMMEDIATELY
        # This pushes feet out of the ground BEFORE the first step
        wp.sim.collide(self.model, self.state)
        
        # D. Update FK again to reflect collision corrections
        wp.sim.eval_fk(
            self.model, 
            self.state.joint_q, 
            self.state.joint_qd, 
            None, 
            self.state
        )
        # ---------------------------------------------------------

        self.seed += self.num_envs

        # Final sync (redundant but safe)
        # wp.sim.eval_fk(...) # Already done above

    def compute_cost_termination(self, state, control, step, traj_length, cost, terminated):
        # Simple termination: Fall over or explode
        if not self.uses_generalized_coordinates:
            wp.sim.eval_ik(self.model, state, state.joint_q, state.joint_qd)
        
        # Terminate if NaN or too low
        # (Logic handled by sampler mostly, but good to have here)
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
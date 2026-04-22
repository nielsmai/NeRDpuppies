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

"""
Passive motion evaluation script with debug instrumentation and W&B logging.

Key behaviours being debugged:
  - Frozen / broken joints in first half of USD playback
  - Environment being pushed around then reset in second half

Hypotheses being probed:
  (H1) Model is not being invoked (passive=True suppresses all action -> env starts/stays at zero)
  (H2) Initial state is degenerate (zero / invalid quaternion / NaN)
  (H3) A mid-rollout env reset is triggered (joint-limit violation, NaN state, or a manual reset)
  (H4) State divergence is gradual - the neural integrator accumulates error until the physics blows up
"""

import sys, os
from typing import Union, Optional  # Add this for compatibility
base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(base_dir)

import argparse
import torch
import yaml
import numpy as np

from envs.warp_sim_envs import RenderMode
from envs.neural_environment import NeuralEnvironment
from utils.torch_utils import num_params_torch_model
from utils.python_utils import set_random_seed
from utils import torch_utils
from utils.evaluator import NeuralSimEvaluator

# ---------------------------------------------------------------------------
# Optional W&B – gracefully skip if not installed or --no-wandb is passed
# ---------------------------------------------------------------------------
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[DEBUG] wandb not installed – console-only logging.")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _tensor_summary(t: torch.Tensor, name: str) -> dict:
    """Return a flat dict of min/max/mean/std/nan_count for a tensor."""
    t_f = t.float()
    nan_count = torch.isnan(t_f).sum().item()
    inf_count = torch.isinf(t_f).sum().item()
    if nan_count == t_f.numel():
        return {f"{name}/all_nan": True}
    valid = t_f[~torch.isnan(t_f) & ~torch.isinf(t_f)]
    return {
        f"{name}/min":       valid.min().item()  if valid.numel() else float('nan'),
        f"{name}/max":       valid.max().item()  if valid.numel() else float('nan'),
        f"{name}/mean":      valid.mean().item() if valid.numel() else float('nan'),
        f"{name}/std":       valid.std().item()  if valid.numel() else float('nan'),
        f"{name}/nan_count": nan_count,
        f"{name}/inf_count": inf_count,
    }


def _check_quat_validity(quat: torch.Tensor, name: str = "quat") -> dict:
    """
    Quaternions should have unit norm (within tolerance).
    Returns norm stats + count of invalid quaternions.
    """
    norms = quat.float().norm(dim=-1)  # (...,)
    invalid = ((norms < 0.99) | (norms > 1.01)).sum().item()
    return {
        f"{name}/norm_mean":    norms.mean().item(),
        f"{name}/norm_min":     norms.min().item(),
        f"{name}/norm_max":     norms.max().item(),
        f"{name}/invalid_count": invalid,
    }


def _log(metrics: dict, step: Optional[int] = None, use_wandb: bool = False, prefix: str = ""):
    """Print metrics to console (always) and optionally to W&B."""
    if prefix:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
    print(f"  [step={step}]" if step is not None else "  [summary]", end="  ")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}={v:.6f}", end="  ")
        else:
            print(f"{k}={v}", end="  ")
    print()
    if use_wandb and WANDB_AVAILABLE:
        wandb.log(metrics, step=step)


# ---------------------------------------------------------------------------
# Post-evaluation trajectory analyser
# ---------------------------------------------------------------------------

def analyse_trajectories(
    trajectories: dict,
    next_states_diff: torch.Tensor,
    base_position_idx: list,
    base_orientation_idx: list,
    joint_idx: list,
    use_wandb: bool,
    rollout_horizon: int,
):
    """
    Deep-dive analysis of rollout_states to surface the frozen/pushed behaviour.

    Expects trajectories to contain at minimum:
        'rollout_states'  : Tensor[T, num_envs, state_dim]   (T = rollout_horizon + 1)

    Optionally also inspects:
        'target_states'   : same shape
        'rollout_actions' : Tensor[T-1, num_envs, action_dim]
    """
    rollout_states = trajectories.get('rollout_states')   # [T, N, S]
    target_states  = trajectories.get('target_states')    # [T, N, S] or None
    rollout_actions= trajectories.get('rollout_actions')  # [T-1, N, A] or None

    if rollout_states is None:
        print("[DEBUG] trajectories dict has no 'rollout_states' key – skipping analysis.")
        print(f"[DEBUG] Available keys: {list(trajectories.keys())}")
        return

    T, N, S = rollout_states.shape
    print(f"\n{'='*60}")
    print(f"[DEBUG] Trajectory shape: T={T}, N={N}, S={S}")
    print(f"[DEBUG] rollout_horizon arg = {rollout_horizon}  (expect T = horizon + 1 = {rollout_horizon+1})")
    print(f"{'='*60}")

    # -----------------------------------------------------------------------
    # (H3) Detect mid-rollout resets: a sudden discontinuous jump in base pos
    # -----------------------------------------------------------------------
    print("\n--- (H3) Mid-rollout reset detection ---")
    if len(base_position_idx) > 0:
        base_pos = rollout_states[:, :, base_position_idx].float()   # [T, N, 3]
        pos_delta = (base_pos[1:] - base_pos[:-1]).norm(dim=-1)      # [T-1, N]
        JUMP_THRESHOLD = 0.5  # metres – tune if needed
        jumps = (pos_delta > JUMP_THRESHOLD)
        if jumps.any():
            jump_steps, jump_envs = jumps.nonzero(as_tuple=True)
            print(f"  [!] RESET DETECTED – {jumps.sum().item()} sudden jumps > {JUMP_THRESHOLD}m")
            for s, e in zip(jump_steps[:10].tolist(), jump_envs[:10].tolist()):
                print(f"       step {s} -> {s+1}, env {e}: delta = {pos_delta[s, e]:.4f} m")
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({"reset_detection/num_jumps": jumps.sum().item(),
                           "reset_detection/first_jump_step": jump_steps[0].item()})
        else:
            print(f"  No resets detected (all position deltas < {JUMP_THRESHOLD}m)")

        # Also log per-step position norm to spot the "frozen then moving" pattern
        pos_norm_per_step = base_pos.norm(dim=-1).mean(dim=-1)  # [T]
        print(f"  Base position norm (mean over envs) at each step:")
        half = T // 2
        print(f"    First half  (steps 0..{half-1}):  "
              f"mean={pos_norm_per_step[:half].mean():.4f}  "
              f"std={pos_norm_per_step[:half].std():.4f}")
        print(f"    Second half (steps {half}..{T-1}): "
              f"mean={pos_norm_per_step[half:].mean():.4f}  "
              f"std={pos_norm_per_step[half:].std():.4f}")

        if use_wandb and WANDB_AVAILABLE:
            for t_idx, val in enumerate(pos_norm_per_step.tolist()):
                wandb.log({"trajectory/base_pos_norm": val}, step=t_idx)

    # -----------------------------------------------------------------------
    # (H1) Are actions zero / constant throughout?
    # -----------------------------------------------------------------------
    print("\n--- (H1) Action statistics ---")
    if rollout_actions is not None:
        act = rollout_actions.float()   # [T-1, N, A]
        print(f"  Action tensor shape: {tuple(act.shape)}")
        zero_steps = (act.abs().sum(dim=-1) < 1e-8).sum().item()  # steps where action is exactly 0
        print(f"  Steps with all-zero actions: {zero_steps} / {act.shape[0] * act.shape[1]}")
        print(f"  Action abs mean per step:   "
              f"first half={act[:act.shape[0]//2].abs().mean():.6f}  "
              f"second half={act[act.shape[0]//2:].abs().mean():.6f}")
        stats = _tensor_summary(act, "actions")
        _log(stats, use_wandb=use_wandb)
    else:
        print("  [!] 'rollout_actions' not present in trajectories dict.")
        print(f"       Available keys: {list(trajectories.keys())}")
        print("       Consider saving actions in your evaluator for debugging.")

    # -----------------------------------------------------------------------
    # (H2) Initial state validity
    # -----------------------------------------------------------------------
    print("\n--- (H2) Initial state validity ---")
    init_state = rollout_states[0]  # [N, S]
    print(f"  Initial state (env 0): {init_state[0].cpu().numpy()}")
    if len(base_orientation_idx) == 4:
        init_quat = init_state[:, base_orientation_idx]
        quat_stats = _check_quat_validity(init_quat, "init_quat")
        _log(quat_stats, use_wandb=use_wandb)
    nan_in_init = torch.isnan(init_state).sum().item()
    print(f"  NaN values in initial state: {nan_in_init}")
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"init_state/nan_count": nan_in_init})

    # -----------------------------------------------------------------------
    # (H4) State divergence over time (per-step error growth)
    # -----------------------------------------------------------------------
    print("\n--- (H4) State divergence (rollout vs target) ---")
    if target_states is not None:
        diff = (rollout_states.float() - target_states.float())  # [T, N, S]
        per_step_rmse = diff.pow(2).mean(dim=(-1, -2)).sqrt()    # [T]
        half = T // 2
        print(f"  Per-step RMSE summary:")
        print(f"    First half  mean={per_step_rmse[:half].mean():.6f}  "
              f"max={per_step_rmse[:half].max():.6f}")
        print(f"    Second half mean={per_step_rmse[half:].mean():.6f}  "
              f"max={per_step_rmse[half:].max():.6f}")
        if use_wandb and WANDB_AVAILABLE:
            for t_idx, val in enumerate(per_step_rmse.tolist()):
                wandb.log({"trajectory/state_rmse": val}, step=t_idx)
    else:
        print("  [!] 'target_states' not in trajectories – skipping divergence analysis.")

    # -----------------------------------------------------------------------
    # Per-step joint position stats (frozen joints show near-zero std)
    # -----------------------------------------------------------------------
    print("\n--- Joint motion analysis (frozen joints have std ≈ 0) ---")
    if len(joint_idx) > 0:
        joints = rollout_states[:, :, joint_idx].float()  # [T, N, J]
        J = joints.shape[-1]
        half = T // 2
        print(f"  {'DOF':<6} {'1st-half mean':>16} {'1st-half std':>14} {'2nd-half mean':>16} {'2nd-half std':>14}")
        dof_stats = {}
        for j in range(J):
            h1 = joints[:half, :, j];  h2 = joints[half:, :, j]
            m1, s1 = h1.mean().item(), h1.std().item()
            m2, s2 = h2.mean().item(), h2.std().item()
            print(f"  {j:<6} {m1:>16.6f} {s1:>14.6f} {m2:>16.6f} {s2:>14.6f}")
            dof_stats[f"joint_dof{j}/h1_mean"] = m1
            dof_stats[f"joint_dof{j}/h1_std"]  = s1
            dof_stats[f"joint_dof{j}/h2_mean"] = m2
            dof_stats[f"joint_dof{j}/h2_std"]  = s2
        if use_wandb and WANDB_AVAILABLE:
            wandb.log(dof_stats)

    # -----------------------------------------------------------------------
    # next_states_diff summary (already computed by evaluator)
    # -----------------------------------------------------------------------
    print("\n--- next_states_diff statistics ---")
    diff_stats = _tensor_summary(next_states_diff, "next_states_diff")
    _log(diff_stats, use_wandb=use_wandb)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--env-name',
                        default='Pendulum', type=str)
    parser.add_argument('--model-path',
                        default=None, type=str)
    parser.add_argument('--dataset-path',
                        default=None, type=str)
    parser.add_argument('--env-mode',
                        default='neural', type=str,
                        choices=['neural', 'ground-truth'])
    parser.add_argument('--num-envs',
                        default=1, type=int)
    parser.add_argument('--num-rollouts',
                        default=100, type=int)
    parser.add_argument('--rollout-horizon',
                        default=10, type=int)
    parser.add_argument('--seed',
                        default=0, type=int)
    parser.add_argument('--render',
                        action='store_true')
    parser.add_argument('--export-video',
                        action='store_true')
    parser.add_argument('--export-video-path',
                        type=str, default='video.gif')
    parser.add_argument('--export-usd',
                        action='store_true')
    parser.add_argument('--usd-output-path',
                        type=str, default=None,
                        help='Custom USD save path. Auto-generates a timestamped name if None or if file exists.')
    # --- new debug flags ---
    parser.add_argument('--debug',
                        action='store_true',
                        help='Enable detailed console debug output')
    parser.add_argument('--wandb',
                        action='store_true',
                        dest='use_wandb',
                        help='Log debug statistics to Weights & Biases')
    parser.add_argument('--wandb-project',
                        default='nerd-passive-motion-debug', type=str)
    parser.add_argument('--wandb-run-name',
                        default=None, type=str)
    parser.add_argument('--no-wandb',
                        action='store_true',
                        help='Explicitly disable W&B even if installed')

    args = parser.parse_args()

    use_wandb = args.use_wandb and WANDB_AVAILABLE and not args.no_wandb

    # -----------------------------------------------------------------------
    # W&B initialisation
    # -----------------------------------------------------------------------
    if use_wandb:
        run_name = args.wandb_run_name or f"{args.env_name}_{args.env_mode}_seed{args.seed}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )
        print(f"[DEBUG] W&B run initialised: {wandb.run.url}")
    elif args.use_wandb and not WANDB_AVAILABLE:
        print("[DEBUG] --wandb requested but wandb is not installed. "
              "Run `pip install wandb` to enable.")

    # -----------------------------------------------------------------------
    device = 'cuda:0'
    set_random_seed(args.seed)

    env_cfg = {
        "env_name": args.env_name,
        "num_envs": args.num_envs,
        "render": args.render,
        "warp_env_cfg": {"seed": args.seed},
        "default_env_mode": args.env_mode,
    }

    if args.export_usd:
        args.render = True
        env_cfg["render"] = True
        env_cfg["warp_env_cfg"]["render_mode"] = RenderMode.USD

    # -----------------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------------
    if args.model_path is not None:
        print(f"[DEBUG] Loading model from: {args.model_path}")
        model, robot_name = torch.load(args.model_path, map_location='cuda:0')
        n_params = num_params_torch_model(model)
        print(f"[DEBUG] Model type       : {type(model).__name__}")
        print(f"[DEBUG] Robot name (ckpt): {robot_name}")
        print(f"[DEBUG] Num parameters   : {n_params:,}")
        model.to(device)

        train_dir = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(args.model_path)), '../')
        )
        cfg_path = os.path.join(train_dir, 'cfg.yaml')
        print(f"[DEBUG] Loading config from: {cfg_path}")
        with open(cfg_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
        env_cfg["neural_integrator_cfg"] = cfg["env"]["neural_integrator_cfg"]

        if args.debug or use_wandb:
            ni_cfg = cfg["env"]["neural_integrator_cfg"]
            print(f"[DEBUG] neural_integrator_cfg: {ni_cfg}")
            if use_wandb:
                wandb.config.update({"neural_integrator_cfg": ni_cfg})

        # Sanity-check model weights for NaN/Inf
        if args.debug:
            print("\n[DEBUG] --- Model weight health check ---")
            bad_params = []
            for name, param in model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    bad_params.append(name)
            if bad_params:
                print(f"  [!] NaN/Inf found in parameters: {bad_params}")
                if use_wandb:
                    wandb.log({"model/corrupt_param_count": len(bad_params)})
            else:
                print("  All model parameters are finite.")
    else:
        model = None
        print("[DEBUG] No model path provided – running ground-truth physics only.")

    # -----------------------------------------------------------------------
    # Env construction
    # -----------------------------------------------------------------------
    print(f"\n[DEBUG] Constructing NeuralEnvironment...")
    print(f"[DEBUG] env_cfg = {env_cfg}")

    neural_env = NeuralEnvironment(neural_model=model, **env_cfg)

    if model is not None:
        assert neural_env.robot_name == robot_name, \
            f"Robot name mismatch: env={neural_env.robot_name}, ckpt={robot_name}"
        print(f"[DEBUG] Robot name check passed: {robot_name}")

    # -----------------------------------------------------------------------
    # Evaluator
    # -----------------------------------------------------------------------
    evaluator = NeuralSimEvaluator(
        neural_env,
        args.dataset_path,
        args.rollout_horizon,
        device=device,
    )

    set_random_seed(args.seed)

    print(f"\n[DEBUG] Starting evaluation:")
    print(f"  env_mode      = {args.env_mode}")
    print(f"  passive       = True")
    print(f"  num_rollouts  = {args.num_rollouts}")
    print(f"  horizon       = {args.rollout_horizon}")
    print(f"  num_envs      = {args.num_envs}")

    next_states_diff, trajectories, _ = evaluator.evaluate_action_mode(
        num_traj=args.num_rollouts,
        trajectory_source="sampler",
        eval_mode="rollout",
        env_mode=args.env_mode,
        passive=True,
        render=args.render,
        export_video=args.export_video,
        export_video_path=args.export_video_path,
    )

    print(f"\n[DEBUG] Evaluation complete.")
    print(f"[DEBUG] next_states_diff shape: {next_states_diff.shape}")
    print(f"[DEBUG] trajectories keys     : {list(trajectories.keys())}")

    # -----------------------------------------------------------------------
    # DOF index mapping (same as original)
    # -----------------------------------------------------------------------
    if args.env_name == 'Cartpole':
        base_position_idx    = [0]
        base_orientation_idx = []
        joint_idx            = [1]
    elif args.env_name == 'PendulumWithContact':
        base_position_idx    = []
        base_orientation_idx = [0]
        joint_idx            = [1]
    elif args.env_name == 'Ant':
        base_position_idx    = [0, 1, 2]
        base_orientation_idx = [3, 4, 5, 6]
        joint_idx            = [7, 8, 9, 10, 11, 12, 13, 14]
    elif 'Anymal' in args.env_name:
        base_position_idx    = [0, 1, 2]
        base_orientation_idx = [3, 4, 5, 6]
        joint_idx            = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    elif args.env_name == 'Pupper':
        base_position_idx    = [0, 1, 2]
        base_orientation_idx = [3, 4, 5, 6]
        joint_idx            = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    elif args.env_name == 'CubeToss':
        base_position_idx    = [0, 1, 2]
        base_orientation_idx = [3, 4, 5, 6]
        joint_idx            = []
    else:
        raise NotImplementedError

    # -----------------------------------------------------------------------
    # Debug analysis
    # -----------------------------------------------------------------------
    if args.debug or use_wandb:
        analyse_trajectories(
            trajectories=trajectories,
            next_states_diff=next_states_diff,
            base_position_idx=base_position_idx,
            base_orientation_idx=base_orientation_idx,
            joint_idx=joint_idx,
            use_wandb=use_wandb,
            rollout_horizon=args.rollout_horizon,
        )

    # -----------------------------------------------------------------------
    # Original error reporting (unchanged)
    # -----------------------------------------------------------------------
    print('=========================================')

    if len(base_position_idx) > 0:
        base_position_error = next_states_diff[..., base_position_idx].norm(dim=-1).mean()
        base_position_error_std = next_states_diff[..., base_position_idx].norm(dim=-1).std()
    else:
        base_position_error = None

    if len(base_orientation_idx) > 0:
        if len(base_orientation_idx) == 1:
            base_orientation_error     = next_states_diff[..., base_orientation_idx].abs().mean()
            base_orientation_error_std = next_states_diff[..., base_orientation_idx].abs().std()
        else:
            quat_rollout = trajectories['rollout_states'][1:, :, base_orientation_idx]
            quat_target  = quat_rollout + next_states_diff[..., base_orientation_idx]
            quat_rollout = quat_rollout.view(-1, 4)
            quat_target  = quat_target.view(-1, 4)
            quat_angle_diff = torch_utils.quat_angle_diff(quat_rollout, quat_target)
            base_orientation_error     = quat_angle_diff.mean()
            base_orientation_error_std = quat_angle_diff.std()
    else:
        base_orientation_error = None

    final_metrics = {}

    if base_position_error is not None:
        print("{:<30} = {:.6f}".format("Base position error mean", base_position_error.cpu().item()))
        print("{:<30} = {:.6f}".format("Base position error std",  base_position_error_std.cpu().item()))
        final_metrics["eval/base_position_error_mean"] = base_position_error.cpu().item()
        final_metrics["eval/base_position_error_std"]  = base_position_error_std.cpu().item()

    if base_orientation_error is not None:
        print("{:<30} = {:.6f} rad ({:.6f} deg)".format(
            "Base orientaion error mean",
            base_orientation_error.cpu().item(),
            np.rad2deg(base_orientation_error.cpu().item())
        ))
        print("{:<30} = {:.6f} rad ({:.6f} deg)".format(
            "Base orientation error std",
            base_orientation_error_std.cpu().item(),
            np.rad2deg(base_orientation_error_std.cpu().item())
        ))
        final_metrics["eval/base_orientation_error_mean_rad"] = base_orientation_error.cpu().item()
        final_metrics["eval/base_orientation_error_std_rad"]  = base_orientation_error_std.cpu().item()

    if len(joint_idx) > 0:
        joint_pos_error = next_states_diff[..., joint_idx].abs().mean()
        print('{:<30} = {:.6f} rad ({:.6f} deg)'.format(
            "Joint position error mean",
            joint_pos_error.cpu().item(),
            np.rad2deg(joint_pos_error.cpu().item())
        ))
        final_metrics["eval/joint_position_error_mean_rad"] = joint_pos_error.cpu().item()

    print("{:<30} = {}".format(
        "Joint position Error per dof",
        next_states_diff[..., joint_idx].abs().mean((0, 1))
    ))

    per_dof_errors = next_states_diff[..., joint_idx].abs().mean((0, 1))
    for j, err in enumerate(per_dof_errors.tolist()):
        final_metrics[f"eval/joint_dof{j}_error_mean_rad"] = err

    if use_wandb and WANDB_AVAILABLE:
        wandb.log(final_metrics)
        wandb.finish()
        print("[DEBUG] W&B run finished.")

    print('=========================================')

    if args.export_usd:
        import os
        import datetime

        # 1. Determine output path
        if args.usd_output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            args.usd_output_path = f"env_{neural_env.robot_name}_{timestamp}.usd"
        else:
            # Ensure .usd extension
            if not args.usd_output_path.endswith('.usd'):
                args.usd_output_path += '.usd'
            # Prevent overwrite: append timestamp if file already exists
            if os.path.exists(args.usd_output_path):
                base, ext = os.path.splitext(args.usd_output_path)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                args.usd_output_path = f"{base}_{timestamp}{ext}"

        # 2. Ensure parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(args.usd_output_path)) or '.', exist_ok=True)

        print(f"[DEBUG] Saving USD to: {args.usd_output_path}")
        # 3. Save with explicit path
        # Note: If your NeuralEnvironment.save_usd() uses a different kwarg name 
        # (e.g., 'filename' or 'output_path'), update 'path=' accordingly.
        neural_env.save_usd(path=args.usd_output_path)
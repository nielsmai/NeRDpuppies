import sys
sys.path.append('.')
import warp as wp
import warp.sim as wp_sim
import warp.sim.render
import torch
import numpy as np
import os

wp.init()

print("=== Pupper V2: Framework Video Generation ===")

try:
    from envs.neural_environment import NeuralEnvironment
    from envs.warp_sim_envs import RenderMode

    # 1. Initialize the Framework Environment
    print("\n[1] Initializing NeuralEnvironment...")
    env = NeuralEnvironment(
        env_name='Pupper',
        num_envs=1,
        neural_integrator_cfg={},
        neural_model=None,
        default_env_mode='ground-truth',
        warp_env_cfg={
            "seed": 42,
            "task": "dataset",
            "render_mode": RenderMode.NONE,
        },
        render=False,
        device='cuda:0'
    )
    print("✅ Env Initialized.")

    # Verify contact settings came through correctly (READ ONLY — no writes)
    print(f"\n🔍 Contact settings verification:")
    print(f"   rigid_contact_max:    {env.env.model.rigid_contact_max}")
    print(f"   rigid_contact_margin: {env.env.model.rigid_contact_margin}")

    # 2. Reset
    print("\n[2] Resetting environment...")
    env.reset()
    wp.sim.eval_fk(
        env.env.model,
        env.env.state.joint_q,
        env.env.state.joint_qd,
        None,
        env.env.state
    )
    start_z = env.env.state.body_q.numpy()[0, 2]
    print(f"   Starting Base Z: {start_z:.3f}m")

    # 3. Attach Renderer
    print("\n[3] Attaching Manual Renderer...")
    output_file = "outputs/pupper_framework_demo.usd"
    os.makedirs("outputs", exist_ok=True)
    renderer = wp_sim.render.SimRenderer(
        env.env.model,
        output_file,
        fps=60,
    )
    print(f"   Renderer ready for: {output_file}")

    # 4. Simulation Loop
    total_steps = 100
    dt = 1.0 / 120.0
    print(f"\n[4] Running {total_steps} steps...")

    for i in range(total_steps):
        action = torch.zeros((1, env.action_dim), device=env.torch_device)
        next_state = env.step(action)

        renderer.begin_frame(i * dt)
        renderer.render(env.env.state)
        renderer.end_frame()

        if i % 20 == 0:
            curr_q = env.env.state.body_q.numpy()[0]
            print(f"   Step {i:3d} | Base Pos: [{curr_q[0]:.3f}, {curr_q[1]:.3f}, {curr_q[2]:.3f}]")

    # 5. Save
    print("\n[5] Saving Video...")
    renderer.save()
    print(f"✅ Saved to: {output_file}")

    env.close()

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

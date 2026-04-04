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
            "render_mode": RenderMode.NONE, # Disable internal renderer to avoid conflicts
        },
        render=False,
        device='cuda:0'
    )
    print("✅ Env Initialized.")
        # ... after env initialization ...
    
    print("\n🔍 INSPECTING MODEL GEOMETRY...")
    model = env.env.model
    
    print(f"Total Bodies: {model.body_count}")
    print(f"Total Shapes: {model.shape_count}")
    
    # Check if Ground Flag is active
    has_ground_flag = getattr(model, 'ground', False)
    print(f"Model 'ground' flag: {has_ground_flag}")
    
    # Iterate through all shapes to find the ground
    found_ground = False
    for i in range(model.shape_count):
        # Get shape type (0=Sphere, 1=Box, 2=Capsule, 3=Plane, etc. - varies by Warp version)
        # In Warp, we often check the body index. Ground is usually Body 0 or a specific static body.
        
        # Try to get shape body mapping
        shape_body = model.shape_body.numpy()[i] if hasattr(model, 'shape_body') else -1
        
        # Check if this shape is a plane (Type 3 in many engines, or infinite box)
        # We can also check the scale. Ground planes often have huge scales.
        if hasattr(model, 'shape_scale'):
            scale = model.shape_scale.numpy()[i]
            if np.any(scale > 10.0): # Ground is usually huge
                print(f"   [Shape {i}] Found HUGE object (Likely Ground): Scale={scale}, Body={shape_body}")
                found_ground = True
                
        # Print first few shapes for debugging
        if i < 5:
            print(f"   [Shape {i}] Scale: {model.shape_scale.numpy()[i] if hasattr(model, 'shape_scale') else 'N/A'}, Body: {shape_body}")

    if not found_ground and not has_ground_flag:
        print("   ❌ CRITICAL: No ground plane detected in model!")
    else:
        print("   ✅ Ground plane detected.")

    # 2. Force Reset & Sync (Crucial!)
    print("\n[2] Forcing Reset & Sync...")
    env.reset()
    wp.sim.eval_fk(
        env.env.model, 
        env.env.state.joint_q, 
        env.env.state.joint_qd, 
        None, 
        env.env.state
    )
    
    # Verify start
    start_z = env.env.state.body_q.numpy()[0, 2]
    print(f"   Starting Base Z: {start_z:.3f}m")

    # 3. Attach Standalone Renderer (From your working script)
    print("\n[3] Attaching Manual Renderer...")
    output_file = "outputs/pupper_framework_demo.usd"
    os.makedirs("outputs", exist_ok=True)
    
    # Create renderer directly from the environment's model
    renderer = wp_sim.render.SimRenderer(
        env.env.model, 
        output_file, 
        fps=60,
        #up_vector='z' # Ensure renderer knows Z is up
    )
    print(f"   Renderer ready for: {output_file}")

    # 4. Simulation Loop
    total_steps = 100
    dt = 1.0 / 120.0 # Match your stable script's timestep
    
    print(f"\n[4] Running {total_steps} steps...")
    
    # Enable small random actions to see movement
    use_actions = True 

    for i in range(total_steps):
        # --- Generate Action ---
        if use_actions:
            # Small noise: range [-0.1, 0.1]
            action = torch.rand((1, env.action_dim), device=env.torch_device) * 0.2 - 0.1
        else:
            action = torch.zeros((1, env.action_dim), device=env.torch_device)
            
        # --- Step the Framework ---
        next_state = env.step(action)
        
        # --- Render Frame ---
        # We render the CURRENT state (env.env.state)
        renderer.begin_frame(i * dt)
        renderer.render(env.env.state)
        renderer.end_frame()
        
        # Log progress
        if i % 20 == 0:
            curr_q = env.env.state.body_q.numpy()[0]
            print(f"   Step {i:3d} | Base Pos: [{curr_q[0]:.3f}, {curr_q[1]:.3f}, {curr_q[2]:.3f}]")

    # 5. Save
    print("\n[5] Saving Video...")
    renderer.save()
    print(f"Successfully saved to: {output_file}")

    print("\n" + "="*50)
    print("VIEWING INSTRUCTIONS:")
    print("="*50)
    print(f"1. Open '{output_file}' in NVIDIA Omniverse Viewer or Blender.")
    print("2. Press 'F' to frame the robot immediately.")
    print("3. You should see the Pupper walking/wiggling upright!")
    print("="*50)

    env.close()

except Exception as e:
    print(f"\n ERROR: {e}")
    import traceback
    traceback.print_exc()

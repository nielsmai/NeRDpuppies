#!/usr/bin/env python3
"""
Test cleaned URDF in Warp simulation.
"""

import warp as wp
import warp.sim
import numpy as np
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================
URDF_PATH = "/teamspace/studios/this_studio/urdf/stanford_pupper_warp.urdf"
DEVICE = "cuda"  # Use "cpu" if GPU not available

# =============================================================================
# MAIN
# =============================================================================
def main():
    print(f"Testing URDF in Warp ({DEVICE})...")
    print("=" * 70)
    
    # Initialize Warp
    wp.init()
    
    # Build model
    builder = warp.sim.ModelBuilder()
    
    try:
        wp.sim.parse_urdf(
            URDF_PATH,
            builder,
            xform=wp.transform([0.0, 0.0, 0.40], wp.quat_identity()),
            floating=True,
            contact_ke=1e4,      # Stiff contacts (GPU can handle)
            contact_kd=1e3,
            contact_kf=1e3,
            contact_mu=0.8,
            collapse_fixed_joints=True,
            enable_self_collisions=False,
        )
    except Exception as e:
        print(f"❌ Failed to parse URDF: {e}")
        return False
    
    model = builder.finalize(DEVICE)
    model.ground = True
    model.gravity = wp.vec3(0.0, 0.0, -9.81)
    
    print(f"✅ Model built: {model.body_q.shape[0]} bodies, {len(model.joint_name)} joints")
    
    # Initialize state
    state = model.state()
    wp.sim.eval_fk(model, state.joint_q, state.joint_qd, None, state)
    wp.sim.collide(model, state)
    
    # Quick stability test
    integrator = warp.sim.XPBDIntegrator(iterations=8)
    dt = 1.0 / 120.0
    
    print("\nRunning 100-step stability test...")
    for step in range(100):
        wp.sim.collide(model, state)
        integrator.simulate(model, state, state, dt=dt)
        
        bz = state.body_q.numpy()[0][2]
        if np.isnan(bz):
            print(f"❌ NaN at step {step}")
            return False
        
        if step % 20 == 0:
            print(f"  Step {step:3d}: base_z={bz:.4f}")
    
    print("\n✅ 100 steps stable — URDF is Warp-compatible!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
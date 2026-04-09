import warp as wp
import warp.sim as wp_sim
import warp.sim.render

wp.init()
device = "cuda" if wp.is_cuda_available() else "cpu"

# 1. Setup Model
builder = wp_sim.ModelBuilder()
# ... (your parse_urdf code from before) ...
wp_sim.parse_urdf("urdf/standford_pupper.urdf", builder, floating=True)

model = builder.finalize(device)
model.ground = True
state = model.state()

# 2. Initialize the Integrator (This replaces wp_sim.simulate)
# XPBD is the "Gold Standard" for stable robotics in Warp 1.8
integrator = wp_sim.XPBDIntegrator(iterations=10)

# 3. Setup Renderer
renderer = wp_sim.render.SimRenderer(model, "output.usd", fps=60)

# 4. Simulation Loop
for i in range(100):
    state.clear_forces()
    
    # Use the integrator instance to step the simulation
    integrator.simulate(model, state, stapwfte, dt=1.0/60.0)
    
    renderer.begin_frame(i / 60.0)
    renderer.render(state)
    renderer.end_frame()

renderer.save()

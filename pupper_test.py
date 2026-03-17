import warp as wp
import warp.sim as wp_sim
import warp.sim.render
import numpy as np

# 0. initialize warp simulator
wp.init()
device = "cuda" if wp.is_cuda_available() else "cpu"

# 1. Setup Model
#builder = wp_sim.ModelBuilder()
builder = wp_sim.ModelBuilder(up_vector=(0.0, 0.0, 1.0))        #change the up axis in Warp, to match the URDF one
builder.default_shape_ke = 1e4          #contact stiffness
builder.default_shape_kd = 250.0        #contact damping, 
builder.default_shape_kf = 500.0        #fiction stiffness
builder.default_shape_margin = 0.01     #collision margin (1cm, how close objects can get before collision detected)

wp_sim.parse_urdf("/teamspace/studios/this_studio/puppersim/puppersim/puppersim/data/pupper_v2a.urdf", builder, floating=True) #load urdf into builder
model = builder.finalize(device)    #uploads model to gpu, final
model.ground = False               #enable ground plane

# 2. Setup State 
state_in = model.state()            #state inputs buffer
state_out = model.state()           #tate outputs buffer

q = state_in.body_q.numpy()  
q[0] = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0] #sets the base link pose
state_in.body_q.assign(q)                #assign position to warp tensor on GPU

# 3. Initialize the Integrator
integrator = wp_sim.SemiImplicitIntegrator()             #create a stable physics solver

# 4. Setup Renderer
renderer = wp_sim.render.SimRenderer(model, "outputs/o4.usd", fps=60)  #to visualize

# 5. Simulation Loop
state_in.body_qd.zero_()        #set velocities to 0 to start

# let the robot fall and settle before recording
for i in range(200):
    wp.sim.collide(model, state_in)
    integrator.simulate(model, state_in, state_out, dt=1.0/2000.0)
    state_in, state_out = state_out, state_in
    q = state_in.body_q.numpy()
    print(f"step {i:3d} | body z={q[0][2]:.4f} | RF_TOE z={q[13][2]:.4f}")

# now run the real loop
for i in range(100):
    state_in.clear_forces()
    wp.sim.collide(model, state_in)
    integrator.simulate(model, state_in, state_out, dt=1.0/60.0)
    state_in, state_out = state_out, state_in
    q = state_in.body_q.numpy()
    renderer.begin_frame(i / 60.0)
    renderer.render(state_in)
    renderer.end_frame()

renderer.save()
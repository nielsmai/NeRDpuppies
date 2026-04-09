import numpy as np
import xml.etree.ElementTree as ET

urdf_path = "/teamspace/studios/this_studio/urdf/standford_pupper_clean.urdf"

try:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
except FileNotFoundError:
    print(f"ERROR: Could not find {urdf_path}")
    print("Please check the path.")
    exit()

joint_limits_min = []
joint_limits_max = []
joint_vel_limits = []

# Standard Pupper Joint Order (Verify this matches your URDF joint names)
# If your URDF has different names, this script will try to find all 12 continuous/revolute joints
expected_joints = 12
found_joints = []

print(f"Parsing {urdf_path}...\n")

for joint in root.findall('.//joint'):
    j_type = joint.get('type')
    if j_type in ['revolute', 'continuous']:
        name = joint.get('name')
        limit = joint.find('limit')
        
        if limit is not None:
            lower = float(limit.get('lower', -3.14))
            upper = float(limit.get('upper', 3.14))
            velocity = float(limit.get('velocity', 10.0))
            
            found_joints.append(name)
            joint_limits_min.append(lower)
            joint_limits_max.append(upper)
            joint_vel_limits.append(velocity)

if len(found_joints) != expected_joints:
    print(f"WARNING: Found {len(found_joints)} joints, expected {expected_joints}.")
    print(f"Joints found: {found_joints}")
    print("Check if your URDF has fixed joints or if the order is different.\n")
else:
    print(f"Successfully found {len(found_joints)} joints.")


print("\n--- COPY THE BELOW INTO utils/commons.py ---\n")

print("# Pupper V2 Limits extracted from URDF")
print(f"# Joints order: {found_joints}\n")

print("JOINT_Q_MIN['Pupper'] = np.array([")
for i in range(0, len(joint_limits_min), 3):
    chunk = joint_limits_min[i:i+3]
    comment = ""
    if i == 0: comment = "  # FL"
    elif i == 3: comment = "  # FR"
    elif i == 6: comment = "  # RL"
    elif i == 9: comment = "  # RR"
    print(f"    {chunk[0]:.4f}, {chunk[1]:.4f}, {chunk[2]:.4f},{comment}")
print("], dtype=np.float32)\n")

print("JOINT_Q_MAX['Pupper'] = np.array([")
for i in range(0, len(joint_limits_max), 3):
    chunk = joint_limits_max[i:i+3]
    comment = ""
    if i == 0: comment = "  # FL"
    elif i == 3: comment = "  # FR"
    elif i == 6: comment = "  # RL"
    elif i == 9: comment = "  # RR"
    print(f"    {chunk[0]:.4f}, {chunk[1]:.4f}, {chunk[2]:.4f},{comment}")
print("], dtype=np.float32)\n")

print("# Velocity limits (Rad/s)")
max_vel = max(joint_vel_limits)
# We usually set a safe global limit slightly higher than the max in URDF or uniform
print(f"JOINT_QD_MIN['Pupper'] = -{max_vel:.2f} * np.ones(12, dtype=np.float32)")
print(f"JOINT_QD_MAX['Pupper'] = {max_vel:.2f} * np.ones(12, dtype=np.float32)\n")

print("# Action Scale (Torque)")
print("# Estimate: Pupper servos are typically 2-5 Nm. Setting to 4.0 as safe default.")
print("# If you know your specific servo stall torque, change the 4.0 below.")
print("JOINT_ACT_SCALE['Pupper'] = np.array([")
for i in range(0, 12, 3):
    print(f"    4.0, 4.0, 4.0,  # Leg {i//3 + 1}")
print("], dtype=np.float32)")
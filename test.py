# =========================
# TEST UAV ENV WITH VISUALIZATION
# =========================

from env import UAVHMARLEnv
import numpy as np
import matplotlib.pyplot as plt
import time

# ✅ Enable interactive plotting
plt.ion()

# =========================
# CREATE ENV
# =========================
env = UAVHMARLEnv(num_uavs=5)

# =========================
# RESET ENV
# =========================
obs, _ = env.reset()

print("Starting simulation...")

# =========================
# RUN SIMULATION
# =========================
for step in range(200):

    # ✅ Random actions (for testing only)
    actions = {
        a: np.random.uniform(-1, 1, 3)
        for a in env.agents
    }

    # Step environment
    obs, rewards, terms, truncs, infos = env.step(actions)

    # Render UAVs
    env.render()

    # Small delay for smooth animation
    time.sleep(0.05)

    # Stop if episode ends
    if any(terms.values()):
        print(f"Episode finished at step {step}")
        break

# =========================
# KEEP WINDOW OPEN
# =========================
plt.ioff()
plt.show()
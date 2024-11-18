from market_env import MarketEnv
import numpy as np

# Initialize the environment
env = MarketEnv(config_name='default')
observation, info = env.reset()

print("Initial Wallet Balances:", env.wallet.balances)
print("Starting Simulation...")

# Number of steps to simulate
num_steps = 300

for step in range(num_steps):
    print(f"\nStep {step + 1}")

    # For testing, let's alternate between buying and selling
    actions = []
    actions = env.action_space.sample()

    actions = np.array(actions)
    observation, reward, done, info = env.step(actions)
    
    if done:
        break

print("\nSimulation Finished.")

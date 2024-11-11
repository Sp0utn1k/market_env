# test_script.py

from market_env import MarketEnv
import numpy as np

# Initialize the environment
env = MarketEnv(config_name='default')
observation, info = env.reset()

print("Initial Wallet Balances:", env.wallet.balances)
print("Starting Simulation...")

# Number of steps to simulate
num_steps = 5

for step in range(num_steps):
    print(f"\nStep {step + 1}")

    # For testing, let's alternate between buying and selling
    actions = []
    for idx, instrument in enumerate(env.instruments):
        base_currency = instrument.base_currency
        quote_currency = instrument.quote_currency
        base_balance = env.wallet.balances.get(base_currency, 0)
        quote_balance = env.wallet.balances.get(quote_currency, 0)

        # Alternate actions
        if step % 2 == 0:
            # Buy quote currency with 10% of base currency
            action_value = 0.1 if base_balance > 0 else 0.0
        else:
            # Sell quote currency to get base currency
            action_value = -0.1 if quote_balance > 0 else 0.0

        actions.append(action_value)

    actions = np.array(actions)
    observation, reward, done, info = env.step(actions)
    env.render()
    print("Wallet Balances:", env.wallet.balances)
    print("Reward:", reward)
    if done:
        break

print("\nSimulation Finished.")

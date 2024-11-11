"""
Market Environment Module.

Defines the MarketEnv class compatible with Gymnasium for RL agent training.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import pandas as pd
from typing import List
from .wallet import Wallet
from .instrument import Instrument
from .data_loader import load_multiple_instruments, synchronize_data, load_instrument_data
from .utils import compute_exchange_rates, normalize_actions, parse_instrument_name
from .config import Config


class MarketEnv(gym.Env):
    """Market Simulation Environment for RL Agents."""

    metadata = {'render.modes': ['human']}

    def __init__(self, config_name: str = 'default'):
        """
        Initializes the MarketEnv environment.

        Args:
            config_name (str): Name of the configuration file.
        """
        super(MarketEnv, self).__init__()
        self.config = Config(config_name).config['environment']
        self._initialize_environment()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.instruments),), dtype=np.float32)
        obs_space_shape = (len(self.instruments), self.window_size, 7)  # 7 data columns
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_space_shape, dtype=np.float32)

    def _initialize_environment(self):
        """Initializes environment components."""
        self.window_size = self.config['window_size']
        self.currencies = self.config['currencies']
        self.base_currency = self.config['base_currency']
        self.starting_balance = self.config['starting_balance']
        self.fees = self.config['fees']

        # Initialize Wallet
        initial_balances = {curr: 0.0 for curr in self.currencies}
        initial_balances[self.base_currency] = self.starting_balance
        self.wallet = Wallet(initial_balances, self.fees)

        # Load and initialize instruments
        instrument_names = self._select_instruments()
        datasets = load_multiple_instruments(instrument_names, interval=self.config['interval'])
        datasets = synchronize_data(datasets)
        self.instruments = []
        print("Loaded Instruments:")
        for name in instrument_names:
            data = datasets[name]
            instrument = Instrument(name, data)
            self.instruments.append(instrument)
            quote_currency = instrument.quote_currency
            base_currency = instrument.base_currency
            start_time = pd.to_datetime(data['time'].iloc[0], unit='s')
            end_time = pd.to_datetime(data['time'].iloc[-1], unit='s')
            print(f"Instrument: {name}, Base: {base_currency}, Quote: {quote_currency}, "
                  f"Start: {start_time}, End: {end_time}")
        self.data_length = len(self.instruments[0].data)
        self.current_step = self.window_size - 1

    def _select_instruments(self) -> List[str]:
        """
        Selects instruments based on available data and configuration.

        Returns:
            List[str]: List of instrument names to be used.
        """
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        instrument_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        available_instruments = set()
        ignore_instruments = set(self.config.get('ignore_instruments', []))

        for file_name in instrument_files:
            # Extract instrument name and interval
            parts = file_name.split('_')
            if len(parts) != 2:
                continue  # Unexpected file name format
            instrument_name, file_interval = parts
            file_interval = file_interval.replace('.csv', '')
            if int(file_interval) != self.config['interval']:
                continue  # Skip if interval doesn't match

            # Parse quote and base currencies
            quote_currency, base_currency = parse_instrument_name(instrument_name)
            if base_currency in self.currencies and quote_currency in self.currencies:
                if instrument_name not in ignore_instruments:
                    available_instruments.add(instrument_name)

        if not available_instruments:
            raise ValueError("No instruments found that match the currencies and interval specified.")

        return list(available_instruments)

    def reset(self):
        """Resets the environment to the initial state."""
        self.wallet.reset()
        self.current_step = self.window_size - 1
        self.previous_value = self.starting_balance
        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        """
        Executes one time step within the environment.

        Args:
            action (np.ndarray): Action provided by the agent.

        Returns:
            observation (np.ndarray): Next observation.
            reward (float): Reward from the action.
            done (bool): Whether the episode has ended.
            info (dict): Additional information.
        """
        action = normalize_actions(action)
        self._execute_actions(action)
        self.current_step += 1
        observation = self._get_observation()
        reward = self._compute_reward()
        done = self.current_step >= self.data_length - 1
        info = {}
        return observation, reward, done, info

    def _execute_actions(self, actions: List[float]):
        """Executes the agent's actions."""
        rates = {instr.base_currency: {} for instr in self.instruments}
        for idx, (action_value, instrument) in enumerate(zip(actions, self.instruments)):
            rate = instrument.get_rate(self.current_step)
            from_currency = instrument.base_currency if action_value >= 0 else instrument.quote_currency
            to_currency = instrument.quote_currency if action_value >= 0 else instrument.base_currency
            amount = abs(action_value) * self.wallet.balances.get(from_currency, 0)
            self.wallet.execute_transaction(from_currency, to_currency, amount, rate)

    def _get_observation(self) -> np.ndarray:
        """Retrieves the current observation."""
        observations = []
        for instrument in self.instruments:
            data_window = instrument.get_history(self.current_step, self.window_size)
            observations.append(data_window.values)
        return np.array(observations)

    def _compute_reward(self) -> float:
        """Computes the reward for the current step."""
        exchange_rates = {}
        for instrument in self.instruments:
            rate = instrument.get_rate(self.current_step)
            exchange_rates[instrument.quote_currency] = rate
            exchange_rates[instrument.base_currency] = 1 / rate
        total_value = self.wallet.get_total_value(exchange_rates, self.base_currency)
        reward = total_value - self.previous_value
        self.previous_value = total_value
        return reward

    def render(self, mode='human'):
        """Renders the environment state."""
        print(f"Step: {self.current_step}")
        print(f"Wallet Balances: {self.wallet.balances}")

    def close(self):
        """Cleans up the environment."""
        pass
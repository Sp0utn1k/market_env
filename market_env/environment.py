"""
Market Environment Module.

Defines the MarketEnv class compatible with Gymnasium for RL agent training.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import pandas as pd
from typing import List, Dict
import random

from .wallet import Wallet
from .instrument import Instrument
from .data_loader import load_multiple_instruments, synchronize_data, load_instrument_data
from .utils import compute_exchange_rates, parse_instrument_name
from .config import Config


class MarketEnv(gym.Env):
    """Market Simulation Environment for RL Agents."""

    metadata = {'render.modes': ['human']}

    def __init__(self, config_name: str = 'default', external_config_dir: str = None):
        """
        Initializes the MarketEnv environment.

        Args:
            config_name (str): Name of the configuration file.
        """
        super(MarketEnv, self).__init__()
        self.config = Config(external_config_dir, config_name= config_name).config['environment']

        self._init_componemts()
        self._init_data_format()
        self._init_wallet()
        self._init_instruments()
        self.print_synchronized_period()
        self._init_usable_instruments()
        self._init_action_space()
        self._init_observation_space()

    def _init_observation_space(self):
        sequence = self.obs_format['sequence']
        constant = self.obs_format['constant']
        shape_seq = (len(self.instruments), self.window_size, len(sequence['numerical']))
        shape_const = (len(constant['numerical']), )
        observation_space_dict = {
            'sequence_numerical': spaces.Box(low= -np.inf, high=np.inf, shape=shape_seq, dtype=np.float32),
            'constant_numerical': spaces.Box(low= -np.inf, high=np.inf, shape=shape_const, dtype=np.float32),
        }

        n_seq = []
        for name in sequence['categorical']:
            n_name = get_n_states(name)
            n_seq.append(n_name)

        n_const = []
        for name in sequence['categorical']:
            n_name = get_n_states(name)
            n_const.append(n_name)

        observation_space_dict['sequence_categorical'] = spaces.MultiDiscrete([n_seq]*self.window_size, dtype=np.int16)
        observation_space_dict['constant_categorical'] = spaces.MultiDiscrete(n_const, dtype=np.int16)

        self.observation_space = spaces.Dict(observation_space_dict)    
        

    def _init_componemts(self):
        self.window_size = self.config['window_size']
        self.currencies = self.config['currencies']
        self.base_currency = self.config['base_currency']
        self.starting_balance = self.config['starting_balance']
        self.fees = self.config['fees']
        self.episode_length = self.config['episode_length']
        self.discretize = self.config.get('discretize', False)
        self.interval = self.config.get('interval', 1)

    def _init_data_format(self):
        self.obs_format = self.config['data']
        test_sample = self.obs_format.pop('test_sample', False)
        self.data_fields = []
        for d in self.obs_format.values():
            for fields in d.values():
                self.data_fields += fields

        self.nrows = 0
        if test_sample:
            self.nrows = 2*max(self.window_size, self.episode_length)



    def _init_wallet(self):
        initial_balances = {curr: 0.0 for curr in self.currencies}
        initial_balances[self.base_currency] = self.starting_balance
        self.wallet = Wallet(initial_balances, self.fees)

    def _init_instruments(self):


        instrument_names = self._select_instruments()
        datasets = load_multiple_instruments(instrument_names, self.data_fields, nrows=self.nrows)
        self.print_loaded_instruments(datasets)
        datasets = synchronize_data(datasets)
        print("Data synchronized.")
        self.instruments = []
        for name in instrument_names:
            data = datasets[name]
            instrument = Instrument(name, data)
            self.instruments.append(instrument)
        self.data_length = len(self.instruments[0].data)

    def _select_instruments(self) -> List[str]:
        """
        Selects instruments based on available data and configuration.

        Returns:
            List[str]: List of instrument names to be used.
        """

        instrument_files = self.load_instrument_files()
        available_instruments = set()
        ignore_instruments = set(self.config.get('ignore_instruments', []))
        for file_name in instrument_files:
            instrument_name = self.verify_csv(file_name)
            if not instrument_name:
                continue

            quote_currency, base_currency = parse_instrument_name(instrument_name)
            if base_currency in self.currencies and quote_currency in self.currencies:
                if instrument_name not in ignore_instruments:
                    available_instruments.add(instrument_name)

        if not available_instruments:
            raise ValueError("No instruments found that match the currencies and interval specified.")

        return sorted(list(available_instruments))

    def load_instrument_files(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        instrument_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        return instrument_files

    def verify_csv(self, file_name):
        parts = file_name.split('_')
        if len(parts) != 2:
            return None
        instrument_name, file_interval = parts
        file_interval = file_interval.replace('.csv', '')
        if int(file_interval) != self.interval:
            return None
        return instrument_name

    def print_loaded_instruments(self, datasets):
        print("Loaded Instruments (before synchronization):")
        for name in datasets.keys():
            data = datasets[name]
            quote_currency, base_currency = parse_instrument_name(name)
            start_time = pd.to_datetime(data['time'].iloc[0], unit='s')
            end_time = pd.to_datetime(data['time'].iloc[-1], unit='s')
            print(f"Instrument: {name}, Base: {base_currency}, Quote: {quote_currency}, "
                  f"Start: {start_time}, End: {end_time}")

    def print_synchronized_period(self):
        data = self.instruments[0].data
        start_time = pd.to_datetime(data['time'].iloc[0], unit='s')
        end_time = pd.to_datetime(data['time'].iloc[-1], unit='s')
        print(f"Synchronized times are : \n\tStart: {start_time}, \n\tEnd: {end_time}")

    def _init_usable_instruments(self):
        self.usable_instruments = self.config.get('usable_instruments', [i.name for i in self.instruments])
        for instrument in self.usable_instruments:
            if instrument not in [i.name for i in self.instruments]:
                raise ValueError(f'{instrument} selected in usable_instruments, but uses at least one currency absent from the wallet.')

    def _init_action_space(self):
        if self.discretize:
            self.discrete_actions: List[int] = self.config['discrete_actions']
            N_actions = len(self.discrete_actions) * len(self.usable_instruments)
            self.action_space = spaces.Discrete(N_actions)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.usable_instruments),), dtype=np.float32)

    def reset(self):
        """Resets the environment to the initial state."""
        self.wallet.reset()
        self.step0 = random.randrange(self.window_size - 1, self.data_length - self.window_size)
        self.current_step = self.step0
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
        self._execute_actions(action)
        self.current_step += 1
        done = self._get_is_done()
        reward = self._compute_reward()
        if done:
            observation = None
        else:
            observation = self._get_observation()
        info = {}
        return observation, reward, done, info

    def _execute_actions(self, actions: List[float]):
        """Executes the agent's actions, ensuring that total sold amounts do not exceed wallet balances."""
        if self.discretize:
            actions = self._preprocess_discrete_actions(actions)
        action_details = self._compute_action_details(actions)
        from_currency_totals = self._calculate_total_sold_amounts(action_details)
        adjusted_action_details = self._adjust_action_amounts(action_details, from_currency_totals)
        self._apply_actions(adjusted_action_details)

    def _preprocess_discrete_actions(self, action, debug=False):
        instr_idx = action // len(self.discrete_actions)
        value_idx = action % len(self.discrete_actions)
        action_value = self.discrete_actions[value_idx]
        actions = [0 for _ in range(len(self.usable_instruments))]
        actions[instr_idx] = action_value

        if debug:
            print(f'\nDiscrete action: {action}')
            print(f'instr_idx: {instr_idx}')
            print(f'instr_name: {self.usable_instruments[instr_idx]}')
            print(f'value_idx: {value_idx}')
            print(f'actions: {actions}')
        return actions

    def _compute_action_details(self, actions: List[float]) -> List[dict]:
        """Computes action details including amounts to be traded."""
        action_details = []
        for action_value, instrument_name in zip(actions, self.usable_instruments):
            instrument = [i for i in self.instruments if i.name == instrument_name][0]
            if action_value == 0:
                continue  # No action
            from_currency, to_currency, amount, rate = self._get_trade_parameters(action_value, instrument)
            if amount <= 0:
                continue  # Skip actions with zero or negative amount
            action_details.append({
                'from_currency': from_currency,
                'to_currency': to_currency,
                'amount': amount,
                'rate': rate
            })
        return action_details

    def _get_trade_parameters(self, action_value: float, instrument: Instrument, debug: bool = False):
        """Determines the currencies involved, amount to trade, and exchange rate."""
        rate = instrument.get_rate(self.current_step)
        if action_value > 0:
            from_currency = instrument.base_currency
            to_currency = instrument.quote_currency
            balance = self.wallet.balances.get(from_currency, 0)
            amount = action_value * balance
            if debug:
                print(f"Buying {to_currency} with {from_currency}")
                print(f"Rate: {rate}, Amount: {amount}")
        else:
            from_currency = instrument.quote_currency
            to_currency = instrument.base_currency
            balance = self.wallet.balances.get(from_currency, 0)
            amount = -action_value * balance
            rate = 1 / rate  # Invert the rate when selling
            if debug:
                print(f"Selling {from_currency} for {to_currency}")
                print(f"Inverted Rate: {rate}, Amount: {amount}")
        return from_currency, to_currency, amount, rate

    def _calculate_total_sold_amounts(self, action_details: List[dict]) -> Dict[str, float]:
        """Calculates the total amounts to be sold per currency."""
        from_currency_totals = {}
        for action in action_details:
            from_currency = action['from_currency']
            amount = action['amount']
            from_currency_totals[from_currency] = from_currency_totals.get(from_currency, 0) + amount
        return from_currency_totals

    def _adjust_action_amounts(self, action_details: List[dict], from_currency_totals: Dict[str, float]) -> List[dict]:
        """Adjusts action amounts if total sold exceeds balance."""
        adjusted_actions = []
        for action in action_details:
            from_currency = action['from_currency']
            total_sold = from_currency_totals[from_currency]
            balance = self.wallet.balances.get(from_currency, 0)
            if total_sold > balance:
                scale_factor = balance / total_sold
                action['amount'] *= scale_factor
            adjusted_actions.append(action)
        return adjusted_actions

    def _apply_actions(self, action_details: List[dict], debug: bool = False):
        """Applies the adjusted actions to the wallet."""
        for action in action_details:
            if debug:
                print(f"Executing Transaction: {action}")
            self.wallet.execute_transaction(
                action['from_currency'],
                action['to_currency'],
                action['amount'],
                action['rate']
            )
            if debug:
                print(f"Balances after transaction: {self.wallet.balances}")

    def _get_is_done(self):
        steps_elapsed = self.current_step - self.step0
        return steps_elapsed >= self.episode_length

    def _compute_reward(self) -> float:
        """Computes the reward for the current step."""
        currency_pairs = {}
        for instrument in self.instruments:
            quote_currency = instrument.quote_currency
            base_currency = instrument.base_currency
            rate = instrument.get_rate(self.current_step)
            currency_pairs[(base_currency, quote_currency)] = rate
        exchange_rates = compute_exchange_rates(self.base_currency, currency_pairs)
        total_value = self.wallet.get_total_value(exchange_rates, self.base_currency)
        reward = total_value - self.previous_value
        self.previous_value = total_value
        return reward

    def _get_observation(self) -> np.ndarray:
        """Retrieves the current observation."""
        observations = {}


        for cat_num in ['numerical', 'categorical']:
            obs = []
            field_names = self.obs_format['sequence'][cat_num]
            for instrument in self.instruments:
                data_window = instrument.get_history(self.current_step, self.window_size, cols=field_names) # (window_size, N_obs)
                obs.append(data_window.values) # (N_instr, window_size, N_obs)
    
            observations[f'sequence_{cat_num}'] = np.transpose(np.array(obs), (1, 0, 2)) # (window_size, N_instr, N_obs)

        for cat_num in ['numerical', 'categorical']:
            obs = []
            field_names = self.obs_format['constant'][cat_num]
            instrument = self.instruments[0]
            data_window = instrument.get_last(self.current_step, cols=field_names)
            obs.append(data_window.values)
    
            observations[f'constant_{cat_num}'] = np.array(obs)    
        
        return observations

    def render(self, mode='human'):
        """Renders the environment state."""
        print(f"Step: {self.current_step}")
        print(f"Wallet Balances: {self.wallet.balances}")

    def close(self):
        """Cleans up the environment."""
        pass

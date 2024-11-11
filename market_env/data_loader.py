"""
Data Loading and Preprocessing Module.

This module handles loading market data from CSV files and preprocessing it for use in the environment.
"""

import os
from typing import List, Dict
import pandas as pd
import numpy as np
from joblib import Parallel, delayed


def load_instrument_data(
    instrument_name: str,
    interval: int = 1,
    fill_missing: bool = True
) -> pd.DataFrame:
    """
    Loads data for a single instrument.

    Args:
        instrument_name (str): Name of the instrument.
        interval (int): Time interval in minutes.
        fill_missing (bool): Whether to fill missing data.

    Returns:
        pd.DataFrame: DataFrame containing the instrument's data.
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    file_path = os.path.join(data_dir, f"{instrument_name}_{interval}.csv")
    columns = ["time", "open", "high", "low", "close", "volume", "trades"]
    data = pd.read_csv(file_path, header=None, names=columns, skiprows=1)
    if fill_missing:
        data = _fill_missing_data(data, interval)
    return data


def load_multiple_instruments(
    instrument_names: List[str],
    interval: int = 1,
    fill_missing: bool = True,
    n_jobs: int = -1
) -> Dict[str, pd.DataFrame]:
    """
    Loads data for multiple instruments in parallel.

    Args:
        instrument_names (List[str]): List of instrument names.
        interval (int): Time interval in minutes.
        fill_missing (bool): Whether to fill missing data.
        n_jobs (int): Number of parallel jobs.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping instrument names to their data.
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(load_instrument_data)(name, interval, fill_missing) for name in instrument_names
    )
    return dict(zip(instrument_names, results))


def synchronize_data(datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Synchronizes multiple datasets to have the same time range.

    Args:
        datasets (Dict[str, pd.DataFrame]): Datasets to synchronize.

    Returns:
        Dict[str, pd.DataFrame]: Synchronized datasets.
    """
    start_time = max(df['time'].iloc[0] for df in datasets.values())
    end_time = min(df['time'].iloc[-1] for df in datasets.values())
    synchronized = {}
    for name, df in datasets.items():
        mask = (df['time'] >= start_time) & (df['time'] <= end_time)
        synchronized[name] = df.loc[mask].reset_index(drop=True)
    return synchronized


def _fill_missing_data(data: pd.DataFrame, interval: int) -> pd.DataFrame:
    """
    Fills missing data in the DataFrame.

    Args:
        data (pd.DataFrame): DataFrame to fill.
        interval (int): Time interval in minutes.

    Returns:
        pd.DataFrame: DataFrame with missing data filled.
    """
    full_time_range = np.arange(data['time'].min(), data['time'].max() + interval * 60, interval * 60)
    df_full = pd.DataFrame({'time': full_time_range})
    df_full = df_full.merge(data, on='time', how='left')
    df_full[['open', 'high', 'low', 'close']] = df_full[['open', 'high', 'low', 'close']].fillna(method='ffill')
    df_full[['volume', 'trades']] = df_full[['volume', 'trades']].fillna(0)
    return df_full

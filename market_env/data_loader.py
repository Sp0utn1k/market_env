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
    cols: List[str],
    interval: int = 1,
    fill_missing: bool = True,
    dtype: str = 'float32',
    print_sample: bool = False,
    nrows: int = 0
) -> pd.DataFrame:
    """
    Loads data for a single instrument.

    Args:
        instrument_name (str): Name of the instrument.
        interval (int): Time interval in minutes.
        fill_missing (bool): Whether to fill missing data.

    Returns:
        pd.DataFrame: DataFrame containing the instrument's data.

    Raises:
        FileNotFoundError: If the data file does not exist.
        ValueError: If the data file is empty or corrupted.
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    file_path = os.path.join(data_dir, f"{instrument_name}_{interval}.csv")
    columns = ["time", "open", "high", "low", "close", "volume", "trades"]
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file for instrument {instrument_name} with interval {interval} not found.")

    try:
        data = pd.read_csv(file_path, header=None, names=columns, skiprows=1)
        if data.empty:
            raise ValueError(f"Data file {file_path} is empty.")
    except Exception as e:
        raise ValueError(f"Error reading data file {file_path}: {e}")

    if fill_missing:
        data = _fill_missing_data(data, interval)

    data = _preprocess_data(data, cols)

    data = convert_to_type(data, dtype)
    if print_sample:
        print(f"Sample data for {instrument_name}:")
        print(data.head())
    return data


def load_multiple_instruments(
    instrument_names: List[str],
    cols: List[str],
    interval: int = 1,
    fill_missing: bool = True,
    dtype: str = 'float32',
    print_sample: bool = False,
    nrows: int = 0,
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
        delayed(load_instrument_data)(name, cols, interval=interval, fill_missing=fill_missing, dtype=dtype, print_sample=print_sample, nrows=nrows) for name in instrument_names
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
    # Use ffill() instead of fillna(method='ffill')
    df_full[['open', 'high', 'low', 'close']] = df_full[['open', 'high', 'low', 'close']].ffill()
    df_full[['volume', 'trades']] = df_full[['volume', 'trades']].fillna(0)
    return df_full

def _preprocess_data(df, cols):

    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    
    for col in cols:
        match col:
            # Time data
            case 'year':
                df['year'] = df['datetime'].dt.year
            case 'year_scaled':
                df['year_scaled'] = (df['datetime'].dt.year - 2001) / 20
            case 'day_of_year':
                df['day_of_year'] = df['datetime'].dt.dayofyear
            case 'day_of_week':
                df['day_of_week'] = df['datetime'].dt.dayofweek
            case 'sin_hour':
                df['sin_hour'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
            case 'cos_hour':
                df['cos_hour'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
            case 'sin_min':
                df['sin_min'] = np.sin(2 * np.pi * df['datetime'].dt.minute / 60)
            case 'cos_min':
                df['cos_min'] = np.cos(2 * np.pi * df['datetime'].dt.minute / 60)

            # Trading data
            case 'volume_log':
                df['volume_log'] = np.log(df['volume'] + 1)
            case 'trades_log':
                df['trades_log'] = np.log(df['trades'] + 1)

    # for col in ['open', 'high', 'low', 'close']:
    #     df[f'{col}_log_return'] = np.log(df[col] / df[col].shift(1)).fillna(0)
    
    
    
    df = df.drop(columns=['datetime'])
    
    return df[['time']+cols]

def get_available_instruments(interval: int = None) -> List[str]:
    """
    Returns a list of available instrument names.

    Args:
        interval (int, optional): Time interval in minutes. If specified, only instruments with this interval are returned.

    Returns:
        List[str]: List of available instrument names.
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    instrument_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    instruments = set()

    for file_name in instrument_files:
        parts = file_name.split('_')
        if len(parts) != 2:
            continue  # Unexpected file name format
        instrument_name, file_interval = parts
        file_interval = file_interval.replace('.csv', '')
        if interval is not None and int(file_interval) != interval:
            continue  # Skip if interval doesn't match
        instruments.add(instrument_name)
    return list(instruments)

if __name__ == '__main__':
    instruments = get_available_instruments(interval=1)
    print(instruments)

def convert_to_type(df, type):
    # Convert all columns except 'time' to float32
    for col in df.columns:
        if col != 'time':
            df[col] = df[col].astype(type)
    return df

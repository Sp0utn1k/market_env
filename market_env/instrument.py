"""
Instrument Module.

Defines the Instrument class representing a trading instrument.
"""

import pandas as pd


class Instrument:
    """Represents a trading instrument with associated market data."""

    def __init__(self, name: str, data: pd.DataFrame):
        """
        Initializes the Instrument.

        Args:
            name (str): Name of the instrument.
            data (pd.DataFrame): Historical market data for the instrument.
        """
        self.name = name
        self.data = data
        self.quote_currency, self.base_currency = self._parse_instrument_name()

    def _parse_instrument_name(self):
        """
        Parses the instrument name to extract quote and base currencies.

        Returns:
            Tuple[str, str]: Quote and base currencies.
        """
        if '.' in self.name:
            return tuple(self.name.split('.'))
        else:
            # Assuming the base currency is the last 3 characters
            return self.name[:-3], self.name[-3:]

    def get_rate(self, index: int) -> float:
        """
        Retrieves the exchange rate at a specific index.

        Args:
            index (int): Index in the data DataFrame.

        Returns:
            float: Exchange rate (closing price).
        """
        return self.data.iloc[index]['close']

    def get_history(self, index: int, window: int) -> pd.DataFrame:
        """
        Retrieves historical data up to a specific index.

        Args:
            index (int): Current index in the data DataFrame.
            window (int): Number of past data points to retrieve.

        Returns:
            pd.DataFrame: Historical data window.
        """
        if index - window + 1 < 0:
            raise IndexError("Index out of bounds for the historical window.")
        return self.data.iloc[index - window + 1:index + 1].reset_index(drop=True)

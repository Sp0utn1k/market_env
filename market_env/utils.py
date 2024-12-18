"""
Utility Functions Module.

Provides utility functions for economic computations and action normalization.
"""

from typing import Dict, List, Tuple
from collections import deque


# List of known base currencies
BASE_CURRENCIES = [
    'BTC', 'USD', 'EUR', 'USDT', 'ETH', 'AUD', 'GBP', 'CHF', 'CAD',
    'USDC', 'JPY', 'DAI', 'AED', '.SETH', 'XBT', 'POL', 'DOT'
]


def compute_exchange_rates(base_currency: str, currency_pairs: Dict[Tuple[str, str], float]) -> Dict[str, float]:
    """
    Computes exchange rates from various currencies to the base currency.

    Args:
        base_currency (str): The base currency to compute rates to.
        currency_pairs (Dict[Tuple[str, str], float]): Known currency pairs and their rates.

    Returns:
        Dict[str, float]: Exchange rates to the base currency.
    """
    rates = {base_currency: 1.0}
    queue = deque([base_currency])

    while queue:
        current = queue.popleft()
        current_rate = rates[current]
        for (from_curr, to_curr), rate in currency_pairs.items():
            if from_curr == current and to_curr not in rates:
                rates[to_curr] = current_rate * rate
                queue.append(to_curr)
            elif to_curr == current and from_curr not in rates:
                rates[from_curr] = current_rate / rate
                queue.append(from_curr)
    return rates


def normalize_actions(actions: List[float]) -> List[float]:
    """
    Normalizes actions to ensure they are within [-1, 1].

    Args:
        actions (List[float]): List of raw action values.

    Returns:
        List[float]: Normalized action values.
    """
    normalized = []
    for action in actions:
        if action > 1.0:
            normalized.append(1.0)
        elif action < -1.0:
            normalized.append(-1.0)
        else:
            normalized.append(action)
    return normalized


def parse_instrument_name(name: str) -> Tuple[str, str]:
    """
    Parses an instrument name to extract quote and base currencies.

    Args:
        name (str): Instrument name.

    Returns:
        Tuple[str, str]: (quote_currency, base_currency)
    """
    # Check if name contains a dot notation
    if '.' in name:
        return tuple(name.split('.'))

    # Try to match known base currencies at the end of the name
    for base_currency in sorted(BASE_CURRENCIES, key=len, reverse=True):
        if name.endswith(base_currency):
            quote_currency = name[:-len(base_currency)]
            return quote_currency, base_currency

    # If no known base currency is found, raise an error
    raise ValueError(f"Cannot parse instrument name '{name}'. Unknown base currency.")

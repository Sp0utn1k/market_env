"""
Wallet Module.

Manages the wallet for handling balances and executing transactions.
"""

from typing import Dict
import warnings


class Wallet:
    """Manages currency balances and executes transactions."""

    def __init__(self, currencies: Dict[str, float], fees: float = 0.00075):
        """
        Initializes the Wallet with given currencies and balances.

        Args:
            currencies (Dict[str, float]): Initial currency balances.
            fees (float): Transaction fee percentage.
        """
        self.initial_balances = currencies.copy()
        self.balances = currencies.copy()
        self.fees = fees

    def reset(self):
        """Resets the wallet to initial balances."""
        self.balances = self.initial_balances.copy()

    def execute_transaction(self, from_currency: str, to_currency: str, amount: float, rate: float):
        """
        Executes a currency exchange transaction.

        Args:
            from_currency (str): Currency to sell.
            to_currency (str): Currency to buy.
            amount (float): Amount of from_currency to sell.
            rate (float): Exchange rate from from_currency to to_currency.

        Raises:
            ValueError: If transaction parameters are invalid.
        """
        if amount <= 0:
            raise ValueError("Transaction amount must be positive.")
        if rate <= 0:
            raise ValueError("Exchange rate must be positive.")
        if self.balances.get(from_currency, 0) < amount:
            raise ValueError(f"Insufficient {from_currency} balance.")

        fee = amount * self.fees
        net_amount = amount - fee
        converted_amount = net_amount / rate

        self.balances[from_currency] -= amount
        self.balances[to_currency] = self.balances.get(to_currency, 0) + converted_amount

        # Ensure balances are standard floats
        self.balances[from_currency] = float(self.balances[from_currency])
        self.balances[to_currency] = float(self.balances[to_currency])


    def get_total_value(self, rates: Dict[str, float], base_currency: str) -> float:
        """
        Calculates the total wallet value in the base currency.

        Args:
            rates (Dict[str, float]): Exchange rates to the base currency.
            base_currency (str): Currency to express the total value in.

        Returns:
            float: Total wallet value in base currency.
        """
        total = 0.0
        for currency, balance in self.balances.items():
            if currency == base_currency:
                total += balance
            elif currency in rates:
                total += balance * rates[currency]
            else:
                warnings.warn(f"No exchange rate for {currency} to {base_currency}.")
        return total

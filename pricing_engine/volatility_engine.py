"""
volatility_engine.py
Purpose: Compute dynamic rolling volatility for the beverage portfolio 
pricing optimization engine.
"""

%%writefile volatility_engine.py
import pandas as pd
import numpy as np
import logging

class VolatilityEngine:
    def __init__(self, window=30):
        self.window = window

    def calculate_log_returns(self, prices: pd.Series) -> pd.Series:
        return np.log(prices / prices.shift(1))

    def get_rolling_volatility(self, prices: pd.Series) -> pd.Series:
        log_returns = self.calculate_log_returns(prices)
        return log_returns.rolling(window=self.window).std() * np.sqrt(252)

    def run_engine(self, data: pd.DataFrame, price_col: str) -> pd.DataFrame:
        if price_col not in data.columns:
            raise ValueError(f"The specified column '{price_col}' does not exist.")
        data['volatility'] = self.get_rolling_volatility(data[price_col])
        return data

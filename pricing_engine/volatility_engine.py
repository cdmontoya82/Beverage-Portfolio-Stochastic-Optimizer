"""
volatility_engine.py
Purpose: Compute dynamic rolling volatility for the beverage portfolio 
pricing optimization engine.
"""

import pandas as pd
import numpy as np
import logging

# Log configuration for observability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VolatilityEngine:
    def __init__(self, window=30):
        """
        :param window: Rolling window size (in days) to calculate volatility.
        """
        self.window = window

    def calculate_log_returns(self, prices: pd.Series) -> pd.Series:
        """Calculates logarithmic returns to normalize the time series."""
        return np.log(prices / prices.shift(1))

    def get_rolling_volatility(self, prices: pd.Series) -> pd.Series:
        """
        Calculates annualized rolling volatility assuming 252 trading days.
        """
        log_returns = self.calculate_log_returns(prices)
        rolling_vol = log_returns.rolling(window=self.window).std() * np.sqrt(252)
        return rolling_vol

    def run_engine(self, data: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """
        Executes the volatility calculation pipeline.
        """
        logging.info("Initializing volatility computation...")
        
        # Core data validation
        if price_col not in data.columns:
            raise ValueError(f"The specified column '{price_col}' does not exist in the DataFrame.")

        data['volatility'] = self.get_rolling_volatility(data[price_col])
        
        logging.info("Volatility computation completed successfully.")
        return data

# --- UNIT TESTING AND EXECUTION ---
if __name__ == "__main__":
    # Simulating price data for a beverage product (SKU_001)
    dates = pd.date_range(start='2026-01-01', periods=100)
    prices = 100 + np.cumsum(np.random.normal(0, 1, 100))
    df = pd.DataFrame({'price': prices}, index=dates)

    # Instantiate engine with a standard monthly trading window (21 days)
    engine = VolatilityEngine(window=21)
    result = engine.run_engine(df, 'price')
    
    print(result.tail())

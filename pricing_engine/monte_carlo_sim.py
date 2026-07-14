"""
monte_carlo_sim.py
Purpose: Execute Geometric Brownian Motion (GBM) simulations to model 
future pricing and demand paths under stochastic uncertainty.
"""

import numpy as np
import pandas as pd
import logging

# Log configuration for observability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MonteCarloSimulator:
    def __init__(self, num_simulations=10000, time_horizon=30):
        """
        :param num_simulations: Total number of stochastic paths to generate.
        :param time_horizon: Number of future days to project.
        """
        self.num_simulations = num_simulations
        self.time_horizon = time_horizon

    def simulate_paths(self, current_price: float, volatility: float, risk_free_rate=0.05) -> np.ndarray:
        """
        Generates price paths using Geometric Brownian Motion (GBM).
        Formula: S(t) = S(0) * exp((mu - 0.5 * sigma^2)*dt + sigma * dW)
        """
        logging.info(f"Running {self.num_simulations} Monte Carlo paths for horizon: {self.time_horizon} days.")
        
        # Guard rails for inputs
        if volatility <= 0 or np.isnan(volatility):
            logging.warning("Invalid volatility encountered. Defaulting to historical baseline.")
            volatility = 0.20 # 20% baseline fallback
            
        dt = 1 / 252  # Daily time step assuming 252 trading days per year
        mu = risk_free_rate
        sigma = volatility

        # Initialize simulation matrix (Days x Simulations)
        sim_matrix = np.zeros((self.time_horizon + 1, self.num_simulations))
        sim_matrix[0] = current_price

        # Vectorized generation of standard normal random variables
        for t in range(1, self.time_horizon + 1):
            Z = np.random.normal(0, 1, self.num_simulations)
            # Apply the log-normal drift and diffusion steps
            drift = (mu - 0.5 * sigma ** 2) * dt
            diffusion = sigma * np.sqrt(dt) * Z
            sim_matrix[t] = sim_matrix[t - 1] * np.exp(drift + diffusion)

        logging.info("Simulation matrix generated successfully.")
        return sim_matrix

# --- UNIT TESTING AND EXECUTION ---
if __name__ == "__main__":
    # Mock parameters from previous Volatility Engine step
    current_market_price = 120.50  # Base price for SKU_001
    calculated_volatility = 0.15   # 15% annualized rolling volatility

    simulator = MonteCarloSimulator(num_simulations=5000, time_horizon=15)
    paths = simulator.simulate_paths(current_market_price, calculated_volatility)
    
    # Quick metrics verification
    final_day_prices = paths[-1]
    print(f"Simulation matrix shape: {paths.shape} (Days x Paths)")
    print(f"Expected Mean Price at day 15: {final_day_prices.mean():.2f}")
    print(f"95% Value at Risk (VaR) lower bound: {np.percentile(final_day_prices, 5):.2f}")

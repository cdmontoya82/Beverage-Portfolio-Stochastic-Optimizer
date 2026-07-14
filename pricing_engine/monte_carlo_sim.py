"""
monte_carlo_sim.py
Purpose: Execute Geometric Brownian Motion (GBM) simulations to model 
future pricing and demand paths under stochastic uncertainty.
"""


%%writefile monte_carlo_sim.py
import numpy as np
import logging

class MonteCarloSimulator:
    def __init__(self, num_simulations=10000, time_horizon=30):
        self.num_simulations = num_simulations
        self.time_horizon = time_horizon

    def simulate_paths(self, current_price: float, volatility: float, risk_free_rate=0.05) -> np.ndarray:
        if volatility <= 0 or np.isnan(volatility):
            volatility = 0.20
            
        dt = 1 / 252  
        mu = risk_free_rate
        sigma = volatility

        sim_matrix = np.zeros((self.time_horizon + 1, self.num_simulations))
        sim_matrix[0] = current_price

        for t in range(1, self.time_horizon + 1):
            Z = np.random.normal(0, 1, self.num_simulations)
            drift = (mu - 0.5 * sigma ** 2) * dt
            diffusion = sigma * np.sqrt(dt) * Z
            sim_matrix[t] = sim_matrix[t - 1] * np.exp(drift + diffusion)

        return sim_matrix

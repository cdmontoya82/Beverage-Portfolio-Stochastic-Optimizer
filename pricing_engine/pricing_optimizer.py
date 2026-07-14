"""
pricing_optimizer.py
Purpose: Core orchestrator that ingests historical beverage pricing data, 
computes stochastic risk metrics, and determines the optimal strategic price.
"""

import pandas as pd
import numpy as np
import logging
from volatility_engine import VolatilityEngine
from monte_carlo_sim import MonteCarloSimulator

# Log configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PricingOptimizer:
    def __init__(self, target_margin=0.35, risk_tolerance=0.05):
        """
        :param target_margin: Minimum acceptable profit margin (e.g., 35%).
        :param risk_tolerance: Alpha level for Value at Risk (e.g., 5% for a 95% VaR).
        """
        self.target_margin = target_margin
        self.risk_tolerance = risk_tolerance

    def optimize_price(self, simulation_matrix: np.ndarray, unit_cost: float) -> dict:
        """
        Analyzes simulated price paths against production costs to calculate 
        expected margins and downside financial risk metrics.
        """
        logging.info("Optimizing price structure against simulated risk horizons...")
        
        # Extract projected prices at the end of the time horizon
        terminal_prices = simulation_matrix[-1]
        
        # Calculate standard financial risk metrics
        expected_price = float(np.mean(terminal_prices))
        var_lower_bound = float(np.percentile(terminal_prices, self.risk_tolerance * 100))
        
        # Quantitative logic: Calculate expected margin based on risk-adjusted downside
        expected_margin = (expected_price - unit_cost) / expected_price
        downside_margin = (var_lower_bound - unit_cost) / var_lower_bound
        
        # Optimization rule: Suggest a premium adjustment if downside risk breaches target margin
        suggested_price = expected_price
        if downside_margin < self.target_margin:
            # Adjust price upward to buffer against the 95% worst-case volatility scenario
            suggested_price = unit_cost / (1 - self.target_margin)
            logging.warning("Target margin breached under downside risk. Recommending risk-adjusted price premium.")

        return {
            "Expected Market Price": round(expected_price, 2),
            "95% Value at Risk (VaR) Limit": round(var_lower_bound, 2),
            "Expected Margin Status": f"{expected_margin * 100:.2f}%",
            "Downside Margin Status": f"{downside_margin * 100:.2f}%",
            "Optimized Strategic Price": round(suggested_price, 2)
        }

# --- COMPLETE PIPELINE EXECUTION ENGINE ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("   EXECUTING QUANTITATIVE BEVERAGE PRICING ENGINE   ")
    print("="*50 + "\n")

    # STEP 1: Simulate historical ingestion of beverage sales data (e.g., Premium Soda SKU)
    np.random.seed(42)  # For reproducible results
    dates = pd.date_range(start='2026-01-01', periods=120, freq='D')
    historical_prices = 150 + np.cumsum(np.random.normal(0, 1.5, 120))  # Base price ~150
    beverage_df = pd.DataFrame({'retail_price': historical_prices}, index=dates)

    # STEP 2: Compute historical rolling volatility via VolatilityEngine
    vol_engine = VolatilityEngine(window=21)  # 21-day trading month window
    processed_df = vol_engine.run_engine(beverage_df, 'retail_price')
    
    # Extract latest active state metrics
    latest_price = float(processed_df['retail_price'].iloc[-1])
    latest_volatility = float(processed_df['volatility'].iloc[-1])
    
    print(f"Current Market Price: ${latest_price:.2f}")
    print(f"Annualized Rolling Volatility: {latest_volatility * 100:.2f}%\n")

    # STEP 3: Generate 10,000 stochastic future pricing paths via MonteCarloSimulator
    simulation_horizon = 30  # Project 30 days into the future
    mc_simulator = MonteCarloSimulator(num_simulations=10000, time_horizon=simulation_horizon)
    sim_paths = mc_simulator.simulate_paths(latest_price, latest_volatility)

    # STEP 4: Run Optimization based on cost parameters and profit margins
    beverage_unit_cost = 95.00  # Production + logistics cost per unit
    optimizer = PricingOptimizer(target_margin=0.35, risk_tolerance=0.05)
    optimization_results = optimizer.optimize_price(sim_paths, beverage_unit_cost)

    # STEP 5: Display production-ready diagnostic report
    print("\n" + "-"*40)
    print("        STRATEGIC QUANT REPORT        ")
    print("-"*40)
    for metric, value in optimization_results.items():
        print(f"{metric:<30}: {value}")
    print("-"*40 + "\n")

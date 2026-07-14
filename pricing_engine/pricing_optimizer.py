"""
pricing_optimizer.py
Purpose: Core orchestrator that executes non-linear mathematical optimization 
via SciPy to maximize expected revenue under stochastic risk and market constraints.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Appending current directory to path to ensure clean modular imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Modular imports from your repository framework
from volatility_engine import VolatilityEngine
from monte_carlo_sim import MonteCarloSimulator

# Log configuration for enterprise-level observability
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class PricingOptimizer:
    def __init__(self, target_margin=0.35, risk_tolerance=0.05, price_elasticity=-2.5):
        """
        :param target_margin: Minimum acceptable corporate profit margin (e.g., 35%).
        :param risk_tolerance: Alpha level for Downside Value at Risk (e.g., 5%).
        :param price_elasticity: Empirical price elasticity of demand for the portfolio.
        """
        self.target_margin = target_margin
        self.risk_tolerance = risk_tolerance
        self.elasticity = price_elasticity

    def _estimate_demand(self, current_price: float, base_price: float, base_demand=10000) -> float:
        """
        Models the real economic impact on sales volume using price elasticity.
        """
        price_change_pct = (current_price - base_price) / base_price
        demand_change_pct = price_change_pct * self.elasticity
        projected_demand = base_demand * (1 + demand_change_pct)
        return max(0.0, projected_demand)

    def _objective_function(self, price_multiplier: list, base_price: float, unit_cost: float, var_lower_bound: float) -> float:
        """
        Advanced Optimization Objective: Maximizes overall portfolio net profit 
        while applying a soft mathematical penalty if the downside risk breaches targets.
        """
        projected_price = base_price * price_multiplier[0]
        projected_volume = self._estimate_demand(projected_price, base_price)
        
        # Base commercial net profit
        total_profit = projected_volume * (projected_price - unit_cost)
        
        # Risk evaluation at the 95% worst-case horizon scenario
        simulated_var_price = var_lower_bound * price_multiplier[0]
        downside_margin = (simulated_var_price - unit_cost) / simulated_var_price
        
        # Soft Constraint Penalty Function
        penalty = 0.0
        if downside_margin < self.target_margin:
            penalty = 1000000.0 * (self.target_margin - downside_margin) ** 2
            
        return -(total_profit - penalty)  # Negative for SciPy minimization algorithms

    def optimize_price(self, simulation_matrix: np.ndarray, unit_cost: float) -> dict:
        """
        Executes the optimization routine over simulated paths using the Nelder-Mead simplex algorithm.
        """
        logging.info("Executing penalized mathematical optimization via SciPy...")
        
        terminal_prices = simulation_matrix[-1]
        expected_market_price = float(np.mean(terminal_prices))
        var_lower_bound = float(np.percentile(terminal_prices, self.risk_tolerance * 100))

        initial_guess = [1.0]  # Start at baseline market multiplier
        
        # Robust simplex optimization across strict commercial bounds
        solver_result = minimize(
            self._objective_function, 
            initial_guess, 
            args=(expected_market_price, unit_cost, var_lower_bound), 
            method='Nelder-Mead',
            bounds=[(0.90, 1.30)]  # Limits operational adjustment between -10% and +30%
        )

        optimized_multiplier = solver_result.x[0]
        optimized_price = expected_market_price * optimized_multiplier
        
        expected_margin = (optimized_price - unit_cost) / optimized_price
        downside_margin = ((var_lower_bound * optimized_multiplier) - unit_cost) / (var_lower_bound * optimized_multiplier)
        
        initial_volume = 10000
        optimized_volume = self._estimate_demand(optimized_price, expected_market_price, initial_volume)

        return {
            "Baseline Expected Price": round(expected_market_price, 2),
            "95% Value at Risk (VaR) Limit": round(var_lower_bound, 2),
            "Optimized Strategic Price": round(optimized_price, 2),
            "Risk-Adjusted Expected Margin": f"{expected_margin * 100:.2f}%",
            "Worst-Case Downside Margin": f"{downside_margin * 100:.2f}%",
            "Projected Volume Impact": f"{((optimized_volume - initial_volume) / initial_volume) * 100:.2f}%",
            "Optimizer Convergence Status": "SUCCESS" if solver_result.success else "FAILED"
        }

# --- END-TO-END PIPELINE EXECUTION ENGINE ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("   EXECUTING QUANTITATIVE BEVERAGE PRICING ENGINE   ")
    print("="*50 + "\n")

    # STEP 1: Simulating historical enterprise sales ledger data
    np.random.seed(42)
    dates = pd.date_range(start='2026-01-01', periods=120, freq='D')
    historical_prices = 150 + np.cumsum(np.random.normal(0, 1.5, 120))
    beverage_df = pd.DataFrame({'retail_price': historical_prices}, index=dates)

    # STEP 2: Ingest data into the Volatility Engine
    vol_engine = VolatilityEngine(window=21)
    processed_df = vol_engine.run_engine(beverage_df, 'retail_price')
    
    latest_price = float(processed_df['retail_price'].iloc[-1])
    latest_volatility = float(processed_df['volatility'].iloc[-1])
    
    print(f"Current Market Price: ${latest_price:.2f}")
    print(f"Annualized Rolling Volatility: {latest_volatility * 100:.2f}%\n")

    # STEP 3: Propagate parameters into the Stochastic Simulation Engine
    simulation_horizon = 30
    mc_simulator = MonteCarloSimulator(num_simulations=10000, time_horizon=simulation_horizon)
    sim_paths = mc_simulator.simulate_paths(latest_price, latest_volatility)

    # STEP 4: Run mathematical optimization under constraints
    beverage_unit_cost = 95.00
    optimizer = PricingOptimizer(target_margin=0.35, risk_tolerance=0.05)
    optimization_results = optimizer.optimize_price(sim_paths, beverage_unit_cost)

    # STEP 5: Generate Senior Quant Diagnostic Report
    print("\n" + "-"*40)
    print("        STRATEGIC QUANT REPORT        ")
    print("-"*40)
    for metric, value in optimization_results.items():
        print(f"{metric:<30}: {value}")
    print("-"*40 + "\n")

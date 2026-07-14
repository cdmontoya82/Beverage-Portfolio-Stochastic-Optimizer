"""
pricing_optimizer.py
Purpose: Core orchestrator that integrates the VolatilityEngine, MonteCarloSimulator,
and DemandSimulator to execute mathematical pricing optimization via SciPy.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Notebook-safe path injection: uses current working directory if __file__ is missing
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()

if current_dir not in sys.path:
    sys.path.append(current_dir)

# Importing your decoupled repository modules safely from the background files
from volatility_engine import VolatilityEngine
from monte_carlo_sim import MonteCarloSimulator
from demand_simulator import DemandSimulator

# Log configuration for enterprise-level observability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PricingOptimizer:
    def __init__(self, target_margin=0.35, risk_tolerance=0.05, price_elasticity=-2.5, baseline_demand=10000):
        """
        :param target_margin: Minimum acceptable corporate profit margin (e.g., 35%).
        :param risk_tolerance: Alpha level for Downside Value at Risk (e.g., 5%).
        :param price_elasticity: Empirical price elasticity of demand for the portfolio.
        :param baseline_demand: Standard volume target at baseline pricing.
        """
        self.target_margin = target_margin
        self.risk_tolerance = risk_tolerance
        
        # Injecting your isolated demand module directly into the optimizer context
        self.demand_sim = DemandSimulator(price_elasticity=price_elasticity, baseline_demand=baseline_demand)

    def _objective_function(self, price_multiplier: list, base_price: float, unit_cost: float, var_lower_bound: float) -> float:
        """
        Advanced Optimization Objective: Maximizes aggregate gross net profit 
        using the isolated demand simulator, applying a penalty for risk breaches.
        """
        projected_price = base_price * price_multiplier[0]
        projected_volume = self.demand_sim.estimate_demand(projected_price, base_price)
        
        # Base expected net commercial yield
        total_profit = projected_volume * (projected_price - unit_cost)
        
        # Risk evaluation at the 95% worst-case price simulation distribution tail
        simulated_var_price = var_lower_bound * price_multiplier[0]
        downside_margin = (simulated_var_price - unit_cost) / simulated_var_price
        
        # Soft-Constraint Penalty Function deployment
        penalty = 0.0
        if downside_margin < self.target_margin:
            penalty = 1000000.0 * (self.target_margin - downside_margin) ** 2
            
        return -(total_profit - penalty)  # Inverted value returned for minimization solvers

    def optimize_price(self, simulation_matrix: np.ndarray, unit_cost: float) -> dict:
        """Executes non-linear optimization search over simulated paths using Nelder-Mead Simplex."""
        logging.info("Executing economic optimization under risk constraints via SciPy...")
        
        terminal_prices = simulation_matrix[-1]
        expected_market_price = float(np.mean(terminal_prices))
        var_lower_bound = float(np.percentile(terminal_prices, self.risk_tolerance * 100))

        initial_guess = [1.0]  # Commencing search directly at current market parity
        
        # Executing robust mathematical optimization solver
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
        
        initial_volume = self.demand_sim.baseline_demand
        optimized_volume = self.demand_sim.estimate_demand(optimized_price, expected_market_price)

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

    # STEP 1: Simulate transactional business ingestion
    np.random.seed(42)  
    dates = pd.date_range(start='2026-01-01', periods=120, freq='D')
    historical_prices = 150 + np.cumsum(np.random.normal(0, 1.5, 120))  
    beverage_df = pd.DataFrame({'retail_price': historical_prices}, index=dates)

    # STEP 2: Compute metrics via Volatility Engine file
    vol_engine = VolatilityEngine(window=21)  
    processed_df = vol_engine.run_engine(beverage_df, 'retail_price')
    
    latest_price = float(processed_df['retail_price'].iloc[-1])
    latest_volatility = float(processed_df['volatility'].iloc[-1])
    
    print(f"Current Market Price: ${latest_price:.2f}")
    print(f"Annualized Rolling Volatility: {latest_volatility * 100:.2f}%\n")

    # STEP 3: Generate paths via Monte Carlo Simulation file
    simulation_horizon = 30  
    mc_simulator = MonteCarloSimulator(num_simulations=10000, time_horizon=simulation_horizon)
    sim_paths = mc_simulator.simulate_paths(latest_price, latest_volatility)

    # STEP 4: Run optimization with Decoupled Demand file parameters
    beverage_unit_cost = 95.00  
    optimizer = PricingOptimizer(target_margin=0.35, risk_tolerance=0.05, price_elasticity=-2.5, baseline_demand=10000)
    optimization_results = optimizer.optimize_price(sim_paths, beverage_unit_cost)

    # STEP 5: Render Enterprise Strategic Quant Report
    print("\n" + "-"*40)
    print("        STRATEGIC QUANT REPORT        ")
    print("-"*40)
    for metric, value in optimization_results.items():
        print(f"{metric:<30}: {value}")
    print("-"*40 + "\n")

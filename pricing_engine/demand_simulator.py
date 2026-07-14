
"""
demand_simulator.py
Purpose: Models and simulates market demand curves, price elasticity, 
and volume fluctuations under strategic pricing adjustments.
"""

%%writefile demand_simulator.py
import logging

class DemandSimulator:
    def __init__(self, price_elasticity=-2.5, baseline_demand=10000):
        self.elasticity = price_elasticity
        self.baseline_demand = baseline_demand

    def estimate_demand(self, proposed_price: float, baseline_price: float) -> float:
        if baseline_price <= 0:
            return 0.0
        price_change_pct = (proposed_price - baseline_price) / baseline_price
        demand_change_pct = price_change_pct * self.elasticity
        projected_demand = self.baseline_demand * (1 + demand_change_pct)
        return max(0.0, projected_demand)

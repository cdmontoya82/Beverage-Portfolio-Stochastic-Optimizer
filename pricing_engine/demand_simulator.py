
"""
demand_simulator.py
Purpose: Models and simulates market demand curves, price elasticity, 
and volume fluctuations under strategic pricing adjustments.
"""

import logging

# Log configuration for observability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DemandSimulator:
    def __init__(self, price_elasticity=-2.5, baseline_demand=10000):
        """
        :param price_elasticity: The price elasticity of demand (e.g., -2.5).
        :param baseline_demand: Standard expected volume at baseline pricing.
        """
        self.elasticity = price_elasticity
        self.baseline_demand = baseline_demand

    def estimate_demand(self, proposed_price: float, baseline_price: float) -> float:
        """
        Calculates projected sales volume based on standard price elasticity.
        Formula: Q_projected = Q_base * (1 + elasticity * (dP / P_base))
        """
        if baseline_price <= 0:
            logging.error("Baseline price must be greater than zero. Unable to compute demand shift.")
            return 0.0

        price_change_pct = (proposed_price - baseline_price) / baseline_price
        demand_change_pct = price_change_pct * self.elasticity
        projected_demand = self.baseline_demand * (1 + demand_change_pct)
        
        # Demand cannot drop below zero in any realistic scenario
        final_demand = max(0.0, projected_demand)
        
        logging.debug(f"Price Shift: {price_change_pct*100:.2f}% | Projected Demand: {final_demand:.2f} units")
        return final_demand

# --- UNIT TESTING AND EXECUTION ---
if __name__ == "__main__":
    # Ingesting mock data to verify elasticity metrics
    base_p = 136.57
    test_p_high = 145.18  # Recommending a ~6.3% price premium
    test_p_low = 125.00   # Simulating a discount scenario
    
    simulator = DemandSimulator(price_elasticity=-2.5, baseline_demand=10000)
    
    volume_high = simulator.estimate_demand(test_p_high, base_p)
    volume_low = simulator.estimate_demand(test_p_low, base_p)
    
    print("\n" + "="*45)
    print("      DEMAND SIMULATION DIAGNOSTICS      ")
    print("="*45)
    print(f"Baseline Scenario : Price ${base_p:.2f}  ->  Demand: 10,000 units")
    print(f"Premium Scenario  : Price ${test_p_high:.2f}  ->  Demand: {volume_high:.2f} units ({((volume_high-10000)/10000)*100:.2f}%)")
    print(f"Discount Scenario : Price ${test_p_low:.2f}  ->  Demand: {volume_low:.2f} units (+{((volume_low-10000)/10000)*100:.2f}%)")
    print("="*45 + "\n")

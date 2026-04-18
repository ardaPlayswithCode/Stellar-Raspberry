"""03_Risk_Manager.py (The CFO)
This file sits between your strategy and your bank account. It calculates the actual dollar amount for every trade.

Balance Auditing: It pings the broker to see your Live Equity ($10,321) and calculates your "Deployable Capital" after your 20% Cash Reserve rule.

Diversification Check: It ensures no single ticker consumes too much of your buying power, preventing "All-in" scenarios that would violate prop firm consistency rules.

Output: It converts the "Position Size %" from Scaling_Rules into a specific Number of Shares."""
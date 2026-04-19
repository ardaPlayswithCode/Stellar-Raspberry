"""01_Scaling_Rules.py (The Strategy)
This is the "Math Logic" file. It defines how you interact with the market based on the model’s confidence.

The 60/65 Rule: It houses the logic for the 60% Floor (full exit if confidence drops below 0.60) and the 65% Hysteresis (no re-entry until confidence clears 0.65 to avoid "chatter").

Position Weighting: It contains the linear scaling formula that determines if a stock gets a "Full Unit" (e.g., at 95% confidence) or a "Starter Unit" (e.g., at 70% confidence).

Purpose: To turn a raw probability into a specific Position Size Percentage."""
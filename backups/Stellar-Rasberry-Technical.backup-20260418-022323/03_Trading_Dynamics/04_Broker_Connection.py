"""04_Broker_Connection.py (The Hands)
This is the technical bridge to your API (Alpaca, IBKR, etc.). It handles the "dirty work" of talking to the exchange.

Order Type Logic: It contains the code to place Mid-point Limit Orders, ensuring you aren't paying the "spread tax" by buying at the Ask or selling at the Bid.

Status Checking: It reports back to the system whether an order was "Filled," "Partial," or "Cancelled."

Safety: It includes "Error Handling" to ensure a lost internet connection doesn't leave an order hanging."""
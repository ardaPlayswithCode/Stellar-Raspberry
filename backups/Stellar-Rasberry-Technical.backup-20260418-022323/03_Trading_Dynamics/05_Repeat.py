"""05_Repeat.py (The Heartbeat)
This is a simple but vital "Wrapper" script that keeps the entire system alive.

The 60-Second Pulse: It uses a loop to trigger 02_go.py exactly once every minute.

Crash Recovery: If go.py crashes due to a weird data error, Repeat.py logs the error, waits for the next minute, and re-launches the process automatically.

Purpose: Total automation. It allows the system to run on your laptop or a server without you having to manually click "Run" every time."""
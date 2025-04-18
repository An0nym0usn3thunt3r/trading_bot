#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Run the trading bot in backtest mode
python src/run_trading_bot.py --mode backtest --symbol NQ --start-date 2025-01-01 --end-date 2025-04-01

#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Run the trading bot in paper trading mode
python src/run_trading_bot.py --mode paper --symbol NQ

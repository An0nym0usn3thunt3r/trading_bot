#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Run the trading bot in live trading mode
python src/run_trading_bot.py --mode live --symbol NQ

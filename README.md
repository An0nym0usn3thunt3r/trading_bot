# Nasdaq-100 E-mini Futures Trading Bot

A comprehensive algorithmic trading system for day trading Nasdaq-100 E-mini futures (NQ) with multi-strategy capabilities, advanced AI/ML models, and robust risk management.

## Overview

This trading bot is designed specifically for the Nasdaq-100 E-mini futures market, with a focus on the 10-minute trading window. It incorporates multiple trading strategies, advanced AI/ML models, comprehensive risk management, and efficient execution capabilities.

### Key Features

- **Multi-Strategy Trading Engine**: 25+ trading strategies across different categories
- **Advanced AI/ML Implementation**: Ensemble deep learning and reinforcement learning models
- **Time-Optimized Trading**: Specifically designed for a 10-minute trading window
- **Multi-Timeframe Analysis**: Analyzes market data across multiple timeframes
- **Comprehensive Position Management**: Sophisticated position sizing and management
- **Market-Specific Logic**: Tailored for NQ E-mini futures characteristics
- **Execution Optimization**: Smart order routing and execution algorithms
- **Extensive Risk Management**: Multiple layers of risk controls
- **Broker Integration**: Support for Interactive Brokers and TD Ameritrade
- **Monitoring and Reporting**: Real-time performance monitoring and detailed reporting

## Installation

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment (recommended)

### Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd nq_trading_bot
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Configure the trading bot (see Configuration section)

5. Run the trading bot (see Usage section)

## Configuration

The trading bot can be configured using a JSON configuration file. A default configuration is provided, but you can customize it to suit your needs.

### Configuration File Structure

```json
{
  "mode": "backtest",
  "symbol": "NQ",
  "timeframe": "1m",
  "start_date": "2025-01-01",
  "end_date": "2025-04-01",
  "initial_balance": 100000.0,
  "data_dir": "data",
  "strategies": {
    "mean_reversion": {
      "enabled": true,
      "allocation": 0.3,
      "params": {
        "bollinger_bands": {
          "enabled": true,
          "window": 20,
          "num_std": 2.0
        },
        "rsi": {
          "enabled": true,
          "window": 14,
          "overbought": 70,
          "oversold": 30
        }
      }
    },
    "momentum": {
      "enabled": true,
      "allocation": 0.3,
      "params": {
        "moving_average_crossover": {
          "enabled": true,
          "fast_window": 10,
          "slow_window": 30
        },
        "macd": {
          "enabled": true,
          "fast_window": 12,
          "slow_window": 26,
          "signal_window": 9
        }
      }
    },
    "breakout": {
      "enabled": true,
      "allocation": 0.2,
      "params": {
        "donchian_channel": {
          "enabled": true,
          "window": 20
        },
        "volatility_breakout": {
          "enabled": true,
          "window": 20,
          "multiplier": 2.0
        }
      }
    },
    "pattern_recognition": {
      "enabled": true,
      "allocation": 0.1,
      "params": {
        "candlestick_pattern": {
          "enabled": true,
          "patterns": ["hammer", "engulfing", "doji"]
        },
        "chart_pattern": {
          "enabled": true,
          "patterns": ["head_and_shoulders", "double_top", "double_bottom"]
        }
      }
    },
    "microstructure": {
      "enabled": true,
      "allocation": 0.1,
      "params": {
        "volume_profile": {
          "enabled": true,
          "window": 20
        },
        "order_flow": {
          "enabled": true,
          "window": 20
        }
      }
    }
  },
  "models": {
    "enabled": true,
    "allocation": 0.2,
    "params": {
      "lstm": {
        "enabled": true,
        "units": 50,
        "dropout": 0.2,
        "window": 20
      },
      "gru": {
        "enabled": true,
        "units": 50,
        "dropout": 0.2,
        "window": 20
      },
      "dqn": {
        "enabled": true,
        "window": 20
      }
    }
  },
  "risk": {
    "max_position_size": 5,
    "max_daily_loss": 1000.0,
    "max_drawdown": 0.05,
    "max_trades_per_day": 20,
    "volatility_scaling": true,
    "correlation_risk_control": true,
    "black_swan_protection": true
  },
  "execution": {
    "broker": "simulated",
    "commission_rate": 0.0001,
    "slippage": 0.0001,
    "smart_routing": true,
    "execution_algo": "twap",
    "broker_settings": {
      "interactive_brokers": {
        "host": "127.0.0.1",
        "port": 7496,
        "client_id": 1
      },
      "td_ameritrade": {
        "api_key": "",
        "redirect_uri": "",
        "token_path": ""
      }
    }
  },
  "monitoring": {
    "enabled": true,
    "dashboard_port": 8080,
    "alert_channels": ["log"],
    "report_schedule": {
      "daily": "17:00",
      "weekly": "Friday 17:00",
      "monthly": "1 17:00"
    }
  },
  "logging": {
    "level": "INFO",
    "file": "trading_bot.log"
  }
}
```

Save your custom configuration to `config/config.json`.

## Usage

The trading bot can be run in three modes:

1. **Backtest Mode**: Test strategies on historical data
2. **Paper Trading Mode**: Simulate trading with real-time data but no real money
3. **Live Trading Mode**: Trade with real money on a live account

### Running in Backtest Mode

```bash
./run_backtest.sh
```

Or with custom parameters:

```bash
python src/run_trading_bot.py --mode backtest --symbol NQ --start-date 2025-01-01 --end-date 2025-04-01 --config config/config.json
```

### Running in Paper Trading Mode

```bash
./run_paper_trading.sh
```

Or with custom parameters:

```bash
python src/run_trading_bot.py --mode paper --symbol NQ --config config/config.json
```

### Running in Live Trading Mode

```bash
./run_live_trading.sh
```

Or with custom parameters:

```bash
python src/run_trading_bot.py --mode live --symbol NQ --config config/config.json
```

## Using Historical Data

The trading bot can use your historical data in CSV format. The expected format is:

```
timestamp,open,high,low,close,volume
2025-03-23 18:00:00,20050.5,20072.5,20050.5,20066.75,1215
2025-03-23 18:01:00,20068.0,20099.0,20066.25,20096.0,947
2025-03-23 18:02:00,20096.0,20102.5,20085.5,20102.5,603
```

To use your historical data:

1. Place your CSV file in the `data` directory
2. Update the configuration file to point to your data file
3. Run the trading bot in backtest mode

Example configuration for custom data:

```json
{
  "mode": "backtest",
  "symbol": "NQ",
  "timeframe": "1m",
  "data_file": "data/your_historical_data.csv",
  "initial_balance": 100000.0
}
```

## Project Structure

```
nq_trading_bot/
├── config/                 # Configuration files
├── data/                   # Historical and market data
├── docs/                   # Documentation
├── logs/                   # Log files
├── reports/                # Generated reports
├── src/                    # Source code
│   ├── data/               # Data handling modules
│   ├── strategies/         # Trading strategies
│   ├── models/             # AI/ML models
│   ├── risk/               # Risk management
│   ├── execution/          # Order execution
│   ├── monitoring/         # Monitoring and reporting
│   ├── run_backtest.py     # Backtesting script
│   └── run_trading_bot.py  # Main trading bot script
├── tests/                  # Unit and integration tests
├── venv/                   # Virtual environment
├── requirements.txt        # Python dependencies
├── run_backtest.sh         # Backtest runner script
├── run_paper_trading.sh    # Paper trading runner script
└── run_live_trading.sh     # Live trading runner script
```

## Trading Strategies

The trading bot includes multiple strategy categories:

### Mean Reversion Strategies
- Bollinger Bands
- RSI Mean Reversion
- Statistical Mean Reversion
- MACD Mean Reversion

### Momentum Strategies
- Moving Average Crossover
- MACD Momentum
- RSI Momentum
- ADX Trend Following

### Breakout Strategies
- Donchian Channel Breakout
- Volatility Breakout
- Range Breakout
- Support/Resistance Breakout

### Pattern Recognition Strategies
- Candlestick Patterns
- Chart Patterns
- Harmonic Patterns
- Price Action Patterns

### Market Microstructure Strategies
- Volume Profile
- Order Flow Analysis
- Market Depth Analysis
- Liquidity Analysis

## AI/ML Models

The trading bot incorporates advanced AI/ML models:

### Traditional ML Models
- Random Forest
- Gradient Boosting
- Support Vector Machines
- K-Nearest Neighbors

### Deep Learning Models
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Deep Neural Networks
- Convolutional Neural Networks

### Reinforcement Learning Models
- Deep Q-Network (DQN)
- Advantage Actor-Critic (A2C)
- Proximal Policy Optimization (PPO)
- Deep Deterministic Policy Gradient (DDPG)

### Ensemble Approaches
- Voting Ensemble
- Stacking Ensemble
- Boosting Ensemble
- Bagging Ensemble

## Risk Management

The trading bot includes comprehensive risk management:

- Position Size Control
- Drawdown Control
- Daily Loss Limits
- Trade Frequency Control
- Volatility-Based Position Sizing
- Overnight Risk Control
- Strategy Allocation Control
- Correlation Risk Management
- Black Swan Protection

## Broker Integration

The trading bot supports multiple brokers:

- Interactive Brokers
- TD Ameritrade
- Simulated Broker (for testing)

## Monitoring and Reporting

The trading bot includes comprehensive monitoring and reporting:

- Real-Time Performance Monitoring
- System Health Monitoring
- Alert Management
- Report Generation
- Log Management
- Dashboard Visualization

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This trading bot is provided for educational and informational purposes only. Trading financial instruments involves risk, and past performance is not indicative of future results. Use this software at your own risk.

"""
Main entry point for the Nasdaq-100 E-mini Futures Trading Bot.
This module initializes and runs the complete trading system.
"""

import os
import sys
import logging
import argparse
import json
import datetime
import time
import threading
import signal
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import components
from data.data_loader import DataLoader
from strategies.base_strategy import StrategyManager
from strategies.mean_reversion import BollingerBandsStrategy, RSIMeanReversionStrategy
from strategies.momentum import MovingAverageCrossoverStrategy, MACDStrategy
from strategies.breakout import DonchianChannelStrategy, VolatilityBreakoutStrategy
from strategies.pattern_recognition import CandlestickPatternStrategy, ChartPatternStrategy
from strategies.microstructure import VolumeProfileStrategy, OrderFlowStrategy
from models.base_models import ModelManager
from models.deep_learning import LSTMModel, GRUModel
from models.reinforcement_learning import DQNAgent
from risk.risk_management import RiskManager, RiskMonitor
from execution.execution import (
    SimulatedBroker, InteractiveBrokersBroker, TDAmeritradeBroker,
    ExecutionHandler, OrderManager, SmartOrderRouter
)
from monitoring.monitoring import MonitoringSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingBot:
    """
    Main trading bot class that coordinates all components.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the trading bot.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to the configuration file.
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize state
        self.is_running = False
        self.trading_thread = None
        self.shutdown_event = threading.Event()
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Trading bot initialized")
    
    def _load_config(self, config_path):
        """
        Load configuration from file.
        
        Parameters:
        -----------
        config_path : str
            Path to the configuration file.
        
        Returns:
        --------
        dict
            Trading bot configuration.
        """
        default_config = {
            'mode': 'backtest',  # 'backtest', 'paper', 'live'
            'symbol': 'NQ',  # Nasdaq-100 E-mini futures
            'timeframe': '1m',  # 1-minute timeframe
            'start_date': (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d'),
            'end_date': datetime.datetime.now().strftime('%Y-%m-%d'),
            'initial_balance': 100000.0,
            'data_dir': 'data',
            'strategies': {
                'mean_reversion': {
                    'enabled': True,
                    'allocation': 0.3,
                    'params': {
                        'bollinger_bands': {
                            'enabled': True,
                            'window': 20,
                            'num_std': 2.0
                        },
                        'rsi': {
                            'enabled': True,
                            'window': 14,
                            'overbought': 70,
                            'oversold': 30
                        }
                    }
                },
                'momentum': {
                    'enabled': True,
                    'allocation': 0.3,
                    'params': {
                        'moving_average_crossover': {
                            'enabled': True,
                            'fast_window': 10,
                            'slow_window': 30
                        },
                        'macd': {
                            'enabled': True,
                            'fast_window': 12,
                            'slow_window': 26,
                            'signal_window': 9
                        }
                    }
                },
                'breakout': {
                    'enabled': True,
                    'allocation': 0.2,
                    'params': {
                        'donchian_channel': {
                            'enabled': True,
                            'window': 20
                        },
                        'volatility_breakout': {
                            'enabled': True,
                            'window': 20,
                            'multiplier': 2.0
                        }
                    }
                },
                'pattern_recognition': {
                    'enabled': True,
                    'allocation': 0.1,
                    'params': {
                        'candlestick_pattern': {
                            'enabled': True,
                            'patterns': ['hammer', 'engulfing', 'doji']
                        },
                        'chart_pattern': {
                            'enabled': True,
                            'patterns': ['head_and_shoulders', 'double_top', 'double_bottom']
                        }
                    }
                },
                'microstructure': {
                    'enabled': True,
                    'allocation': 0.1,
                    'params': {
                        'volume_profile': {
                            'enabled': True,
                            'window': 20
                        },
                        'order_flow': {
                            'enabled': True,
                            'window': 20
                        }
                    }
                }
            },
            'models': {
                'enabled': True,
                'allocation': 0.2,
                'params': {
                    'lstm': {
                        'enabled': True,
                        'units': 50,
                        'dropout': 0.2,
                        'window': 20
                    },
                    'gru': {
                        'enabled': True,
                        'units': 50,
                        'dropout': 0.2,
                        'window': 20
                    },
                    'dqn': {
                        'enabled': True,
                        'window': 20
                    }
                }
            },
            'risk': {
                'max_position_size': 5,
                'max_daily_loss': 1000.0,
                'max_drawdown': 0.05,
                'max_trades_per_day': 20,
                'volatility_scaling': True,
                'correlation_risk_control': True,
                'black_swan_protection': True
            },
            'execution': {
                'broker': 'simulated',  # 'simulated', 'interactive_brokers', 'td_ameritrade'
                'commission_rate': 0.0001,  # 0.01%
                'slippage': 0.0001,  # 0.01%
                'smart_routing': True,
                'execution_algo': 'twap',  # 'twap', 'vwap', 'market'
                'broker_settings': {
                    'interactive_brokers': {
                        'host': '127.0.0.1',
                        'port': 7496,
                        'client_id': 1
                    },
                    'td_ameritrade': {
                        'api_key': '',
                        'redirect_uri': '',
                        'token_path': ''
                    }
                }
            },
            'monitoring': {
                'enabled': True,
                'dashboard_port': 8080,
                'alert_channels': ['log'],
                'report_schedule': {
                    'daily': '17:00',
                    'weekly': 'Friday 17:00',
                    'monthly': '1 17:00'
                }
            },
            'logging': {
                'level': 'INFO',
                'file': 'trading_bot.log'
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Merge with default config
                for key, value in config.items():
                    if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
                
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        return default_config
    
    def _initialize_components(self):
        """Initialize trading bot components."""
        # Initialize data loader
        self.data_loader = DataLoader(
            data_dir=self.config.get('data_dir', 'data'),
            default_timeframe=self.config.get('timeframe', '1m')
        )
        
        # Initialize broker
        broker_type = self.config.get('execution', {}).get('broker', 'simulated')
        broker_settings = self.config.get('execution', {}).get('broker_settings', {})
        
        if broker_type == 'interactive_brokers':
            self.broker = InteractiveBrokersBroker(broker_settings.get('interactive_brokers', {}))
        elif broker_type == 'td_ameritrade':
            self.broker = TDAmeritradeBroker(broker_settings.get('td_ameritrade', {}))
        else:
            # Default to simulated broker
            self.broker = SimulatedBroker({
                'initial_balance': self.config.get('initial_balance', 100000.0),
                'commission_rate': self.config.get('execution', {}).get('commission_rate', 0.0001),
                'slippage': self.config.get('execution', {}).get('slippage', 0.0001)
            })
        
        # Initialize risk manager
        self.risk_manager = RiskManager({
            'initial_balance': self.config.get('initial_balance', 100000.0),
            'max_position_size': self.config.get('risk', {}).get('max_position_size', 5),
            'max_daily_loss': self.config.get('risk', {}).get('max_daily_loss', 1000.0),
            'max_drawdown': self.config.get('risk', {}).get('max_drawdown', 0.05),
            'max_trades_per_day': self.config.get('risk', {}).get('max_trades_per_day', 20),
            'volatility_scaling': self.config.get('risk', {}).get('volatility_scaling', True),
            'correlation_risk_control': self.config.get('risk', {}).get('correlation_risk_control', True),
            'black_swan_protection': self.config.get('risk', {}).get('black_swan_protection', True),
            'strategy_allocations': {
                'mean_reversion': self.config.get('strategies', {}).get('mean_reversion', {}).get('allocation', 0.3),
                'momentum': self.config.get('strategies', {}).get('momentum', {}).get('allocation', 0.3),
                'breakout': self.config.get('strategies', {}).get('breakout', {}).get('allocation', 0.2),
                'pattern_recognition': self.config.get('strategies', {}).get('pattern_recognition', {}).get('allocation', 0.1),
                'microstructure': self.config.get('strategies', {}).get('microstructure', {}).get('allocation', 0.1)
            }
        })
        
        # Initialize risk monitor
        self.risk_monitor = RiskMonitor(self.risk_manager, log_dir='logs/risk')
        
        # Initialize execution handler
        self.execution_handler = ExecutionHandler(self.broker, self.risk_manager)
        
        # Initialize order manager
        self.order_manager = OrderManager(self.execution_handler)
        
        # Initialize smart order router
        self.smart_router = SmartOrderRouter(self.execution_handler)
        
        # Initialize strategy manager
        self.strategy_manager = StrategyManager()
        
        # Initialize model manager
        self.model_manager = ModelManager()
        
        # Initialize monitoring system
        self.monitoring_system = MonitoringSystem({
            'monitoring_interval': 60,
            'data_dir': 'logs',
            'alert_channels': self.config.get('monitoring', {}).get('alert_channels', ['log']),
            'report_schedule': self.config.get('monitoring', {}).get('report_schedule', {})
        })
        
        # Set performance monitor in report generator
        self.monitoring_system.report_generator.set_performance_monitor(self.monitoring_system.performance_monitor)
        
        # Initialize dashboard server if enabled
        if self.config.get('monitoring', {}).get('enabled', True):
            dashboard_port = self.config.get('monitoring', {}).get('dashboard_port', 8080)
            self.dashboard_server = self.monitoring_system.DashboardServer(self.monitoring_system, port=dashboard_port)
        else:
            self.dashboard_server = None
    
    def _initialize_strategies(self):
        """Initialize trading strategies."""
        # Mean reversion strategies
        if self.config.get('strategies', {}).get('mean_reversion', {}).get('enabled', True):
            mean_reversion_params = self.config.get('strategies', {}).get('mean_reversion', {}).get('params', {})
            
            # Bollinger Bands strategy
            if mean_reversion_params.get('bollinger_bands', {}).get('enabled', True):
                bb_params = mean_reversion_params.get('bollinger_bands', {})
                bb_strategy = BollingerBandsStrategy(
                    window=bb_params.get('window', 20),
                    num_std=bb_params.get('num_std', 2.0)
                )
                self.strategy_manager.add_strategy('bollinger_bands', bb_strategy)
            
            # RSI Mean Reversion strategy
            if mean_reversion_params.get('rsi', {}).get('enabled', True):
                rsi_params = mean_reversion_params.get('rsi', {})
                rsi_strategy = RSIMeanReversionStrategy(
                    window=rsi_params.get('window', 14),
                    overbought=rsi_params.get('overbought', 70),
                    oversold=rsi_params.get('oversold', 30)
                )
                self.strategy_manager.add_strategy('rsi_mean_reversion', rsi_strategy)
        
        # Momentum strategies
        if self.config.get('strategies', {}).get('momentum', {}).get('enabled', True):
            momentum_params = self.config.get('strategies', {}).get('momentum', {}).get('params', {})
            
            # Moving Average Crossover strategy
            if momentum_params.get('moving_average_crossover', {}).get('enabled', True):
                ma_params = momentum_params.get('moving_average_crossover', {})
                ma_strategy = MovingAverageCrossoverStrategy(
                    fast_window=ma_params.get('fast_window', 10),
                    slow_window=ma_params.get('slow_window', 30)
                )
                self.strategy_manager.add_strategy('moving_average_crossover', ma_strategy)
            
            # MACD strategy
            if momentum_params.get('macd', {}).get('enabled', True):
                macd_params = momentum_params.get('macd', {})
                macd_strategy = MACDStrategy(
                    fast_window=macd_params.get('fast_window', 12),
                    slow_window=macd_params.get('slow_window', 26),
                    signal_window=macd_params.get('signal_window', 9)
                )
                self.strategy_manager.add_strategy('macd', macd_strategy)
        
        # Breakout strategies
        if self.config.get('strategies', {}).get('breakout', {}).get('enabled', True):
            breakout_params = self.config.get('strategies', {}).get('breakout', {}).get('params', {})
            
            # Donchian Channel strategy
            if breakout_params.get('donchian_channel', {}).get('enabled', True):
                dc_params = breakout_params.get('donchian_channel', {})
                dc_strategy = DonchianChannelStrategy(
                    window=dc_params.get('window', 20)
                )
                self.strategy_manager.add_strategy('donchian_channel', dc_strategy)
            
            # Volatility Breakout strategy
            if breakout_params.get('volatility_breakout', {}).get('enabled', True):
                vb_params = breakout_params.get('volatility_breakout', {})
                vb_strategy = VolatilityBreakoutStrategy(
                    window=vb_params.get('window', 20),
                    multiplier=vb_params.get('multiplier', 2.0)
                )
                self.strategy_manager.add_strategy('volatility_breakout', vb_strategy)
        
        # Pattern Recognition strategies
        if self.config.get('strategies', {}).get('pattern_recognition', {}).get('enabled', True):
            pattern_params = self.config.get('strategies', {}).get('pattern_recognition', {}).get('params', {})
            
            # Candlestick Pattern strategy
            if pattern_params.get('candlestick_pattern', {}).get('enabled', True):
                cp_params = pattern_params.get('candlestick_pattern', {})
                cp_strategy = CandlestickPatternStrategy(
                    patterns=cp_params.get('patterns', ['hammer', 'engulfing', 'doji'])
                )
                self.strategy_manager.add_strategy('candlestick_pattern', cp_strategy)
            
            # Chart Pattern strategy
            if pattern_params.get('chart_pattern', {}).get('enabled', True):
                chart_params = pattern_params.get('chart_pattern', {})
                chart_strategy = ChartPatternStrategy(
                    patterns=chart_params.get('patterns', ['head_and_shoulders', 'double_top', 'double_bottom'])
                )
                self.strategy_manager.add_strategy('chart_pattern', chart_strategy)
        
        # Microstructure strategies
        if self.config.get('strategies', {}).get('microstructure', {}).get('enabled', True):
            micro_params = self.config.get('strategies', {}).get('microstructure', {}).get('params', {})
            
            # Volume Profile strategy
            if micro_params.get('volume_profile', {}).get('enabled', True):
                vp_params = micro_params.get('volume_profile', {})
                vp_strategy = VolumeProfileStrategy(
                    window=vp_params.get('window', 20)
                )
                self.strategy_manager.add_strategy('volume_profile', vp_strategy)
            
            # Order Flow strategy
            if micro_params.get('order_flow', {}).get('enabled', True):
                of_params = micro_params.get('order_flow', {})
                of_strategy = OrderFlowStrategy(
                    window=of_params.get('window', 20)
                )
                self.strategy_manager.add_strategy('order_flow', of_strategy)
    
    def _initialize_models(self):
        """Initialize AI/ML models."""
        if self.config.get('models', {}).get('enabled', True):
            model_params = self.config.get('models', {}).get('params', {})
            
            # LSTM model
            if model_params.get('lstm', {}).get('enabled', True):
                lstm_params = model_params.get('lstm', {})
                lstm_model = LSTMModel(
                    units=lstm_params.get('units', 50),
                    dropout=lstm_params.get('dropout', 0.2),
                    window=lstm_params.get('window', 20)
                )
                self.model_manager.add_model('lstm', lstm_model)
            
            # GRU model
            if model_params.get('gru', {}).get('enabled', True):
                gru_params = model_params.get('gru', {})
                gru_model = GRUModel(
                    units=gru_params.get('units', 50),
                    dropout=gru_params.get('dropout', 0.2),
                    window=gru_params.get('window', 20)
                )
                self.model_manager.add_model('gru', gru_model)
            
            # DQN model
            if model_params.get('dqn', {}).get('enabled', True):
                dqn_params = model_params.get('dqn', {})
                dqn_model = DQNAgent(
                    window=dqn_params.get('window', 20)
                )
                self.model_manager.add_model('dqn', dqn_model)
    
    def start(self):
        """Start the trading bot."""
        if self.is_running:
            logger.warning("Trading bot already running")
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        # Connect to broker
        logger.info("Connecting to broker...")
        self.broker.connect()
        
        # Initialize strategies
        logger.info("Initializing strategies...")
        self._initialize_strategies()
        
        # Initialize models
        logger.info("Initializing models...")
        self._initialize_models()
        
        # Start monitoring system
        logger.info("Starting monitoring system...")
        self.monitoring_system.start()
        
        # Start dashboard server if enabled
        if self.dashboard_server:
            logger.info("Starting dashboard server...")
            self.dashboard_server.start()
        
        # Start trading thread
        logger.info("Starting trading thread...")
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Trading bot started")
    
    def stop(self):
        """Stop the trading bot."""
        if not self.is_running:
            logger.warning("Trading bot not running")
            return
        
        logger.info("Stopping trading bot...")
        
        # Set shutdown event
        self.shutdown_event.set()
        
        # Wait for trading thread to finish
        if self.trading_thread:
            self.trading_thread.join(timeout=10.0)
        
        # Stop dashboard server
        if self.dashboard_server:
            logger.info("Stopping dashboard server...")
            self.dashboard_server.stop()
        
        # Stop monitoring system
        logger.info("Stopping monitoring system...")
        self.monitoring_system.stop()
        
        # Disconnect from broker
        logger.info("Disconnecting from broker...")
        self.broker.disconnect()
        
        self.is_running = False
        logger.info("Trading bot stopped")
    
    def _signal_handler(self, sig, frame):
        """
        Handle signals.
        
        Parameters:
        -----------
        sig : int
            Signal number.
        frame : frame
            Current stack frame.
        """
        logger.info(f"Received signal {sig}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def _trading_loop(self):
        """Main trading loop."""
        mode = self.config.get('mode', 'backtest')
        
        if mode == 'backtest':
            self._run_backtest()
        else:
            self._run_live_trading()
    
    def _run_backtest(self):
        """Run backtesting."""
        logger.info("Running backtesting...")
        
        # Get configuration
        symbol = self.config.get('symbol', 'NQ')
        timeframe = self.config.get('timeframe', '1m')
        start_date = self.config.get('start_date')
        end_date = self.config.get('end_date')
        
        # Load historical data
        logger.info(f"Loading historical data for {symbol} from {start_date} to {end_date}...")
        data = self.data_loader.load_historical_data(symbol, timeframe, start_date, end_date)
        
        if data is None or len(data) == 0:
            logger.error("No historical data available")
            return
        
        logger.info(f"Loaded {len(data)} bars of historical data")
        
        # Initialize backtest state
        initial_balance = self.config.get('initial_balance', 100000.0)
        current_position = 0
        equity = [initial_balance]
        trades = []
        
        # Run backtest
        for i in range(len(data) - 1):
            # Check if shutdown requested
            if self.shutdown_event.is_set():
                logger.info("Shutdown requested, stopping backtest")
                break
            
            # Get current and next bar
            current_bar = data.iloc[i]
            next_bar = data.iloc[i + 1]
            
            # Prepare market data for strategies
            market_data = {
                'symbol': symbol,
                'timestamp': current_bar.name,
                'open': current_bar['open'],
                'high': current_bar['high'],
                'low': current_bar['low'],
                'close': current_bar['close'],
                'volume': current_bar['volume'],
                'position': current_position
            }
            
            # Update broker market data
            if isinstance(self.broker, SimulatedBroker):
                self.broker.update_market_data(symbol, {
                    'price': current_bar['close'],
                    'timestamp': current_bar.name
                })
            
            # Get signals from strategies
            signals = self.strategy_manager.generate_signals(market_data, data.iloc[:i+1])
            
            # Get predictions from models
            predictions = self.model_manager.generate_predictions(market_data, data.iloc[:i+1])
            
            # Combine signals and predictions
            combined_signal = self._combine_signals(signals, predictions)
            
            # Execute trading logic
            if combined_signal > 0 and current_position <= 0:
                # Buy signal
                quantity = 1
                
                # Create order
                order = self.order_manager.create_market_order(
                    symbol=symbol,
                    side='buy',
                    quantity=quantity
                )
                
                # Submit order
                self.order_manager.submit_order(order)
                
                # Update position
                current_position += quantity
                
                # Record trade
                trade = {
                    'timestamp': current_bar.name,
                    'symbol': symbol,
                    'side': 'buy',
                    'quantity': quantity,
                    'price': next_bar['open'],
                    'pnl': 0.0
                }
                trades.append(trade)
                
                logger.info(f"BUY {quantity} {symbol} at {next_bar['open']}")
            
            elif combined_signal < 0 and current_position >= 0:
                # Sell signal
                quantity = 1
                
                # Create order
                order = self.order_manager.create_market_order(
                    symbol=symbol,
                    side='sell',
                    quantity=quantity
                )
                
                # Submit order
                self.order_manager.submit_order(order)
                
                # Update position
                current_position -= quantity
                
                # Record trade
                trade = {
                    'timestamp': current_bar.name,
                    'symbol': symbol,
                    'side': 'sell',
                    'quantity': quantity,
                    'price': next_bar['open'],
                    'pnl': 0.0
                }
                trades.append(trade)
                
                logger.info(f"SELL {quantity} {symbol} at {next_bar['open']}")
            
            # Calculate equity
            if current_position != 0:
                position_value = current_position * next_bar['close']
                unrealized_pnl = position_value - sum(t['price'] * (1 if t['side'] == 'buy' else -1) * t['quantity'] for t in trades if t['pnl'] == 0.0)
            else:
                unrealized_pnl = 0.0
            
            current_equity = initial_balance + sum(t['pnl'] for t in trades if t['pnl'] != 0.0) + unrealized_pnl
            equity.append(current_equity)
            
            # Update monitoring system
            self.monitoring_system.update_trading_data({
                'equity': current_equity,
                'position': current_position,
                'timestamp': current_bar.name
            })
        
        # Calculate backtest results
        final_equity = equity[-1]
        total_return = (final_equity - initial_balance) / initial_balance
        
        # Calculate drawdown
        max_equity = initial_balance
        max_drawdown = 0.0
        
        for eq in equity:
            if eq > max_equity:
                max_equity = eq
            
            drawdown = (max_equity - eq) / max_equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Calculate Sharpe ratio
        returns = [(equity[i] - equity[i-1]) / equity[i-1] for i in range(1, len(equity))]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
        
        # Print backtest results
        logger.info("Backtest completed")
        logger.info(f"Initial balance: ${initial_balance:.2f}")
        logger.info(f"Final equity: ${final_equity:.2f}")
        logger.info(f"Total return: {total_return:.2%}")
        logger.info(f"Max drawdown: {max_drawdown:.2%}")
        logger.info(f"Sharpe ratio: {sharpe_ratio:.2f}")
        logger.info(f"Total trades: {len(trades)}")
        
        # Generate backtest report
        self._generate_backtest_report(equity, trades, data)
    
    def _run_live_trading(self):
        """Run live trading."""
        logger.info("Running live trading...")
        
        # Get configuration
        symbol = self.config.get('symbol', 'NQ')
        timeframe = self.config.get('timeframe', '1m')
        
        # Initialize data buffer
        data_buffer = []
        last_bar_time = None
        
        # Main trading loop
        while not self.shutdown_event.is_set():
            try:
                # Get current time
                current_time = datetime.datetime.now()
                
                # Check if market is open
                if not self._is_market_open(current_time):
                    logger.info("Market is closed, waiting...")
                    time.sleep(60)
                    continue
                
                # Get latest market data
                latest_data = self._get_latest_market_data(symbol)
                
                if latest_data is None:
                    logger.warning("No market data available")
                    time.sleep(5)
                    continue
                
                # Check if we have a new bar
                bar_time = self._get_bar_time(current_time, timeframe)
                
                if bar_time != last_bar_time and last_bar_time is not None:
                    # We have a new bar, process the previous one
                    logger.info(f"Processing bar at {last_bar_time}")
                    
                    # Create OHLCV bar
                    if data_buffer:
                        open_price = data_buffer[0]['price']
                        high_price = max(d['price'] for d in data_buffer)
                        low_price = min(d['price'] for d in data_buffer)
                        close_price = data_buffer[-1]['price']
                        volume = sum(d['volume'] for d in data_buffer if 'volume' in d)
                        
                        bar = pd.Series({
                            'open': open_price,
                            'high': high_price,
                            'low': low_price,
                            'close': close_price,
                            'volume': volume
                        }, name=last_bar_time)
                        
                        # Process bar
                        self._process_bar(symbol, bar)
                    
                    # Reset data buffer
                    data_buffer = []
                
                # Add latest data to buffer
                data_buffer.append(latest_data)
                last_bar_time = bar_time
                
                # Sleep until next update
                time.sleep(1)
            
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(5)
    
    def _is_market_open(self, current_time):
        """
        Check if the market is open.
        
        Parameters:
        -----------
        current_time : datetime.datetime
            Current time.
        
        Returns:
        --------
        bool
            True if the market is open, False otherwise.
        """
        # Simplified check for NQ E-mini futures
        # NQ trades Sunday-Friday, 6:00 PM - 5:00 PM ET the next day
        # with a trading halt from 4:15 PM - 4:30 PM ET
        
        # Convert to Eastern Time
        et_time = current_time - datetime.timedelta(hours=4)  # Simplified, doesn't account for DST
        
        # Check day of week (0 = Monday, 6 = Sunday)
        day_of_week = et_time.weekday()
        
        if day_of_week == 5:  # Saturday
            return False
        
        # Check time of day
        hour = et_time.hour
        minute = et_time.minute
        
        if day_of_week == 6:  # Sunday
            # Market opens at 6:00 PM ET on Sunday
            return hour >= 18
        
        if day_of_week == 4:  # Friday
            # Market closes at 5:00 PM ET on Friday
            return hour < 17
        
        # Check trading halt (4:15 PM - 4:30 PM ET)
        if hour == 16:
            return minute < 15 or minute >= 30
        
        return True
    
    def _get_latest_market_data(self, symbol):
        """
        Get latest market data.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol.
        
        Returns:
        --------
        dict
            Latest market data.
        """
        # In a real implementation, this would get real-time market data
        # For now, return simulated data
        
        # Get current time
        current_time = datetime.datetime.now()
        
        # Generate random price and volume
        last_price = 100.0 + np.random.normal(0, 1)
        volume = np.random.randint(1, 100)
        
        return {
            'symbol': symbol,
            'timestamp': current_time,
            'price': last_price,
            'volume': volume
        }
    
    def _get_bar_time(self, current_time, timeframe):
        """
        Get the bar time for a given timestamp and timeframe.
        
        Parameters:
        -----------
        current_time : datetime.datetime
            Current time.
        timeframe : str
            Timeframe.
        
        Returns:
        --------
        datetime.datetime
            Bar time.
        """
        if timeframe == '1m':
            return current_time.replace(second=0, microsecond=0)
        elif timeframe == '5m':
            minute = (current_time.minute // 5) * 5
            return current_time.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == '15m':
            minute = (current_time.minute // 15) * 15
            return current_time.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == '30m':
            minute = (current_time.minute // 30) * 30
            return current_time.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == '1h':
            return current_time.replace(minute=0, second=0, microsecond=0)
        elif timeframe == '4h':
            hour = (current_time.hour // 4) * 4
            return current_time.replace(hour=hour, minute=0, second=0, microsecond=0)
        elif timeframe == '1d':
            return current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            return current_time.replace(second=0, microsecond=0)
    
    def _process_bar(self, symbol, bar):
        """
        Process a new price bar.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol.
        bar : pd.Series
            Price bar.
        """
        # Get current position
        positions = self.broker.get_positions()
        current_position = positions.get(symbol, {}).get('quantity', 0)
        
        # Prepare market data for strategies
        market_data = {
            'symbol': symbol,
            'timestamp': bar.name,
            'open': bar['open'],
            'high': bar['high'],
            'low': bar['low'],
            'close': bar['close'],
            'volume': bar['volume'],
            'position': current_position
        }
        
        # Update broker market data
        if isinstance(self.broker, SimulatedBroker):
            self.broker.update_market_data(symbol, {
                'price': bar['close'],
                'timestamp': bar.name
            })
        
        # Get historical data
        historical_data = self.data_loader.get_data_buffer(symbol)
        
        if historical_data is None:
            historical_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        
        # Append new bar
        historical_data = historical_data.append(bar)
        
        # Update data buffer
        self.data_loader.update_data_buffer(symbol, historical_data)
        
        # Get signals from strategies
        signals = self.strategy_manager.generate_signals(market_data, historical_data)
        
        # Get predictions from models
        predictions = self.model_manager.generate_predictions(market_data, historical_data)
        
        # Combine signals and predictions
        combined_signal = self._combine_signals(signals, predictions)
        
        # Execute trading logic
        if combined_signal > 0 and current_position <= 0:
            # Buy signal
            quantity = 1
            
            # Create order
            order = self.order_manager.create_market_order(
                symbol=symbol,
                side='buy',
                quantity=quantity
            )
            
            # Submit order
            self.order_manager.submit_order(order)
            
            logger.info(f"BUY {quantity} {symbol} at market")
        
        elif combined_signal < 0 and current_position >= 0:
            # Sell signal
            quantity = 1
            
            # Create order
            order = self.order_manager.create_market_order(
                symbol=symbol,
                side='sell',
                quantity=quantity
            )
            
            # Submit order
            self.order_manager.submit_order(order)
            
            logger.info(f"SELL {quantity} {symbol} at market")
        
        # Update monitoring system
        account_info = self.broker.get_account_info()
        
        self.monitoring_system.update_trading_data({
            'equity': account_info.get('equity', 0.0),
            'position': current_position,
            'timestamp': bar.name
        })
    
    def _combine_signals(self, signals, predictions):
        """
        Combine signals from strategies and predictions from models.
        
        Parameters:
        -----------
        signals : dict
            Signals from strategies.
        predictions : dict
            Predictions from models.
        
        Returns:
        --------
        float
            Combined signal (-1.0 to 1.0).
        """
        # Get strategy allocations
        strategy_allocations = self.config.get('strategies', {})
        model_allocation = self.config.get('models', {}).get('allocation', 0.2)
        
        # Calculate weighted signal
        weighted_signal = 0.0
        total_weight = 0.0
        
        # Process strategy signals
        for strategy_type, signal_dict in signals.items():
            allocation = strategy_allocations.get(strategy_type, {}).get('allocation', 0.0)
            
            if allocation > 0:
                for strategy_name, signal in signal_dict.items():
                    weighted_signal += signal * allocation
                    total_weight += allocation
        
        # Process model predictions
        if model_allocation > 0:
            for model_name, prediction in predictions.items():
                weighted_signal += prediction * model_allocation
                total_weight += model_allocation
        
        # Normalize signal
        if total_weight > 0:
            combined_signal = weighted_signal / total_weight
        else:
            combined_signal = 0.0
        
        return combined_signal
    
    def _generate_backtest_report(self, equity, trades, data):
        """
        Generate backtest report.
        
        Parameters:
        -----------
        equity : list
            Equity curve.
        trades : list
            List of trades.
        data : pd.DataFrame
            Historical price data.
        """
        # Create report directory
        report_dir = os.path.join('reports', 'backtest', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(report_dir, exist_ok=True)
        
        # Save equity curve
        equity_df = pd.DataFrame({
            'equity': equity
        }, index=data.index[:len(equity)])
        
        equity_df.to_csv(os.path.join(report_dir, 'equity.csv'))
        
        # Save trades
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(os.path.join(report_dir, 'trades.csv'), index=False)
        
        # Generate equity curve chart
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df.index, equity_df['equity'])
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.savefig(os.path.join(report_dir, 'equity_curve.png'))
        plt.close()
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #0066cc;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .section {{
                    margin-bottom: 30px;
                }}
                .chart {{
                    max-width: 100%;
                    height: auto;
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <h1>Backtest Report</h1>
            <p>Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Performance Summary</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Initial Balance</td><td>${self.config.get('initial_balance', 100000.0):.2f}</td></tr>
                    <tr><td>Final Equity</td><td>${equity[-1]:.2f}</td></tr>
                    <tr><td>Total Return</td><td>{(equity[-1] - equity[0]) / equity[0]:.2%}</td></tr>
                    <tr><td>Total Trades</td><td>{len(trades)}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Equity Curve</h2>
                <img class="chart" src="equity_curve.png" alt="Equity Curve">
            </div>
            
            <div class="section">
                <h2>Recent Trades</h2>
                <table>
                    <tr>
                        <th>Timestamp</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Quantity</th>
                        <th>Price</th>
                        <th>P&L</th>
                    </tr>
        """
        
        # Add recent trades
        for trade in trades[-20:]:
            html_content += f"""
                    <tr>
                        <td>{trade['timestamp']}</td>
                        <td>{trade['symbol']}</td>
                        <td>{trade['side']}</td>
                        <td>{trade['quantity']}</td>
                        <td>{trade['price']:.2f}</td>
                        <td>{trade['pnl']:.2f}</td>
                    </tr>
            """
        
        # Close HTML
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        with open(os.path.join(report_dir, 'report.html'), 'w') as f:
            f.write(html_content)
        
        logger.info(f"Backtest report generated: {os.path.join(report_dir, 'report.html')}")


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Nasdaq-100 E-mini Futures Trading Bot')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['backtest', 'paper', 'live'], help='Trading mode')
    parser.add_argument('--symbol', type=str, help='Trading symbol')
    parser.add_argument('--start-date', type=str, help='Start date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for backtesting (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Create trading bot
    bot = TradingBot(args.config)
    
    # Override configuration with command line arguments
    if args.mode:
        bot.config['mode'] = args.mode
    
    if args.symbol:
        bot.config['symbol'] = args.symbol
    
    if args.start_date:
        bot.config['start_date'] = args.start_date
    
    if args.end_date:
        bot.config['end_date'] = args.end_date
    
    # Start trading bot
    try:
        bot.start()
        
        # Keep main thread alive
        while bot.is_running:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    
    finally:
        bot.stop()


if __name__ == "__main__":
    main()

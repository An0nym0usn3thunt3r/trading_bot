"""
Risk Management System for Nasdaq-100 E-mini futures trading bot.
This module implements various risk management components for controlling trading risk.
"""

import pandas as pd
import numpy as np
import logging
import os
import json
import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RiskManager:
    """
    Main risk management class that coordinates various risk controls.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the risk manager.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to the risk configuration file.
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize risk controls
        self.risk_controls = []
        self._initialize_risk_controls()
        
        # Initialize risk state
        self.risk_state = {
            'account_balance': self.config.get('initial_balance', 100000.0),
            'current_position': 0,
            'daily_pnl': 0.0,
            'max_daily_loss': self.config.get('max_daily_loss', 1000.0),
            'max_position_size': self.config.get('max_position_size', 5),
            'max_drawdown': self.config.get('max_drawdown', 0.05),
            'peak_balance': self.config.get('initial_balance', 100000.0),
            'trades_today': 0,
            'max_trades_per_day': self.config.get('max_trades_per_day', 20),
            'last_trade_time': None,
            'min_time_between_trades': self.config.get('min_time_between_trades', 60),  # seconds
            'strategy_allocations': {},
            'market_regime': 'normal',
            'volatility_multiplier': 1.0,
            'risk_level': 'normal',
            'overnight_position_limit': self.config.get('overnight_position_limit', 2),
            'is_trading_allowed': True,
            'trading_suspended_reason': None,
            'last_risk_check_time': datetime.datetime.now()
        }
        
        # Initialize strategy allocations
        for strategy, allocation in self.config.get('strategy_allocations', {}).items():
            self.risk_state['strategy_allocations'][strategy] = allocation
        
        logger.info("Risk Manager initialized")
    
    def _load_config(self, config_path):
        """
        Load risk configuration from file.
        
        Parameters:
        -----------
        config_path : str
            Path to the configuration file.
        
        Returns:
        --------
        dict
            Risk configuration.
        """
        default_config = {
            'initial_balance': 100000.0,
            'max_daily_loss': 1000.0,
            'max_position_size': 5,
            'max_drawdown': 0.05,
            'max_trades_per_day': 20,
            'min_time_between_trades': 60,  # seconds
            'strategy_allocations': {
                'mean_reversion': 0.3,
                'momentum': 0.3,
                'breakout': 0.2,
                'pattern_recognition': 0.1,
                'ml_prediction': 0.1
            },
            'overnight_position_limit': 2,
            'volatility_scaling': True,
            'correlation_risk_control': True,
            'black_swan_protection': True,
            'risk_controls': [
                'PositionSizeControl',
                'DrawdownControl',
                'DailyLossControl',
                'TradeFrequencyControl',
                'VolatilityControl',
                'OvernightRiskControl',
                'StrategyAllocationControl',
                'CorrelationRiskControl',
                'BlackSwanProtection'
            ]
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Merge with default config
                for key, value in config.items():
                    default_config[key] = value
                
                logger.info(f"Loaded risk configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading risk configuration: {e}")
        
        return default_config
    
    def _initialize_risk_controls(self):
        """Initialize risk control components based on configuration."""
        risk_control_classes = {
            'PositionSizeControl': PositionSizeControl,
            'DrawdownControl': DrawdownControl,
            'DailyLossControl': DailyLossControl,
            'TradeFrequencyControl': TradeFrequencyControl,
            'VolatilityControl': VolatilityControl,
            'OvernightRiskControl': OvernightRiskControl,
            'StrategyAllocationControl': StrategyAllocationControl,
            'CorrelationRiskControl': CorrelationRiskControl,
            'BlackSwanProtection': BlackSwanProtection
        }
        
        for control_name in self.config.get('risk_controls', []):
            if control_name in risk_control_classes:
                control_class = risk_control_classes[control_name]
                control = control_class(self.config)
                self.risk_controls.append(control)
                logger.info(f"Added risk control: {control_name}")
            else:
                logger.warning(f"Unknown risk control: {control_name}")
    
    def update_account_state(self, account_data):
        """
        Update account state with latest data.
        
        Parameters:
        -----------
        account_data : dict
            Dictionary containing account information.
        """
        # Update account balance
        if 'balance' in account_data:
            self.risk_state['account_balance'] = account_data['balance']
            
            # Update peak balance
            if account_data['balance'] > self.risk_state['peak_balance']:
                self.risk_state['peak_balance'] = account_data['balance']
        
        # Update current position
        if 'position' in account_data:
            self.risk_state['current_position'] = account_data['position']
        
        # Update daily P&L
        if 'daily_pnl' in account_data:
            self.risk_state['daily_pnl'] = account_data['daily_pnl']
        
        # Update trades today
        if 'trades_today' in account_data:
            self.risk_state['trades_today'] = account_data['trades_today']
        
        # Update last trade time
        if 'last_trade_time' in account_data:
            self.risk_state['last_trade_time'] = account_data['last_trade_time']
        
        logger.debug("Updated account state")
    
    def update_market_state(self, market_data):
        """
        Update market state with latest data.
        
        Parameters:
        -----------
        market_data : dict
            Dictionary containing market information.
        """
        # Update market regime
        if 'market_regime' in market_data:
            self.risk_state['market_regime'] = market_data['market_regime']
        
        # Update volatility multiplier
        if 'volatility_multiplier' in market_data:
            self.risk_state['volatility_multiplier'] = market_data['volatility_multiplier']
        
        logger.debug("Updated market state")
    
    def check_risk_limits(self, order_request=None):
        """
        Check if the proposed order violates any risk limits.
        
        Parameters:
        -----------
        order_request : dict, optional
            Dictionary containing order information.
        
        Returns:
        --------
        tuple
            (is_allowed, reason, modified_order)
        """
        # Update last risk check time
        self.risk_state['last_risk_check_time'] = datetime.datetime.now()
        
        # Check if trading is allowed
        if not self.risk_state['is_trading_allowed']:
            return (False, self.risk_state['trading_suspended_reason'], None)
        
        # Initialize modified order
        modified_order = order_request.copy() if order_request else None
        
        # Check each risk control
        for control in self.risk_controls:
            is_allowed, reason, modified_order = control.check(self.risk_state, modified_order)
            
            if not is_allowed:
                logger.warning(f"Risk limit violated: {reason}")
                return (False, reason, None)
        
        return (True, None, modified_order)
    
    def suspend_trading(self, reason):
        """
        Suspend all trading activity.
        
        Parameters:
        -----------
        reason : str
            Reason for suspending trading.
        """
        self.risk_state['is_trading_allowed'] = False
        self.risk_state['trading_suspended_reason'] = reason
        logger.warning(f"Trading suspended: {reason}")
    
    def resume_trading(self):
        """Resume trading activity."""
        self.risk_state['is_trading_allowed'] = True
        self.risk_state['trading_suspended_reason'] = None
        logger.info("Trading resumed")
    
    def adjust_position_size(self, original_size):
        """
        Adjust position size based on risk factors.
        
        Parameters:
        -----------
        original_size : int
            Original position size.
        
        Returns:
        --------
        int
            Adjusted position size.
        """
        adjusted_size = original_size
        
        # Apply volatility scaling
        if self.config.get('volatility_scaling', True):
            adjusted_size = int(adjusted_size * self.risk_state['volatility_multiplier'])
        
        # Apply risk level adjustment
        if self.risk_state['risk_level'] == 'low':
            adjusted_size = int(adjusted_size * 0.5)
        elif self.risk_state['risk_level'] == 'high':
            adjusted_size = int(adjusted_size * 1.5)
        
        # Ensure position size is within limits
        adjusted_size = min(adjusted_size, self.risk_state['max_position_size'])
        adjusted_size = max(adjusted_size, 0)
        
        logger.debug(f"Adjusted position size from {original_size} to {adjusted_size}")
        return adjusted_size
    
    def get_strategy_allocation(self, strategy_name):
        """
        Get the capital allocation for a specific strategy.
        
        Parameters:
        -----------
        strategy_name : str
            Name of the strategy.
        
        Returns:
        --------
        float
            Capital allocation for the strategy.
        """
        return self.risk_state['strategy_allocations'].get(strategy_name, 0.0)
    
    def set_risk_level(self, level):
        """
        Set the overall risk level.
        
        Parameters:
        -----------
        level : str
            Risk level ('low', 'normal', or 'high').
        """
        if level in ['low', 'normal', 'high']:
            self.risk_state['risk_level'] = level
            logger.info(f"Risk level set to {level}")
        else:
            logger.warning(f"Invalid risk level: {level}")
    
    def get_risk_report(self):
        """
        Generate a risk report.
        
        Returns:
        --------
        dict
            Risk report.
        """
        # Calculate drawdown
        current_drawdown = 0.0
        if self.risk_state['peak_balance'] > 0:
            current_drawdown = 1.0 - (self.risk_state['account_balance'] / self.risk_state['peak_balance'])
        
        report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'account_balance': self.risk_state['account_balance'],
            'current_position': self.risk_state['current_position'],
            'daily_pnl': self.risk_state['daily_pnl'],
            'trades_today': self.risk_state['trades_today'],
            'current_drawdown': current_drawdown,
            'max_drawdown_limit': self.risk_state['max_drawdown'],
            'max_daily_loss_limit': self.risk_state['max_daily_loss'],
            'max_position_size': self.risk_state['max_position_size'],
            'risk_level': self.risk_state['risk_level'],
            'market_regime': self.risk_state['market_regime'],
            'volatility_multiplier': self.risk_state['volatility_multiplier'],
            'is_trading_allowed': self.risk_state['is_trading_allowed'],
            'trading_suspended_reason': self.risk_state['trading_suspended_reason'],
            'strategy_allocations': self.risk_state['strategy_allocations']
        }
        
        return report
    
    def plot_risk_metrics(self, risk_history, save_path=None):
        """
        Plot risk metrics over time.
        
        Parameters:
        -----------
        risk_history : list
            List of risk reports.
        save_path : str, optional
            Path to save the plot. If None, the plot is displayed.
        
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure.
        """
        if not risk_history:
            logger.warning("No risk history to plot")
            return None
        
        # Extract data
        timestamps = [datetime.datetime.fromisoformat(report['timestamp']) for report in risk_history]
        balances = [report['account_balance'] for report in risk_history]
        drawdowns = [report['current_drawdown'] for report in risk_history]
        daily_pnls = [report['daily_pnl'] for report in risk_history]
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot account balance
        ax1.plot(timestamps, balances)
        ax1.set_title('Account Balance')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Balance')
        
        # Plot drawdown
        ax2.plot(timestamps, drawdowns)
        ax2.axhline(y=self.risk_state['max_drawdown'], color='r', linestyle='--', label='Max Drawdown Limit')
        ax2.set_title('Drawdown')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Drawdown')
        ax2.legend()
        
        # Plot daily P&L
        ax3.plot(timestamps, daily_pnls)
        ax3.axhline(y=-self.risk_state['max_daily_loss'], color='r', linestyle='--', label='Max Daily Loss Limit')
        ax3.set_title('Daily P&L')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('P&L')
        ax3.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved risk metrics plot to {save_path}")
        
        return fig


class RiskControl:
    """
    Base class for risk controls.
    """
    
    def __init__(self, config):
        """
        Initialize the risk control.
        
        Parameters:
        -----------
        config : dict
            Risk configuration.
        """
        self.config = config
    
    def check(self, risk_state, order_request):
        """
        Check if the proposed order violates risk limits.
        
        Parameters:
        -----------
        risk_state : dict
            Current risk state.
        order_request : dict
            Dictionary containing order information.
        
        Returns:
        --------
        tuple
            (is_allowed, reason, modified_order)
        """
        raise NotImplementedError("Subclasses must implement check method")


class PositionSizeControl(RiskControl):
    """
    Control for position size limits.
    """
    
    def check(self, risk_state, order_request):
        """
        Check if the proposed order violates position size limits.
        
        Parameters:
        -----------
        risk_state : dict
            Current risk state.
        order_request : dict
            Dictionary containing order information.
        
        Returns:
        --------
        tuple
            (is_allowed, reason, modified_order)
        """
        if not order_request:
            return (True, None, None)
        
        # Get current position and order size
        current_position = risk_state['current_position']
        order_size = order_request.get('size', 0)
        order_side = order_request.get('side', 'buy')
        
        # Calculate new position
        new_position = current_position
        if order_side == 'buy':
            new_position += order_size
        elif order_side == 'sell':
            new_position -= order_size
        
        # Check if new position exceeds limits
        if abs(new_position) > risk_state['max_position_size']:
            # Modify order to respect position limits
            modified_order = order_request.copy()
            
            if order_side == 'buy':
                max_allowed = risk_state['max_position_size'] - current_position
                modified_order['size'] = max(0, max_allowed)
            elif order_side == 'sell':
                max_allowed = risk_state['max_position_size'] + current_position
                modified_order['size'] = max(0, max_allowed)
            
            if modified_order['size'] == 0:
                return (False, "Position size limit exceeded", None)
            else:
                return (True, None, modified_order)
        
        return (True, None, order_request)


class DrawdownControl(RiskControl):
    """
    Control for drawdown limits.
    """
    
    def check(self, risk_state, order_request):
        """
        Check if the current drawdown exceeds limits.
        
        Parameters:
        -----------
        risk_state : dict
            Current risk state.
        order_request : dict
            Dictionary containing order information.
        
        Returns:
        --------
        tuple
            (is_allowed, reason, modified_order)
        """
        # Calculate current drawdown
        if risk_state['peak_balance'] <= 0:
            return (True, None, order_request)
        
        current_drawdown = 1.0 - (risk_state['account_balance'] / risk_state['peak_balance'])
        
        # Check if drawdown exceeds limit
        if current_drawdown >= risk_state['max_drawdown']:
            return (False, f"Max drawdown limit exceeded: {current_drawdown:.2%}", None)
        
        return (True, None, order_request)


class DailyLossControl(RiskControl):
    """
    Control for daily loss limits.
    """
    
    def check(self, risk_state, order_request):
        """
        Check if the daily loss exceeds limits.
        
        Parameters:
        -----------
        risk_state : dict
            Current risk state.
        order_request : dict
            Dictionary containing order information.
        
        Returns:
        --------
        tuple
            (is_allowed, reason, modified_order)
        """
        # Check if daily loss exceeds limit
        if risk_state['daily_pnl'] <= -risk_state['max_daily_loss']:
            return (False, f"Max daily loss limit exceeded: ${abs(risk_state['daily_pnl']):.2f}", None)
        
        return (True, None, order_request)


class TradeFrequencyControl(RiskControl):
    """
    Control for trade frequency limits.
    """
    
    def check(self, risk_state, order_request):
        """
        Check if trade frequency exceeds limits.
        
        Parameters:
        -----------
        risk_state : dict
            Current risk state.
        order_request : dict
            Dictionary containing order information.
        
        Returns:
        --------
        tuple
            (is_allowed, reason, modified_order)
        """
        if not order_request:
            return (True, None, None)
        
        # Check if max trades per day is exceeded
        if risk_state['trades_today'] >= risk_state['max_trades_per_day']:
            return (False, f"Max trades per day limit exceeded: {risk_state['trades_today']}", None)
        
        # Check minimum time between trades
        if risk_state['last_trade_time']:
            current_time = datetime.datetime.now()
            time_since_last_trade = (current_time - risk_state['last_trade_time']).total_seconds()
            
            if time_since_last_trade < risk_state['min_time_between_trades']:
                return (False, f"Minimum time between trades not met: {time_since_last_trade:.1f}s", None)
        
        return (True, None, order_request)


class VolatilityControl(RiskControl):
    """
    Control for volatility-based position sizing.
    """
    
    def check(self, risk_state, order_request):
        """
        Adjust position size based on market volatility.
        
        Parameters:
        -----------
        risk_state : dict
            Current risk state.
        order_request : dict
            Dictionary containing order information.
        
        Returns:
        --------
        tuple
            (is_allowed, reason, modified_order)
        """
        if not order_request or not self.config.get('volatility_scaling', True):
            return (True, None, order_request)
        
        # Get volatility multiplier
        volatility_multiplier = risk_state['volatility_multiplier']
        
        # Adjust position size
        modified_order = order_request.copy()
        original_size = modified_order.get('size', 0)
        adjusted_size = int(original_size * volatility_multiplier)
        
        # Ensure minimum size of 1 if original size was positive
        if original_size > 0:
            adjusted_size = max(1, adjusted_size)
        
        modified_order['size'] = adjusted_size
        
        return (True, None, modified_order)


class OvernightRiskControl(RiskControl):
    """
    Control for overnight position limits.
    """
    
    def check(self, risk_state, order_request):
        """
        Check if overnight position limits are respected.
        
        Parameters:
        -----------
        risk_state : dict
            Current risk state.
        order_request : dict
            Dictionary containing order information.
        
        Returns:
        --------
        tuple
            (is_allowed, reason, modified_order)
        """
        if not order_request:
            return (True, None, None)
        
        # Check if it's near market close
        current_time = datetime.datetime.now()
        market_close_time = self._get_market_close_time(current_time)
        
        # If within 15 minutes of market close
        if (market_close_time - current_time).total_seconds() <= 15 * 60:
            # Get current position and order details
            current_position = risk_state['current_position']
            order_size = order_request.get('size', 0)
            order_side = order_request.get('side', 'buy')
            
            # Calculate new position
            new_position = current_position
            if order_side == 'buy':
                new_position += order_size
            elif order_side == 'sell':
                new_position -= order_size
            
            # Check if new position exceeds overnight limit
            if abs(new_position) > risk_state['overnight_position_limit']:
                return (False, f"Overnight position limit exceeded: {abs(new_position)}", None)
        
        return (True, None, order_request)
    
    def _get_market_close_time(self, current_time):
        """
        Get the market close time for the current day.
        
        Parameters:
        -----------
        current_time : datetime.datetime
            Current time.
        
        Returns:
        --------
        datetime.datetime
            Market close time.
        """
        # NQ E-mini futures market closes at 16:00 CT (17:00 ET)
        close_time = datetime.datetime(
            current_time.year,
            current_time.month,
            current_time.day,
            17, 0, 0  # 17:00 ET
        )
        
        return close_time


class StrategyAllocationControl(RiskControl):
    """
    Control for strategy capital allocation.
    """
    
    def check(self, risk_state, order_request):
        """
        Check if strategy allocation limits are respected.
        
        Parameters:
        -----------
        risk_state : dict
            Current risk state.
        order_request : dict
            Dictionary containing order information.
        
        Returns:
        --------
        tuple
            (is_allowed, reason, modified_order)
        """
        if not order_request:
            return (True, None, None)
        
        # Get strategy name and allocation
        strategy_name = order_request.get('strategy', None)
        
        if not strategy_name:
            return (True, None, order_request)
        
        strategy_allocation = risk_state['strategy_allocations'].get(strategy_name, 0.0)
        
        # Calculate maximum position size for this strategy
        max_strategy_position = int(risk_state['max_position_size'] * strategy_allocation)
        
        # Get current strategy position (simplified, in real system would track per-strategy positions)
        current_strategy_position = 0  # Placeholder
        
        # Get order details
        order_size = order_request.get('size', 0)
        order_side = order_request.get('side', 'buy')
        
        # Calculate new strategy position
        new_strategy_position = current_strategy_position
        if order_side == 'buy':
            new_strategy_position += order_size
        elif order_side == 'sell':
            new_strategy_position -= order_size
        
        # Check if new position exceeds strategy allocation
        if abs(new_strategy_position) > max_strategy_position:
            # Modify order to respect allocation
            modified_order = order_request.copy()
            
            if order_side == 'buy':
                max_allowed = max_strategy_position - current_strategy_position
                modified_order['size'] = max(0, max_allowed)
            elif order_side == 'sell':
                max_allowed = max_strategy_position + current_strategy_position
                modified_order['size'] = max(0, max_allowed)
            
            if modified_order['size'] == 0:
                return (False, f"Strategy allocation limit exceeded for {strategy_name}", None)
            else:
                return (True, None, modified_order)
        
        return (True, None, order_request)


class CorrelationRiskControl(RiskControl):
    """
    Control for correlated asset risk.
    """
    
    def __init__(self, config):
        """
        Initialize the correlation risk control.
        
        Parameters:
        -----------
        config : dict
            Risk configuration.
        """
        super().__init__(config)
        
        # Initialize correlation matrix (placeholder)
        self.correlation_matrix = {
            'NQ': {
                'ES': 0.9,  # S&P 500 E-mini
                'YM': 0.8,  # Dow Jones E-mini
                'RTY': 0.7,  # Russell 2000 E-mini
                'QQQ': 0.95  # Nasdaq-100 ETF
            }
        }
    
    def check(self, risk_state, order_request):
        """
        Check if correlated asset risk is acceptable.
        
        Parameters:
        -----------
        risk_state : dict
            Current risk state.
        order_request : dict
            Dictionary containing order information.
        
        Returns:
        --------
        tuple
            (is_allowed, reason, modified_order)
        """
        if not order_request or not self.config.get('correlation_risk_control', True):
            return (True, None, order_request)
        
        # In a real system, this would check positions in correlated assets
        # and adjust risk accordingly. This is a simplified placeholder.
        
        return (True, None, order_request)


class BlackSwanProtection(RiskControl):
    """
    Protection against extreme market events.
    """
    
    def check(self, risk_state, order_request):
        """
        Check if black swan protection is triggered.
        
        Parameters:
        -----------
        risk_state : dict
            Current risk state.
        order_request : dict
            Dictionary containing order information.
        
        Returns:
        --------
        tuple
            (is_allowed, reason, modified_order)
        """
        if not self.config.get('black_swan_protection', True):
            return (True, None, order_request)
        
        # Check for extreme market conditions
        if risk_state['market_regime'] == 'extreme_volatility':
            return (False, "Trading suspended due to extreme market volatility", None)
        
        # Check for extreme drawdown
        if risk_state['peak_balance'] > 0:
            current_drawdown = 1.0 - (risk_state['account_balance'] / risk_state['peak_balance'])
            
            # If drawdown exceeds 15%, trigger black swan protection
            if current_drawdown >= 0.15:
                return (False, f"Black swan protection triggered: {current_drawdown:.2%} drawdown", None)
        
        return (True, None, order_request)


class RiskMonitor:
    """
    Class for monitoring and reporting risk metrics.
    """
    
    def __init__(self, risk_manager, log_dir=None):
        """
        Initialize the risk monitor.
        
        Parameters:
        -----------
        risk_manager : RiskManager
            Risk manager instance.
        log_dir : str, optional
            Directory to save risk logs.
        """
        self.risk_manager = risk_manager
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Initialize risk history
        self.risk_history = []
        
        logger.info("Risk Monitor initialized")
    
    def update(self):
        """Update risk metrics and log them."""
        # Get current risk report
        risk_report = self.risk_manager.get_risk_report()
        
        # Add to history
        self.risk_history.append(risk_report)
        
        # Log to file if log_dir is specified
        if self.log_dir:
            self._log_to_file(risk_report)
        
        # Check for risk alerts
        self._check_alerts(risk_report)
    
    def _log_to_file(self, risk_report):
        """
        Log risk report to file.
        
        Parameters:
        -----------
        risk_report : dict
            Risk report.
        """
        timestamp = datetime.datetime.fromisoformat(risk_report['timestamp'])
        date_str = timestamp.strftime('%Y-%m-%d')
        log_file = os.path.join(self.log_dir, f"risk_log_{date_str}.json")
        
        # Append to log file
        with open(log_file, 'a') as f:
            f.write(json.dumps(risk_report) + '\n')
    
    def _check_alerts(self, risk_report):
        """
        Check for risk alerts.
        
        Parameters:
        -----------
        risk_report : dict
            Risk report.
        """
        # Check drawdown
        if risk_report['current_drawdown'] >= risk_report['max_drawdown_limit'] * 0.8:
            logger.warning(f"Drawdown alert: {risk_report['current_drawdown']:.2%}")
        
        # Check daily loss
        if risk_report['daily_pnl'] <= -risk_report['max_daily_loss_limit'] * 0.8:
            logger.warning(f"Daily loss alert: ${abs(risk_report['daily_pnl']):.2f}")
        
        # Check if trading is suspended
        if not risk_report['is_trading_allowed']:
            logger.warning(f"Trading suspended: {risk_report['trading_suspended_reason']}")
    
    def generate_daily_report(self, date=None):
        """
        Generate a daily risk report.
        
        Parameters:
        -----------
        date : datetime.date, optional
            Date for the report. If None, uses today's date.
        
        Returns:
        --------
        dict
            Daily risk report.
        """
        if date is None:
            date = datetime.date.today()
        
        # Filter risk history for the specified date
        date_str = date.isoformat()
        daily_history = [
            report for report in self.risk_history
            if datetime.datetime.fromisoformat(report['timestamp']).date().isoformat() == date_str
        ]
        
        if not daily_history:
            logger.warning(f"No risk data for {date_str}")
            return None
        
        # Calculate daily metrics
        start_balance = daily_history[0]['account_balance']
        end_balance = daily_history[-1]['account_balance']
        daily_pnl = end_balance - start_balance
        
        max_drawdown = max([report['current_drawdown'] for report in daily_history], default=0)
        
        # Count trades
        trades_count = daily_history[-1]['trades_today']
        
        # Calculate average position size
        position_sizes = [abs(report['current_position']) for report in daily_history]
        avg_position_size = sum(position_sizes) / len(position_sizes) if position_sizes else 0
        
        report = {
            'date': date_str,
            'start_balance': start_balance,
            'end_balance': end_balance,
            'daily_pnl': daily_pnl,
            'daily_return': daily_pnl / start_balance if start_balance > 0 else 0,
            'max_drawdown': max_drawdown,
            'trades_count': trades_count,
            'avg_position_size': avg_position_size,
            'risk_level': daily_history[-1]['risk_level'],
            'market_regime': daily_history[-1]['market_regime']
        }
        
        return report
    
    def plot_daily_metrics(self, date=None, save_path=None):
        """
        Plot daily risk metrics.
        
        Parameters:
        -----------
        date : datetime.date, optional
            Date for the plot. If None, uses today's date.
        save_path : str, optional
            Path to save the plot. If None, the plot is displayed.
        
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure.
        """
        if date is None:
            date = datetime.date.today()
        
        # Filter risk history for the specified date
        date_str = date.isoformat()
        daily_history = [
            report for report in self.risk_history
            if datetime.datetime.fromisoformat(report['timestamp']).date().isoformat() == date_str
        ]
        
        if not daily_history:
            logger.warning(f"No risk data for {date_str}")
            return None
        
        # Create figure
        fig = self.risk_manager.plot_risk_metrics(daily_history, save_path)
        
        return fig


if __name__ == "__main__":
    # Example usage
    print("Risk Management System module ready for use")

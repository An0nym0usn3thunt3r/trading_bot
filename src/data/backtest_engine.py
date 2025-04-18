"""
Backtesting framework for Nasdaq-100 E-mini futures trading bot.
This module provides functionality for testing trading strategies on historical data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
import os
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Engine for backtesting trading strategies on historical data.
    """
    
    def __init__(self, initial_capital=100000.0, commission=2.0, slippage=1.0):
        """
        Initialize the backtesting engine.
        
        Parameters:
        -----------
        initial_capital : float, optional
            Initial capital for backtesting.
        commission : float, optional
            Commission per trade in dollars.
        slippage : float, optional
            Slippage per trade in ticks.
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.reset()
        logger.info(f"BacktestEngine initialized with capital=${initial_capital}, "
                   f"commission=${commission}, slippage={slippage} ticks")
    
    def reset(self):
        """Reset the backtesting engine to initial state."""
        self.capital = self.initial_capital
        self.position = 0
        self.trades = []
        self.equity_curve = []
        self.current_trade = None
        self.trade_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'breakeven_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0
        }
        logger.info("BacktestEngine reset to initial state")
    
    def apply_slippage(self, price, direction):
        """
        Apply slippage to the price based on trade direction.
        
        Parameters:
        -----------
        price : float
            Original price.
        direction : int
            Trade direction (1 for buy, -1 for sell).
            
        Returns:
        --------
        float
            Price with slippage applied.
        """
        # For buys, increase price; for sells, decrease price
        return price + (direction * self.slippage)
    
    def enter_position(self, timestamp, price, quantity, direction, strategy_name):
        """
        Enter a new position.
        
        Parameters:
        -----------
        timestamp : datetime
            Time of the trade.
        price : float
            Entry price.
        quantity : int
            Number of contracts.
        direction : int
            Trade direction (1 for long, -1 for short).
        strategy_name : str
            Name of the strategy generating the trade.
            
        Returns:
        --------
        bool
            True if position was entered successfully, False otherwise.
        """
        if self.position != 0:
            logger.warning(f"Cannot enter position, already in position: {self.position}")
            return False
        
        # Apply slippage
        adjusted_price = self.apply_slippage(price, direction)
        
        # Calculate trade cost
        trade_cost = adjusted_price * quantity
        commission_cost = self.commission * quantity
        
        # Check if we have enough capital
        if trade_cost + commission_cost > self.capital:
            logger.warning(f"Insufficient capital (${self.capital}) for trade: ${trade_cost + commission_cost}")
            return False
        
        # Update position and capital
        self.position = direction * quantity
        self.capital -= commission_cost
        
        # Record the trade
        self.current_trade = {
            'entry_time': timestamp,
            'entry_price': adjusted_price,
            'quantity': quantity,
            'direction': direction,
            'strategy': strategy_name,
            'commission': commission_cost
        }
        
        logger.info(f"Entered {direction * quantity} position at ${adjusted_price} "
                   f"(strategy: {strategy_name})")
        return True
    
    def exit_position(self, timestamp, price, reason=""):
        """
        Exit the current position.
        
        Parameters:
        -----------
        timestamp : datetime
            Time of the exit.
        price : float
            Exit price.
        reason : str, optional
            Reason for exiting the position.
            
        Returns:
        --------
        bool
            True if position was exited successfully, False otherwise.
        """
        if self.position == 0 or self.current_trade is None:
            logger.warning("Cannot exit position, no position to exit")
            return False
        
        # Apply slippage (opposite direction to entry)
        adjusted_price = self.apply_slippage(price, -self.current_trade['direction'])
        
        # Calculate profit/loss
        quantity = abs(self.position)
        direction = 1 if self.position > 0 else -1
        entry_price = self.current_trade['entry_price']
        
        # For long positions: profit = (exit_price - entry_price) * quantity
        # For short positions: profit = (entry_price - exit_price) * quantity
        profit = direction * (adjusted_price - entry_price) * quantity
        
        # Calculate commission
        commission_cost = self.commission * quantity
        
        # Update capital and position
        self.capital += (adjusted_price * quantity) + profit - commission_cost
        self.position = 0
        
        # Complete the trade record
        self.current_trade.update({
            'exit_time': timestamp,
            'exit_price': adjusted_price,
            'profit': profit,
            'commission': self.current_trade['commission'] + commission_cost,
            'reason': reason
        })
        
        # Add to trades list
        self.trades.append(self.current_trade)
        
        # Update trade statistics
        self._update_trade_stats(profit)
        
        logger.info(f"Exited position at ${adjusted_price}, profit: ${profit}, reason: {reason}")
        
        # Reset current trade
        self.current_trade = None
        
        return True
    
    def _update_trade_stats(self, profit):
        """
        Update trade statistics after a completed trade.
        
        Parameters:
        -----------
        profit : float
            Profit/loss from the trade.
        """
        self.trade_stats['total_trades'] += 1
        
        if profit > 0:
            self.trade_stats['winning_trades'] += 1
            self.trade_stats['total_profit'] += profit
        elif profit < 0:
            self.trade_stats['losing_trades'] += 1
            self.trade_stats['total_loss'] += abs(profit)
        else:
            self.trade_stats['breakeven_trades'] += 1
    
    def update_equity(self, timestamp, current_price):
        """
        Update the equity curve with current position value.
        
        Parameters:
        -----------
        timestamp : datetime
            Current timestamp.
        current_price : float
            Current price for valuation.
        """
        # Calculate current equity
        position_value = 0
        if self.position != 0 and self.current_trade is not None:
            # For long positions: value = current_price * quantity
            # For short positions: value = (2*entry_price - current_price) * quantity
            direction = self.current_trade['direction']
            quantity = self.current_trade['quantity']
            if direction > 0:  # Long position
                position_value = current_price * quantity
            else:  # Short position
                entry_price = self.current_trade['entry_price']
                position_value = (2 * entry_price - current_price) * quantity
        
        equity = self.capital + position_value
        
        # Update equity curve
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity,
            'capital': self.capital,
            'position_value': position_value
        })
        
        # Update max drawdown
        self._update_drawdown()
    
    def _update_drawdown(self):
        """Update maximum drawdown statistics."""
        if len(self.equity_curve) < 2:
            return
        
        # Get equity values
        equity_values = [point['equity'] for point in self.equity_curve]
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_values)
        
        # Calculate drawdown in dollars
        drawdown = running_max - equity_values
        
        # Calculate drawdown in percentage
        drawdown_pct = drawdown / running_max
        
        # Update max drawdown
        max_dd = np.max(drawdown)
        max_dd_pct = np.max(drawdown_pct)
        
        if max_dd > self.trade_stats['max_drawdown']:
            self.trade_stats['max_drawdown'] = max_dd
            
        if max_dd_pct > self.trade_stats['max_drawdown_pct']:
            self.trade_stats['max_drawdown_pct'] = max_dd_pct
    
    def run_backtest(self, data, strategy, start_date=None, end_date=None):
        """
        Run a backtest on the given data with the specified strategy.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical price data with OHLCV columns.
        strategy : object
            Strategy object with generate_signals method.
        start_date : datetime, optional
            Start date for backtest. If None, uses the first date in data.
        end_date : datetime, optional
            End date for backtest. If None, uses the last date in data.
            
        Returns:
        --------
        dict
            Backtest results and statistics.
        """
        logger.info(f"Starting backtest with strategy: {strategy.__class__.__name__}")
        
        # Reset the engine
        self.reset()
        
        # Filter data by date range if specified
        if start_date is not None or end_date is not None:
            mask = True
            if start_date is not None:
                mask = mask & (data.index >= start_date)
            if end_date is not None:
                mask = mask & (data.index <= end_date)
            data = data[mask]
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Combine data and signals
        backtest_data = pd.concat([data, signals], axis=1)
        
        # Iterate through each bar
        for timestamp, bar in backtest_data.iterrows():
            # Check for exit signals if in a position
            if self.position != 0:
                if (self.position > 0 and bar.get('exit_long', False)) or \
                   (self.position < 0 and bar.get('exit_short', False)):
                    self.exit_position(timestamp, bar['close'], "Signal")
                
                # Check for stop loss and take profit
                if self.current_trade is not None:
                    entry_price = self.current_trade['entry_price']
                    direction = self.current_trade['direction']
                    
                    # Stop loss (assuming 2% for this example)
                    stop_pct = 0.02
                    stop_price = entry_price * (1 - stop_pct) if direction > 0 else entry_price * (1 + stop_pct)
                    
                    # Take profit (assuming 3% for this example)
                    tp_pct = 0.03
                    tp_price = entry_price * (1 + tp_pct) if direction > 0 else entry_price * (1 - tp_pct)
                    
                    # Check if stop loss hit
                    if (direction > 0 and bar['low'] <= stop_price) or \
                       (direction < 0 and bar['high'] >= stop_price):
                        self.exit_position(timestamp, stop_price, "Stop Loss")
                    
                    # Check if take profit hit
                    elif (direction > 0 and bar['high'] >= tp_price) or \
                         (direction < 0 and bar['low'] <= tp_price):
                        self.exit_position(timestamp, tp_price, "Take Profit")
            
            # Check for entry signals if not in a position
            elif self.position == 0:
                if bar.get('enter_long', False):
                    # Calculate position size (1% risk per trade)
                    risk_pct = 0.01
                    risk_amount = self.capital * risk_pct
                    stop_pct = 0.02  # 2% stop loss
                    price_risk = bar['close'] * stop_pct
                    quantity = max(1, int(risk_amount / price_risk))
                    
                    self.enter_position(timestamp, bar['close'], quantity, 1, strategy.__class__.__name__)
                
                elif bar.get('enter_short', False):
                    # Calculate position size (1% risk per trade)
                    risk_pct = 0.01
                    risk_amount = self.capital * risk_pct
                    stop_pct = 0.02  # 2% stop loss
                    price_risk = bar['close'] * stop_pct
                    quantity = max(1, int(risk_amount / price_risk))
                    
                    self.enter_position(timestamp, bar['close'], quantity, -1, strategy.__class__.__name__)
            
            # Update equity curve
            self.update_equity(timestamp, bar['close'])
        
        # Close any open positions at the end of the backtest
        if self.position != 0:
            last_timestamp = backtest_data.index[-1]
            last_price = backtest_data['close'].iloc[-1]
            self.exit_position(last_timestamp, last_price, "End of Backtest")
        
        # Calculate additional statistics
        self._calculate_performance_metrics()
        
        logger.info(f"Backtest completed: {self.trade_stats['total_trades']} trades, "
                   f"final equity: ${self.equity_curve[-1]['equity']:.2f}")
        
        return self.get_results()
    
    def _calculate_performance_metrics(self):
        """Calculate additional performance metrics after backtest completion."""
        if not self.trades:
            logger.warning("No trades to calculate performance metrics")
            return
        
        # Calculate win rate
        total_trades = self.trade_stats['total_trades']
        winning_trades = self.trade_stats['winning_trades']
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate profit factor
        total_profit = self.trade_stats['total_profit']
        total_loss = self.trade_stats['total_loss']
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate average trade
        net_profit = total_profit - total_loss
        avg_trade = net_profit / total_trades if total_trades > 0 else 0
        
        # Calculate average winner and loser
        avg_winner = total_profit / winning_trades if winning_trades > 0 else 0
        losing_trades = self.trade_stats['losing_trades']
        avg_loser = -total_loss / losing_trades if losing_trades > 0 else 0
        
        # Calculate risk-reward ratio
        risk_reward = abs(avg_winner / avg_loser) if avg_loser != 0 else float('inf')
        
        # Calculate consecutive wins and losses
        consecutive_wins = 0
        consecutive_losses = 0
        current_streak = 0
        
        for trade in self.trades:
            if trade['profit'] > 0:
                if current_streak > 0:
                    current_streak += 1
                else:
                    current_streak = 1
                consecutive_wins = max(consecutive_wins, current_streak)
            elif trade['profit'] < 0:
                if current_streak < 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                consecutive_losses = max(consecutive_losses, abs(current_streak))
            else:
                current_streak = 0
        
        self.trade_stats['win_rate'] = win_rate
        self.trade_stats['profit_factor'] = profit_factor
        self.trade_stats['avg_trade'] = avg_trade
        self.trade_stats['avg_winner'] = avg_winner
        self.trade_stats['avg_loser'] = avg_loser
        self.trade_stats['risk_reward'] = risk_reward
        self.trade_stats['max_consecutive_wins'] = consecutive_wins
        self.trade_stats['max_consecutive_losses'] = consecutive_losses
        
        # Calculate annualized return and Sharpe ratio
        if len(self.equity_curve) > 1:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.set_index('timestamp', inplace=True)
            
            # Calculate returns
            equity_df['returns'] = equity_df['equity'].pct_change()
            
            # Calculate annualized return
            start_date = equity_df.index[0]
            end_date = equity_df.index[-1]
            years = (end_date - start_date).days / 365.25
            
            if years > 0:
                total_return = (equity_df['equity'].iloc[-1] / self.initial_capital) - 1
                annualized_return = (1 + total_return) ** (1 / years) - 1
                
                # Calculate Sharpe ratio (assuming risk-free rate of 0)
                daily_returns = equity_df['returns'].dropna()
                if len(daily_returns) > 0:
                    sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() \
                        if daily_returns.std() > 0 else 0
                    
                    self.trade_stats['annualized_return'] = annualized_return
                    self.trade_stats['sharpe_ratio'] = sharpe_ratio
    
    def get_results(self):
        """
        Get the backtest results.
        
        Returns:
        --------
        dict
            Dictionary containing backtest results and statistics.
        """
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        return {
            'initial_capital': self.initial_capital,
            'final_equity': equity_df['equity'].iloc[-1] if not equity_df.empty else self.initial_capital,
            'total_return': (equity_df['equity'].iloc[-1] / self.initial_capital - 1) if not equity_df.empty else 0,
            'equity_curve': equity_df,
            'trades': trades_df,
            'stats': self.trade_stats
        }
    
    def plot_results(self, save_path=None):
        """
        Plot the backtest results.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot. If None, the plot is displayed.
        """
        if not self.equity_curve:
            logger.warning("No equity curve data to plot")
            return
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        axes[0].plot(equity_df.index, equity_df['equity'], label='Equity')
        axes[0].set_title('Equity Curve')
        axes[0].set_ylabel('Equity ($)')
        axes[0].grid(True)
        axes[0].legend()
        
        # Plot drawdown
        running_max = np.maximum.accumulate(equity_df['equity'])
        drawdown = (running_max - equity_df['equity']) / running_max * 100
        
        axes[1].fill_between(equity_df.index, 0, drawdown, color='red', alpha=0.3)
        axes[1].set_title('Drawdown (%)')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].set_xlabel('Date')
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def generate_report(self, save_path=None):
        """
        Generate a detailed backtest report.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the report. If None, the report is returned as a string.
            
        Returns:
        --------
        str or None
            Report as a string if save_path is None, otherwise None.
        """
        results = self.get_results()
        stats = results['stats']
        
        report = []
        report.append("=" * 50)
        report.append("BACKTEST REPORT")
        report.append("=" * 50)
        report.append("")
        
        report.append("SUMMARY:")
        report.append(f"Initial Capital: ${self.initial_capital:.2f}")
        report.append(f"Final Equity: ${results['final_equity']:.2f}")
        report.append(f"Total Return: {results['total_return']*100:.2f}%")
        if 'annualized_return' in stats:
            report.append(f"Annualized Return: {stats['annualized_return']*100:.2f}%")
        if 'sharpe_ratio' in stats:
            report.append(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        report.append("")
        
        report.append("TRADE STATISTICS:")
        report.append(f"Total Trades: {stats['total_trades']}")
        report.append(f"Winning Trades: {stats['winning_trades']} ({stats.get('win_rate', 0)*100:.2f}%)")
        report.append(f"Losing Trades: {stats['losing_trades']}")
        report.append(f"Breakeven Trades: {stats['breakeven_trades']}")
        report.append("")
        
        report.append("PROFIT/LOSS:")
        report.append(f"Gross Profit: ${stats['total_profit']:.2f}")
        report.append(f"Gross Loss: ${stats['total_loss']:.2f}")
        report.append(f"Net Profit/Loss: ${stats['total_profit']-stats['total_loss']:.2f}")
        report.append(f"Profit Factor: {stats.get('profit_factor', 0):.2f}")
        report.append("")
        
        report.append("AVERAGE TRADE:")
        report.append(f"Average Trade: ${stats.get('avg_trade', 0):.2f}")
        report.append(f"Average Winner: ${stats.get('avg_winner', 0):.2f}")
        report.append(f"Average Loser: ${stats.get('avg_loser', 0):.2f}")
        report.append(f"Risk-Reward Ratio: {stats.get('risk_reward', 0):.2f}")
        report.append("")
        
        report.append("DRAWDOWN:")
        report.append(f"Maximum Drawdown: ${stats['max_drawdown']:.2f} ({stats['max_drawdown_pct']*100:.2f}%)")
        report.append("")
        
        report.append("STREAKS:")
        report.append(f"Maximum Consecutive Winners: {stats.get('max_consecutive_wins', 0)}")
        report.append(f"Maximum Consecutive Losers: {stats.get('max_consecutive_losses', 0)}")
        
        report_str = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_str)
            logger.info(f"Report saved to {save_path}")
            return None
        else:
            return report_str


class Strategy:
    """Base class for trading strategies."""
    
    def __init__(self, name="BaseStrategy"):
        """
        Initialize the strategy.
        
        Parameters:
        -----------
        name : str, optional
            Name of the strategy.
        """
        self.name = name
        logger.info(f"Strategy '{name}' initialized")
    
    def generate_signals(self, data):
        """
        Generate trading signals from data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical price data with OHLCV columns.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with signal columns (enter_long, enter_short, exit_long, exit_short).
        """
        # This is a placeholder to be implemented by subclasses
        signals = pd.DataFrame(index=data.index)
        signals['enter_long'] = False
        signals['enter_short'] = False
        signals['exit_long'] = False
        signals['exit_short'] = False
        return signals


if __name__ == "__main__":
    # Example usage
    engine = BacktestEngine()
    # This would be replaced with actual implementation
    # strategy = SomeStrategy()
    # results = engine.run_backtest(data, strategy)
    print("BacktestEngine module ready for use")

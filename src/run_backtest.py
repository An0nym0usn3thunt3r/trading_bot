"""
Main module for running the data pipeline and backtesting framework.
This serves as an entry point for testing the data processing and backtesting components.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import argparse
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from data.backtest_engine import BacktestEngine, Strategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), 'logs', 'backtest.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MovingAverageCrossoverStrategy(Strategy):
    """
    Simple moving average crossover strategy for testing the backtesting framework.
    """
    
    def __init__(self, short_window=10, long_window=50):
        """
        Initialize the strategy.
        
        Parameters:
        -----------
        short_window : int, optional
            Short moving average window.
        long_window : int, optional
            Long moving average window.
        """
        super().__init__(name=f"MA_Crossover_{short_window}_{long_window}")
        self.short_window = short_window
        self.long_window = long_window
        logger.info(f"MovingAverageCrossoverStrategy initialized with short_window={short_window}, "
                   f"long_window={long_window}")
    
    def generate_signals(self, data):
        """
        Generate trading signals based on moving average crossover.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical price data with OHLCV columns.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with signal columns.
        """
        signals = pd.DataFrame(index=data.index)
        signals['enter_long'] = False
        signals['enter_short'] = False
        signals['exit_long'] = False
        signals['exit_short'] = False
        
        # Calculate moving averages
        data['short_ma'] = data['close'].rolling(window=self.short_window).mean()
        data['long_ma'] = data['close'].rolling(window=self.long_window).mean()
        
        # Generate signals
        # Buy signal: short MA crosses above long MA
        signals['enter_long'] = (data['short_ma'] > data['long_ma']) & \
                               (data['short_ma'].shift(1) <= data['long_ma'].shift(1))
        
        # Sell signal: short MA crosses below long MA
        signals['exit_long'] = (data['short_ma'] < data['long_ma']) & \
                              (data['short_ma'].shift(1) >= data['long_ma'].shift(1))
        
        # Short signal: short MA crosses below long MA
        signals['enter_short'] = (data['short_ma'] < data['long_ma']) & \
                                (data['short_ma'].shift(1) >= data['long_ma'].shift(1))
        
        # Cover signal: short MA crosses above long MA
        signals['exit_short'] = (data['short_ma'] > data['long_ma']) & \
                               (data['short_ma'].shift(1) <= data['long_ma'].shift(1))
        
        return signals


def create_sample_data(output_file, num_rows=5000):
    """
    Create a sample OHLCV dataset for testing.
    
    Parameters:
    -----------
    output_file : str
        Path to save the sample data.
    num_rows : int, optional
        Number of rows to generate.
    """
    logger.info(f"Creating sample data with {num_rows} rows")
    
    # Create a date range
    start_date = datetime(2022, 1, 1)
    dates = pd.date_range(start=start_date, periods=num_rows, freq='1min')
    
    # Initialize with a starting price
    base_price = 20000.0
    
    # Generate random price movements
    np.random.seed(42)  # For reproducibility
    
    # Create price series with some trend and volatility
    price_changes = np.random.normal(0.0001, 0.001, num_rows)  # Small mean return, some volatility
    price_multipliers = np.cumprod(1 + price_changes)
    close_prices = base_price * price_multipliers
    
    # Add some intraday patterns
    time_of_day = np.array([d.hour * 60 + d.minute for d in dates])
    market_open_effect = 0.0005 * np.sin(2 * np.pi * time_of_day / (24 * 60))
    close_prices = close_prices * (1 + market_open_effect)
    
    # Generate OHLC based on close prices
    high_prices = close_prices * (1 + np.abs(np.random.normal(0.001, 0.002, num_rows)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0.001, 0.002, num_rows)))
    open_prices = close_prices.copy()
    np.random.shuffle(open_prices)  # Randomize open prices
    
    # Ensure high >= open, close and low <= open, close
    for i in range(num_rows):
        high_prices[i] = max(high_prices[i], open_prices[i], close_prices[i])
        low_prices[i] = min(low_prices[i], open_prices[i], close_prices[i])
    
    # Generate volume
    volume = np.random.lognormal(6, 1, num_rows).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    logger.info(f"Sample data saved to {output_file}")


def run_backtest_demo(data_file, output_dir):
    """
    Run a demonstration of the backtesting framework.
    
    Parameters:
    -----------
    data_file : str
        Path to the data file.
    output_dir : str
        Directory to save output files.
    """
    logger.info("Starting backtest demonstration")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process data
    loader = DataLoader()
    df = loader.load_csv_data(data_file)
    df = loader.preprocess_data(df)
    
    # Create 5-minute resampled data
    df_5min = loader.resample_timeframe(df, '5min')
    
    # Engineer features
    df = loader.engineer_features(df)
    df_5min = loader.engineer_features(df_5min)
    
    # Split data
    train_df, val_df, test_df = loader.split_data(df)
    
    # Initialize backtest engine
    engine = BacktestEngine(initial_capital=100000.0, commission=2.0, slippage=1.0)
    
    # Create and run strategies
    strategies = [
        MovingAverageCrossoverStrategy(short_window=10, long_window=30),
        MovingAverageCrossoverStrategy(short_window=5, long_window=20),
        MovingAverageCrossoverStrategy(short_window=20, long_window=50)
    ]
    
    results = {}
    for strategy in strategies:
        logger.info(f"Running backtest with strategy: {strategy.name}")
        result = engine.run_backtest(test_df, strategy)
        results[strategy.name] = result
        
        # Save equity curve plot
        plot_path = os.path.join(output_dir, f"{strategy.name}_equity_curve.png")
        engine.plot_results(save_path=plot_path)
        
        # Save report
        report_path = os.path.join(output_dir, f"{strategy.name}_report.txt")
        engine.generate_report(save_path=report_path)
    
    # Compare strategies
    compare_strategies(results, output_dir)
    
    logger.info("Backtest demonstration completed")


def compare_strategies(results, output_dir):
    """
    Compare the performance of multiple strategies.
    
    Parameters:
    -----------
    results : dict
        Dictionary of backtest results by strategy name.
    output_dir : str
        Directory to save output files.
    """
    logger.info("Comparing strategy performance")
    
    # Create comparison DataFrame
    comparison = []
    for strategy_name, result in results.items():
        stats = result['stats']
        comparison.append({
            'Strategy': strategy_name,
            'Total Return (%)': result['total_return'] * 100,
            'Sharpe Ratio': stats.get('sharpe_ratio', 0),
            'Win Rate (%)': stats.get('win_rate', 0) * 100,
            'Profit Factor': stats.get('profit_factor', 0),
            'Max Drawdown (%)': stats['max_drawdown_pct'] * 100,
            'Total Trades': stats['total_trades']
        })
    
    comparison_df = pd.DataFrame(comparison)
    
    # Save comparison to CSV
    comparison_path = os.path.join(output_dir, "strategy_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    
    # Plot equity curves for all strategies
    plt.figure(figsize=(12, 8))
    
    for strategy_name, result in results.items():
        equity_df = result['equity_curve']
        plt.plot(equity_df['timestamp'], equity_df['equity'], label=strategy_name)
    
    plt.title('Strategy Comparison - Equity Curves')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.legend()
    
    # Save plot
    plot_path = os.path.join(output_dir, "strategy_comparison.png")
    plt.savefig(plot_path)
    
    logger.info(f"Strategy comparison saved to {output_dir}")


def main():
    """Main function to run the data pipeline and backtesting framework."""
    parser = argparse.ArgumentParser(description='Run data pipeline and backtesting framework')
    parser.add_argument('--data-file', type=str, help='Path to data file')
    parser.add_argument('--create-sample', action='store_true', help='Create sample data')
    parser.add_argument('--output-dir', type=str, default='../output', help='Output directory')
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample data if requested
    if args.create_sample:
        sample_data_file = os.path.join(output_dir, 'sample_data.csv')
        create_sample_data(sample_data_file)
        data_file = sample_data_file
    else:
        data_file = args.data_file
    
    # Check if data file exists
    if data_file is None or not os.path.exists(data_file):
        logger.error("Data file not found. Please provide a valid data file or use --create-sample")
        return
    
    # Run backtest demo
    run_backtest_demo(data_file, output_dir)


if __name__ == "__main__":
    main()

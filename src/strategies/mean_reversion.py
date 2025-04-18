"""
Mean Reversion Strategies for Nasdaq-100 E-mini futures trading bot.
This module implements various mean reversion trading strategies.
"""

import pandas as pd
import numpy as np
import logging
from strategies.base_strategy import Strategy, add_technical_indicators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BollingerBandStrategy(Strategy):
    """
    Bollinger Band mean reversion strategy.
    
    This strategy generates buy signals when price touches the lower Bollinger Band
    and sell signals when price touches the upper Bollinger Band.
    """
    
    def __init__(self, window=20, num_std=2.0, exit_window=10, rsi_filter=True, rsi_threshold=30):
        """
        Initialize the Bollinger Band strategy.
        
        Parameters:
        -----------
        window : int, optional
            Window period for calculating Bollinger Bands.
        num_std : float, optional
            Number of standard deviations for the bands.
        exit_window : int, optional
            Window for exit signals (mean reversion).
        rsi_filter : bool, optional
            Whether to use RSI as a filter.
        rsi_threshold : int, optional
            RSI threshold for entry signals.
        """
        parameters = {
            'window': window,
            'num_std': num_std,
            'exit_window': exit_window,
            'rsi_filter': rsi_filter,
            'rsi_threshold': rsi_threshold
        }
        super().__init__(name="BollingerBand", parameters=parameters)
    
    def generate_signals(self, data):
        """
        Generate trading signals based on Bollinger Bands.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical price data with OHLCV columns.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with signal columns.
        """
        # Get parameters
        window = self.parameters['window']
        num_std = self.parameters['num_std']
        exit_window = self.parameters['exit_window']
        rsi_filter = self.parameters['rsi_filter']
        rsi_threshold = self.parameters['rsi_threshold']
        
        # Make a copy of the data
        df = data.copy()
        
        # Calculate Bollinger Bands if not already present
        if f'bb_upper_{window}' not in df.columns:
            # Calculate middle band (SMA)
            df[f'bb_mid_{window}'] = df['close'].rolling(window=window).mean()
            
            # Calculate standard deviation
            df[f'bb_std_{window}'] = df['close'].rolling(window=window).std()
            
            # Calculate upper and lower bands
            df[f'bb_upper_{window}'] = df[f'bb_mid_{window}'] + (num_std * df[f'bb_std_{window}'])
            df[f'bb_lower_{window}'] = df[f'bb_mid_{window}'] - (num_std * df[f'bb_std_{window}'])
            
            # Calculate bandwidth
            df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / df[f'bb_mid_{window}']
        
        # Calculate RSI if needed and not already present
        if rsi_filter and 'rsi_14' not in df.columns:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Initialize signal columns
        signals = pd.DataFrame(index=df.index)
        signals['enter_long'] = False
        signals['enter_short'] = False
        signals['exit_long'] = False
        signals['exit_short'] = False
        
        # Generate entry signals
        # Long entry: price below lower band and RSI oversold (if filter enabled)
        if rsi_filter:
            signals['enter_long'] = (df['close'] < df[f'bb_lower_{window}']) & (df['rsi_14'] < rsi_threshold)
        else:
            signals['enter_long'] = df['close'] < df[f'bb_lower_{window}']
        
        # Short entry: price above upper band and RSI overbought (if filter enabled)
        if rsi_filter:
            signals['enter_short'] = (df['close'] > df[f'bb_upper_{window}']) & (df['rsi_14'] > (100 - rsi_threshold))
        else:
            signals['enter_short'] = df['close'] > df[f'bb_upper_{window}']
        
        # Generate exit signals
        # Exit long: price crosses above middle band or has been in position for exit_window bars
        signals['exit_long'] = df['close'] > df[f'bb_mid_{window}']
        
        # Exit short: price crosses below middle band or has been in position for exit_window bars
        signals['exit_short'] = df['close'] < df[f'bb_mid_{window}']
        
        # Add time-based exit
        if exit_window > 0:
            # Create rolling window for entry signals
            long_entry_window = signals['enter_long'].rolling(window=exit_window).sum()
            short_entry_window = signals['enter_short'].rolling(window=exit_window).sum()
            
            # Exit if we've been in position for exit_window bars
            signals['exit_long'] = signals['exit_long'] | (long_entry_window.shift(exit_window) > 0)
            signals['exit_short'] = signals['exit_short'] | (short_entry_window.shift(exit_window) > 0)
        
        logger.info(f"Generated Bollinger Band signals with window={window}, num_std={num_std}")
        return signals


class RSIMeanReversionStrategy(Strategy):
    """
    RSI (Relative Strength Index) mean reversion strategy.
    
    This strategy generates buy signals when RSI is oversold and
    sell signals when RSI is overbought.
    """
    
    def __init__(self, rsi_window=14, oversold=30, overbought=70, exit_window=5):
        """
        Initialize the RSI mean reversion strategy.
        
        Parameters:
        -----------
        rsi_window : int, optional
            Window period for calculating RSI.
        oversold : int, optional
            RSI threshold for oversold condition (buy signal).
        overbought : int, optional
            RSI threshold for overbought condition (sell signal).
        exit_window : int, optional
            Window for exit signals (mean reversion).
        """
        parameters = {
            'rsi_window': rsi_window,
            'oversold': oversold,
            'overbought': overbought,
            'exit_window': exit_window
        }
        super().__init__(name="RSIMeanReversion", parameters=parameters)
    
    def generate_signals(self, data):
        """
        Generate trading signals based on RSI.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical price data with OHLCV columns.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with signal columns.
        """
        # Get parameters
        rsi_window = self.parameters['rsi_window']
        oversold = self.parameters['oversold']
        overbought = self.parameters['overbought']
        exit_window = self.parameters['exit_window']
        
        # Make a copy of the data
        df = data.copy()
        
        # Calculate RSI if not already present
        if f'rsi_{rsi_window}' not in df.columns:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=rsi_window).mean()
            avg_loss = loss.rolling(window=rsi_window).mean()
            
            rs = avg_gain / avg_loss
            df[f'rsi_{rsi_window}'] = 100 - (100 / (1 + rs))
        
        # Initialize signal columns
        signals = pd.DataFrame(index=df.index)
        signals['enter_long'] = False
        signals['enter_short'] = False
        signals['exit_long'] = False
        signals['exit_short'] = False
        
        # Generate entry signals
        # Long entry: RSI crosses below oversold threshold
        signals['enter_long'] = (df[f'rsi_{rsi_window}'] < oversold) & (df[f'rsi_{rsi_window}'].shift(1) >= oversold)
        
        # Short entry: RSI crosses above overbought threshold
        signals['enter_short'] = (df[f'rsi_{rsi_window}'] > overbought) & (df[f'rsi_{rsi_window}'].shift(1) <= overbought)
        
        # Generate exit signals
        # Exit long: RSI crosses above 50 (middle)
        signals['exit_long'] = (df[f'rsi_{rsi_window}'] > 50) & (df[f'rsi_{rsi_window}'].shift(1) <= 50)
        
        # Exit short: RSI crosses below 50 (middle)
        signals['exit_short'] = (df[f'rsi_{rsi_window}'] < 50) & (df[f'rsi_{rsi_window}'].shift(1) >= 50)
        
        # Add time-based exit
        if exit_window > 0:
            # Create rolling window for entry signals
            long_entry_window = signals['enter_long'].rolling(window=exit_window).sum()
            short_entry_window = signals['enter_short'].rolling(window=exit_window).sum()
            
            # Exit if we've been in position for exit_window bars
            signals['exit_long'] = signals['exit_long'] | (long_entry_window.shift(exit_window) > 0)
            signals['exit_short'] = signals['exit_short'] | (short_entry_window.shift(exit_window) > 0)
        
        logger.info(f"Generated RSI mean reversion signals with window={rsi_window}, oversold={oversold}, overbought={overbought}")
        return signals


class MACDMeanReversionStrategy(Strategy):
    """
    MACD (Moving Average Convergence Divergence) mean reversion strategy.
    
    This strategy generates signals based on MACD divergence from price.
    """
    
    def __init__(self, fast_period=12, slow_period=26, signal_period=9, divergence_threshold=0.01):
        """
        Initialize the MACD mean reversion strategy.
        
        Parameters:
        -----------
        fast_period : int, optional
            Fast EMA period for MACD calculation.
        slow_period : int, optional
            Slow EMA period for MACD calculation.
        signal_period : int, optional
            Signal line EMA period.
        divergence_threshold : float, optional
            Threshold for divergence detection (as percentage).
        """
        parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
            'divergence_threshold': divergence_threshold
        }
        super().__init__(name="MACDMeanReversion", parameters=parameters)
    
    def generate_signals(self, data):
        """
        Generate trading signals based on MACD divergence.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical price data with OHLCV columns.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with signal columns.
        """
        # Get parameters
        fast_period = self.parameters['fast_period']
        slow_period = self.parameters['slow_period']
        signal_period = self.parameters['signal_period']
        divergence_threshold = self.parameters['divergence_threshold']
        
        # Make a copy of the data
        df = data.copy()
        
        # Calculate MACD if not already present
        if 'macd' not in df.columns:
            # Calculate EMAs
            fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean()
            slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean()
            
            # Calculate MACD line
            df['macd'] = fast_ema - slow_ema
            
            # Calculate signal line
            df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
            
            # Calculate histogram
            df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Calculate price momentum and MACD momentum
        df['price_momentum'] = df['close'].pct_change(5)
        df['macd_momentum'] = df['macd'].diff(5)
        
        # Detect divergence
        # Bullish divergence: price making lower lows but MACD making higher lows
        df['bullish_divergence'] = (df['price_momentum'] < -divergence_threshold) & (df['macd_momentum'] > divergence_threshold)
        
        # Bearish divergence: price making higher highs but MACD making lower highs
        df['bearish_divergence'] = (df['price_momentum'] > divergence_threshold) & (df['macd_momentum'] < -divergence_threshold)
        
        # Initialize signal columns
        signals = pd.DataFrame(index=df.index)
        signals['enter_long'] = False
        signals['enter_short'] = False
        signals['exit_long'] = False
        signals['exit_short'] = False
        
        # Generate entry signals
        # Long entry: bullish divergence and MACD histogram turns positive
        signals['enter_long'] = df['bullish_divergence'] & (df['macd_hist'] > 0) & (df['macd_hist'].shift(1) <= 0)
        
        # Short entry: bearish divergence and MACD histogram turns negative
        signals['enter_short'] = df['bearish_divergence'] & (df['macd_hist'] < 0) & (df['macd_hist'].shift(1) >= 0)
        
        # Generate exit signals
        # Exit long: MACD crosses below signal line
        signals['exit_long'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        # Exit short: MACD crosses above signal line
        signals['exit_short'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        
        logger.info(f"Generated MACD mean reversion signals with fast={fast_period}, slow={slow_period}, signal={signal_period}")
        return signals


class StatisticalMeanReversionStrategy(Strategy):
    """
    Statistical mean reversion strategy using z-scores.
    
    This strategy generates signals when price deviates significantly from its moving average,
    assuming it will revert back to the mean.
    """
    
    def __init__(self, window=20, entry_zscore=2.0, exit_zscore=0.5, max_holding_period=10):
        """
        Initialize the statistical mean reversion strategy.
        
        Parameters:
        -----------
        window : int, optional
            Window period for calculating moving average and standard deviation.
        entry_zscore : float, optional
            Z-score threshold for entry signals.
        exit_zscore : float, optional
            Z-score threshold for exit signals.
        max_holding_period : int, optional
            Maximum number of bars to hold a position.
        """
        parameters = {
            'window': window,
            'entry_zscore': entry_zscore,
            'exit_zscore': exit_zscore,
            'max_holding_period': max_holding_period
        }
        super().__init__(name="StatisticalMeanReversion", parameters=parameters)
    
    def generate_signals(self, data):
        """
        Generate trading signals based on statistical mean reversion.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical price data with OHLCV columns.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with signal columns.
        """
        # Get parameters
        window = self.parameters['window']
        entry_zscore = self.parameters['entry_zscore']
        exit_zscore = self.parameters['exit_zscore']
        max_holding_period = self.parameters['max_holding_period']
        
        # Make a copy of the data
        df = data.copy()
        
        # Calculate rolling mean and standard deviation
        df['rolling_mean'] = df['close'].rolling(window=window).mean()
        df['rolling_std'] = df['close'].rolling(window=window).std()
        
        # Calculate z-score
        df['zscore'] = (df['close'] - df['rolling_mean']) / df['rolling_std']
        
        # Initialize signal columns
        signals = pd.DataFrame(index=df.index)
        signals['enter_long'] = False
        signals['enter_short'] = False
        signals['exit_long'] = False
        signals['exit_short'] = False
        
        # Generate entry signals
        # Long entry: z-score below negative entry threshold
        signals['enter_long'] = df['zscore'] < -entry_zscore
        
        # Short entry: z-score above positive entry threshold
        signals['enter_short'] = df['zscore'] > entry_zscore
        
        # Generate exit signals
        # Exit long: z-score above negative exit threshold
        signals['exit_long'] = df['zscore'] > -exit_zscore
        
        # Exit short: z-score below positive exit threshold
        signals['exit_short'] = df['zscore'] < exit_zscore
        
        # Add time-based exit
        if max_holding_period > 0:
            # Create rolling window for entry signals
            long_entry_window = signals['enter_long'].rolling(window=max_holding_period).sum()
            short_entry_window = signals['enter_short'].rolling(window=max_holding_period).sum()
            
            # Exit if we've been in position for max_holding_period bars
            signals['exit_long'] = signals['exit_long'] | (long_entry_window.shift(max_holding_period) > 0)
            signals['exit_short'] = signals['exit_short'] | (short_entry_window.shift(max_holding_period) > 0)
        
        logger.info(f"Generated statistical mean reversion signals with window={window}, entry_zscore={entry_zscore}")
        return signals


class MeanReversionEnsemble(Strategy):
    """
    Ensemble of multiple mean reversion strategies.
    """
    
    def __init__(self, use_bollinger=True, use_rsi=True, use_macd=True, use_statistical=True):
        """
        Initialize the mean reversion ensemble strategy.
        
        Parameters:
        -----------
        use_bollinger : bool, optional
            Whether to include Bollinger Band strategy.
        use_rsi : bool, optional
            Whether to include RSI strategy.
        use_macd : bool, optional
            Whether to include MACD strategy.
        use_statistical : bool, optional
            Whether to include Statistical strategy.
        """
        parameters = {
            'use_bollinger': use_bollinger,
            'use_rsi': use_rsi,
            'use_macd': use_macd,
            'use_statistical': use_statistical
        }
        super().__init__(name="MeanReversionEnsemble", parameters=parameters)
        
        # Create individual strategies
        self.strategies = []
        
        if use_bollinger:
            self.strategies.append(BollingerBandStrategy())
        
        if use_rsi:
            self.strategies.append(RSIMeanReversionStrategy())
        
        if use_macd:
            self.strategies.append(MACDMeanReversionStrategy())
        
        if use_statistical:
            self.strategies.append(StatisticalMeanReversionStrategy())
        
        logger.info(f"Initialized MeanReversionEnsemble with {len(self.strategies)} strategies")
    
    def generate_signals(self, data):
        """
        Generate trading signals by combining signals from all strategies.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical price data with OHLCV columns.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with signal columns.
        """
        if not self.strategies:
            logger.warning("MeanReversionEnsemble has no strategies")
            return pd.DataFrame(index=data.index)
        
        # Generate signals for each strategy
        all_signals = []
        for strategy in self.strategies:
            signals = strategy.generate_signals(data)
            all_signals.append(signals)
        
        # Combine signals (voting mechanism)
        combined = pd.DataFrame(index=data.index)
        combined['enter_long_votes'] = 0
        combined['enter_short_votes'] = 0
        combined['exit_long_votes'] = 0
        combined['exit_short_votes'] = 0
        
        for signals in all_signals:
            combined['enter_long_votes'] += signals['enter_long'].astype(int)
            combined['enter_short_votes'] += signals['enter_short'].astype(int)
            combined['exit_long_votes'] += signals['exit_long'].astype(int)
            combined['exit_short_votes'] += signals['exit_short'].astype(int)
        
        # Threshold for final decision (majority vote)
        threshold = len(self.strategies) / 2
        
        # Create final signal columns
        combined['enter_long'] = combined['enter_long_votes'] > threshold
        combined['enter_short'] = combined['enter_short_votes'] > threshold
        combined['exit_long'] = combined['exit_long_votes'] > threshold
        combined['exit_short'] = combined['exit_short_votes'] > threshold
        
        # Keep only the signal columns
        signals = combined[['enter_long', 'enter_short', 'exit_long', 'exit_short']]
        
        logger.info(f"Generated ensemble signals from {len(self.strategies)} mean reversion strategies")
        return signals


if __name__ == "__main__":
    # Example usage
    print("Mean Reversion Strategies module ready for use")

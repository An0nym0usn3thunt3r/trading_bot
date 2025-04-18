"""
Momentum and Trend Following Strategies for Nasdaq-100 E-mini futures trading bot.
This module implements various momentum and trend following trading strategies.
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


class MovingAverageCrossoverStrategy(Strategy):
    """
    Moving Average Crossover strategy.
    
    This strategy generates buy signals when a faster moving average crosses above
    a slower moving average, and sell signals when it crosses below.
    """
    
    def __init__(self, fast_window=10, slow_window=50, use_ema=False, trend_filter=False, trend_window=200):
        """
        Initialize the Moving Average Crossover strategy.
        
        Parameters:
        -----------
        fast_window : int, optional
            Window period for the faster moving average.
        slow_window : int, optional
            Window period for the slower moving average.
        use_ema : bool, optional
            Whether to use exponential moving averages instead of simple moving averages.
        trend_filter : bool, optional
            Whether to use a longer-term trend filter.
        trend_window : int, optional
            Window period for the trend filter moving average.
        """
        parameters = {
            'fast_window': fast_window,
            'slow_window': slow_window,
            'use_ema': use_ema,
            'trend_filter': trend_filter,
            'trend_window': trend_window
        }
        super().__init__(name="MovingAverageCrossover", parameters=parameters)
    
    def generate_signals(self, data):
        """
        Generate trading signals based on moving average crossovers.
        
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
        fast_window = self.parameters['fast_window']
        slow_window = self.parameters['slow_window']
        use_ema = self.parameters['use_ema']
        trend_filter = self.parameters['trend_filter']
        trend_window = self.parameters['trend_window']
        
        # Make a copy of the data
        df = data.copy()
        
        # Calculate moving averages
        if use_ema:
            # Exponential Moving Averages
            df['fast_ma'] = df['close'].ewm(span=fast_window, adjust=False).mean()
            df['slow_ma'] = df['close'].ewm(span=slow_window, adjust=False).mean()
            
            if trend_filter:
                df['trend_ma'] = df['close'].ewm(span=trend_window, adjust=False).mean()
        else:
            # Simple Moving Averages
            df['fast_ma'] = df['close'].rolling(window=fast_window).mean()
            df['slow_ma'] = df['close'].rolling(window=slow_window).mean()
            
            if trend_filter:
                df['trend_ma'] = df['close'].rolling(window=trend_window).mean()
        
        # Initialize signal columns
        signals = pd.DataFrame(index=df.index)
        signals['enter_long'] = False
        signals['enter_short'] = False
        signals['exit_long'] = False
        signals['exit_short'] = False
        
        # Generate crossover signals
        # Buy signal: fast MA crosses above slow MA
        crossover_up = (df['fast_ma'] > df['slow_ma']) & (df['fast_ma'].shift(1) <= df['slow_ma'].shift(1))
        
        # Sell signal: fast MA crosses below slow MA
        crossover_down = (df['fast_ma'] < df['slow_ma']) & (df['fast_ma'].shift(1) >= df['slow_ma'].shift(1))
        
        # Apply trend filter if enabled
        if trend_filter:
            # Only go long if price is above trend MA
            signals['enter_long'] = crossover_up & (df['close'] > df['trend_ma'])
            
            # Only go short if price is below trend MA
            signals['enter_short'] = crossover_down & (df['close'] < df['trend_ma'])
        else:
            signals['enter_long'] = crossover_up
            signals['enter_short'] = crossover_down
        
        # Exit signals are the opposite crossovers
        signals['exit_long'] = crossover_down
        signals['exit_short'] = crossover_up
        
        logger.info(f"Generated Moving Average Crossover signals with fast={fast_window}, slow={slow_window}, use_ema={use_ema}")
        return signals


class MACDMomentumStrategy(Strategy):
    """
    MACD (Moving Average Convergence Divergence) momentum strategy.
    
    This strategy generates signals based on MACD line crossing the signal line.
    """
    
    def __init__(self, fast_period=12, slow_period=26, signal_period=9, histogram_filter=True):
        """
        Initialize the MACD momentum strategy.
        
        Parameters:
        -----------
        fast_period : int, optional
            Fast EMA period for MACD calculation.
        slow_period : int, optional
            Slow EMA period for MACD calculation.
        signal_period : int, optional
            Signal line EMA period.
        histogram_filter : bool, optional
            Whether to use histogram direction as an additional filter.
        """
        parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
            'histogram_filter': histogram_filter
        }
        super().__init__(name="MACDMomentum", parameters=parameters)
    
    def generate_signals(self, data):
        """
        Generate trading signals based on MACD.
        
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
        histogram_filter = self.parameters['histogram_filter']
        
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
        
        # Initialize signal columns
        signals = pd.DataFrame(index=df.index)
        signals['enter_long'] = False
        signals['enter_short'] = False
        signals['exit_long'] = False
        signals['exit_short'] = False
        
        # Generate entry signals
        # Long entry: MACD crosses above signal line
        macd_cross_up = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        
        # Short entry: MACD crosses below signal line
        macd_cross_down = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        if histogram_filter:
            # Add histogram direction filter
            hist_increasing = df['macd_hist'] > df['macd_hist'].shift(1)
            hist_decreasing = df['macd_hist'] < df['macd_hist'].shift(1)
            
            signals['enter_long'] = macd_cross_up & hist_increasing
            signals['enter_short'] = macd_cross_down & hist_decreasing
        else:
            signals['enter_long'] = macd_cross_up
            signals['enter_short'] = macd_cross_down
        
        # Generate exit signals
        # Exit long: MACD crosses below signal line
        signals['exit_long'] = macd_cross_down
        
        # Exit short: MACD crosses above signal line
        signals['exit_short'] = macd_cross_up
        
        logger.info(f"Generated MACD momentum signals with fast={fast_period}, slow={slow_period}, signal={signal_period}")
        return signals


class RSIMomentumStrategy(Strategy):
    """
    RSI (Relative Strength Index) momentum strategy.
    
    This strategy generates signals based on RSI crossing certain thresholds.
    """
    
    def __init__(self, rsi_window=14, entry_threshold=50, exit_bars=5, trend_filter=False, trend_window=100):
        """
        Initialize the RSI momentum strategy.
        
        Parameters:
        -----------
        rsi_window : int, optional
            Window period for calculating RSI.
        entry_threshold : int, optional
            RSI threshold for entry signals (above for long, below for short).
        exit_bars : int, optional
            Number of bars to hold position before exiting.
        trend_filter : bool, optional
            Whether to use a moving average trend filter.
        trend_window : int, optional
            Window period for the trend filter moving average.
        """
        parameters = {
            'rsi_window': rsi_window,
            'entry_threshold': entry_threshold,
            'exit_bars': exit_bars,
            'trend_filter': trend_filter,
            'trend_window': trend_window
        }
        super().__init__(name="RSIMomentum", parameters=parameters)
    
    def generate_signals(self, data):
        """
        Generate trading signals based on RSI momentum.
        
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
        entry_threshold = self.parameters['entry_threshold']
        exit_bars = self.parameters['exit_bars']
        trend_filter = self.parameters['trend_filter']
        trend_window = self.parameters['trend_window']
        
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
        
        # Calculate trend filter if enabled
        if trend_filter:
            df['trend_ma'] = df['close'].rolling(window=trend_window).mean()
        
        # Initialize signal columns
        signals = pd.DataFrame(index=df.index)
        signals['enter_long'] = False
        signals['enter_short'] = False
        signals['exit_long'] = False
        signals['exit_short'] = False
        
        # Generate entry signals
        # Long entry: RSI crosses above threshold
        rsi_cross_up = (df[f'rsi_{rsi_window}'] > entry_threshold) & (df[f'rsi_{rsi_window}'].shift(1) <= entry_threshold)
        
        # Short entry: RSI crosses below (100 - threshold)
        rsi_cross_down = (df[f'rsi_{rsi_window}'] < (100 - entry_threshold)) & (df[f'rsi_{rsi_window}'].shift(1) >= (100 - entry_threshold))
        
        if trend_filter:
            # Only go long if price is above trend MA
            signals['enter_long'] = rsi_cross_up & (df['close'] > df['trend_ma'])
            
            # Only go short if price is below trend MA
            signals['enter_short'] = rsi_cross_down & (df['close'] < df['trend_ma'])
        else:
            signals['enter_long'] = rsi_cross_up
            signals['enter_short'] = rsi_cross_down
        
        # Generate exit signals based on time
        if exit_bars > 0:
            # Create rolling window for entry signals
            for i in range(len(signals)):
                if i >= exit_bars:
                    # Exit long if we entered long exit_bars ago
                    if signals['enter_long'].iloc[i - exit_bars]:
                        signals['exit_long'].iloc[i] = True
                    
                    # Exit short if we entered short exit_bars ago
                    if signals['enter_short'].iloc[i - exit_bars]:
                        signals['exit_short'].iloc[i] = True
        
        # Also exit on opposite signals
        signals['exit_long'] = signals['exit_long'] | signals['enter_short']
        signals['exit_short'] = signals['exit_short'] | signals['enter_long']
        
        logger.info(f"Generated RSI momentum signals with window={rsi_window}, threshold={entry_threshold}")
        return signals


class ADXTrendStrategy(Strategy):
    """
    ADX (Average Directional Index) trend following strategy.
    
    This strategy uses ADX to identify strong trends and directional movement
    indicators (DMI) to determine trend direction.
    """
    
    def __init__(self, adx_window=14, adx_threshold=25, use_parabolic_sar=False, sar_step=0.02, sar_max=0.2):
        """
        Initialize the ADX trend strategy.
        
        Parameters:
        -----------
        adx_window : int, optional
            Window period for calculating ADX.
        adx_threshold : int, optional
            ADX threshold for identifying strong trends.
        use_parabolic_sar : bool, optional
            Whether to use Parabolic SAR for exit signals.
        sar_step : float, optional
            Step parameter for Parabolic SAR calculation.
        sar_max : float, optional
            Maximum step parameter for Parabolic SAR calculation.
        """
        parameters = {
            'adx_window': adx_window,
            'adx_threshold': adx_threshold,
            'use_parabolic_sar': use_parabolic_sar,
            'sar_step': sar_step,
            'sar_max': sar_max
        }
        super().__init__(name="ADXTrend", parameters=parameters)
    
    def generate_signals(self, data):
        """
        Generate trading signals based on ADX and DMI.
        
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
        adx_window = self.parameters['adx_window']
        adx_threshold = self.parameters['adx_threshold']
        use_parabolic_sar = self.parameters['use_parabolic_sar']
        sar_step = self.parameters['sar_step']
        sar_max = self.parameters['sar_max']
        
        # Make a copy of the data
        df = data.copy()
        
        # Calculate ADX and DMI if not already present
        if f'adx_{adx_window}' not in df.columns:
            # True Range
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Directional Movement
            up_move = df['high'] - df['high'].shift()
            down_move = df['low'].shift() - df['low']
            
            pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Smoothed True Range and Directional Movement
            atr = tr.rolling(window=adx_window).mean()
            pos_di = 100 * (pd.Series(pos_dm).rolling(window=adx_window).mean() / atr)
            neg_di = 100 * (pd.Series(neg_dm).rolling(window=adx_window).mean() / atr)
            
            # ADX
            dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
            df[f'adx_{adx_window}'] = dx.rolling(window=adx_window).mean()
            df[f'pos_di_{adx_window}'] = pos_di
            df[f'neg_di_{adx_window}'] = neg_di
        
        # Calculate Parabolic SAR if needed
        if use_parabolic_sar and 'sar' not in df.columns:
            # This is a simplified implementation of Parabolic SAR
            df['sar'] = np.nan
            df['sar_direction'] = 0  # 1 for uptrend, -1 for downtrend
            
            # Initialize SAR with first bar's low for uptrend, high for downtrend
            if df['close'].iloc[0] > df['close'].shift().iloc[0]:
                df['sar'].iloc[0] = df['low'].iloc[0]
                df['sar_direction'].iloc[0] = 1
            else:
                df['sar'].iloc[0] = df['high'].iloc[0]
                df['sar_direction'].iloc[0] = -1
            
            # Calculate SAR for each bar
            for i in range(1, len(df)):
                prev_sar = df['sar'].iloc[i-1]
                direction = df['sar_direction'].iloc[i-1]
                
                # Extreme point
                if direction == 1:
                    ep = max(df['high'].iloc[i-1], df['high'].iloc[i])
                else:
                    ep = min(df['low'].iloc[i-1], df['low'].iloc[i])
                
                # Calculate acceleration factor
                af = min(sar_step * (i // 2), sar_max)
                
                # Calculate new SAR
                new_sar = prev_sar + af * (ep - prev_sar)
                
                # Check if SAR is penetrated
                if direction == 1:
                    # In uptrend, SAR must be below the lows
                    new_sar = min(new_sar, df['low'].iloc[i-1], df['low'].iloc[i-2] if i > 1 else df['low'].iloc[i-1])
                    
                    # Check if trend reverses
                    if new_sar > df['low'].iloc[i]:
                        direction = -1
                        new_sar = max(df['high'].iloc[i-1], df['high'].iloc[i])
                else:
                    # In downtrend, SAR must be above the highs
                    new_sar = max(new_sar, df['high'].iloc[i-1], df['high'].iloc[i-2] if i > 1 else df['high'].iloc[i-1])
                    
                    # Check if trend reverses
                    if new_sar < df['high'].iloc[i]:
                        direction = 1
                        new_sar = min(df['low'].iloc[i-1], df['low'].iloc[i])
                
                df['sar'].iloc[i] = new_sar
                df['sar_direction'].iloc[i] = direction
        
        # Initialize signal columns
        signals = pd.DataFrame(index=df.index)
        signals['enter_long'] = False
        signals['enter_short'] = False
        signals['exit_long'] = False
        signals['exit_short'] = False
        
        # Generate entry signals
        # Strong trend condition
        strong_trend = df[f'adx_{adx_window}'] > adx_threshold
        
        # Long entry: ADX above threshold and +DI crosses above -DI
        di_cross_up = (df[f'pos_di_{adx_window}'] > df[f'neg_di_{adx_window}']) & (df[f'pos_di_{adx_window}'].shift(1) <= df[f'neg_di_{adx_window}'].shift(1))
        signals['enter_long'] = strong_trend & di_cross_up
        
        # Short entry: ADX above threshold and -DI crosses above +DI
        di_cross_down = (df[f'neg_di_{adx_window}'] > df[f'pos_di_{adx_window}']) & (df[f'neg_di_{adx_window}'].shift(1) <= df[f'pos_di_{adx_window}'].shift(1))
        signals['enter_short'] = strong_trend & di_cross_down
        
        # Generate exit signals
        if use_parabolic_sar:
            # Exit long: SAR crosses above price
            signals['exit_long'] = (df['sar'] > df['close']) & (df['sar'].shift(1) <= df['close'].shift(1))
            
            # Exit short: SAR crosses below price
            signals['exit_short'] = (df['sar'] < df['close']) & (df['sar'].shift(1) >= df['close'].shift(1))
        else:
            # Exit on opposite entry signals
            signals['exit_long'] = signals['enter_short']
            signals['exit_short'] = signals['enter_long']
        
        logger.info(f"Generated ADX trend signals with window={adx_window}, threshold={adx_threshold}")
        return signals


class MomentumEnsemble(Strategy):
    """
    Ensemble of multiple momentum and trend following strategies.
    """
    
    def __init__(self, use_ma_crossover=True, use_macd=True, use_rsi=True, use_adx=True):
        """
        Initialize the momentum ensemble strategy.
        
        Parameters:
        -----------
        use_ma_crossover : bool, optional
            Whether to include Moving Average Crossover strategy.
        use_macd : bool, optional
            Whether to include MACD strategy.
        use_rsi : bool, optional
            Whether to include RSI strategy.
        use_adx : bool, optional
            Whether to include ADX strategy.
        """
        parameters = {
            'use_ma_crossover': use_ma_crossover,
            'use_macd': use_macd,
            'use_rsi': use_rsi,
            'use_adx': use_adx
        }
        super().__init__(name="MomentumEnsemble", parameters=parameters)
        
        # Create individual strategies
        self.strategies = []
        
        if use_ma_crossover:
            self.strategies.append(MovingAverageCrossoverStrategy())
        
        if use_macd:
            self.strategies.append(MACDMomentumStrategy())
        
        if use_rsi:
            self.strategies.append(RSIMomentumStrategy())
        
        if use_adx:
            self.strategies.append(ADXTrendStrategy())
        
        logger.info(f"Initialized MomentumEnsemble with {len(self.strategies)} strategies")
    
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
            logger.warning("MomentumEnsemble has no strategies")
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
        
        logger.info(f"Generated ensemble signals from {len(self.strategies)} momentum strategies")
        return signals


if __name__ == "__main__":
    # Example usage
    print("Momentum and Trend Following Strategies module ready for use")

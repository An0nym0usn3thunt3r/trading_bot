"""
Breakout and Range Trading Strategies for Nasdaq-100 E-mini futures trading bot.
This module implements various breakout and range trading strategies.
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


class DonchianChannelBreakoutStrategy(Strategy):
    """
    Donchian Channel Breakout strategy.
    
    This strategy generates buy signals when price breaks above the upper Donchian Channel
    and sell signals when price breaks below the lower Donchian Channel.
    """
    
    def __init__(self, window=20, exit_window=10, atr_filter=True, atr_window=14, atr_multiplier=1.5):
        """
        Initialize the Donchian Channel Breakout strategy.
        
        Parameters:
        -----------
        window : int, optional
            Window period for calculating Donchian Channels.
        exit_window : int, optional
            Window for exit signals (shorter period Donchian Channel).
        atr_filter : bool, optional
            Whether to use ATR for filtering breakouts.
        atr_window : int, optional
            Window period for ATR calculation.
        atr_multiplier : float, optional
            Multiplier for ATR to determine significant breakouts.
        """
        parameters = {
            'window': window,
            'exit_window': exit_window,
            'atr_filter': atr_filter,
            'atr_window': atr_window,
            'atr_multiplier': atr_multiplier
        }
        super().__init__(name="DonchianChannelBreakout", parameters=parameters)
    
    def generate_signals(self, data):
        """
        Generate trading signals based on Donchian Channel breakouts.
        
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
        exit_window = self.parameters['exit_window']
        atr_filter = self.parameters['atr_filter']
        atr_window = self.parameters['atr_window']
        atr_multiplier = self.parameters['atr_multiplier']
        
        # Make a copy of the data
        df = data.copy()
        
        # Calculate Donchian Channels
        df[f'upper_channel_{window}'] = df['high'].rolling(window=window).max()
        df[f'lower_channel_{window}'] = df['low'].rolling(window=window).min()
        df[f'middle_channel_{window}'] = (df[f'upper_channel_{window}'] + df[f'lower_channel_{window}']) / 2
        
        # Calculate exit channels if different from entry
        if exit_window != window:
            df[f'upper_channel_{exit_window}'] = df['high'].rolling(window=exit_window).max()
            df[f'lower_channel_{exit_window}'] = df['low'].rolling(window=exit_window).min()
        
        # Calculate ATR if needed
        if atr_filter:
            # True Range
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Average True Range
            df[f'atr_{atr_window}'] = tr.rolling(window=atr_window).mean()
        
        # Initialize signal columns
        signals = pd.DataFrame(index=df.index)
        signals['enter_long'] = False
        signals['enter_short'] = False
        signals['exit_long'] = False
        signals['exit_short'] = False
        
        # Generate entry signals
        # Long entry: price breaks above upper channel
        breakout_up = (df['close'] > df[f'upper_channel_{window}'].shift(1))
        
        # Short entry: price breaks below lower channel
        breakout_down = (df['close'] < df[f'lower_channel_{window}'].shift(1))
        
        # Apply ATR filter if enabled
        if atr_filter:
            # Only consider significant breakouts (greater than ATR * multiplier)
            upper_distance = df['close'] - df[f'upper_channel_{window}'].shift(1)
            lower_distance = df[f'lower_channel_{window}'].shift(1) - df['close']
            
            significant_up = upper_distance > (df[f'atr_{atr_window}'] * atr_multiplier)
            significant_down = lower_distance > (df[f'atr_{atr_window}'] * atr_multiplier)
            
            signals['enter_long'] = breakout_up & significant_up
            signals['enter_short'] = breakout_down & significant_down
        else:
            signals['enter_long'] = breakout_up
            signals['enter_short'] = breakout_down
        
        # Generate exit signals
        # Exit long: price breaks below middle channel or lower exit channel
        if exit_window != window:
            signals['exit_long'] = df['close'] < df[f'lower_channel_{exit_window}'].shift(1)
            signals['exit_short'] = df['close'] > df[f'upper_channel_{exit_window}'].shift(1)
        else:
            signals['exit_long'] = df['close'] < df[f'middle_channel_{window}'].shift(1)
            signals['exit_short'] = df['close'] > df[f'middle_channel_{window}'].shift(1)
        
        logger.info(f"Generated Donchian Channel Breakout signals with window={window}, exit_window={exit_window}")
        return signals


class VolatilityBreakoutStrategy(Strategy):
    """
    Volatility Breakout strategy.
    
    This strategy generates signals when price breaks out of a volatility-based range.
    """
    
    def __init__(self, window=20, volatility_window=10, volatility_multiplier=2.0, use_atr=True):
        """
        Initialize the Volatility Breakout strategy.
        
        Parameters:
        -----------
        window : int, optional
            Window period for calculating the base price.
        volatility_window : int, optional
            Window period for calculating volatility.
        volatility_multiplier : float, optional
            Multiplier for volatility to determine breakout thresholds.
        use_atr : bool, optional
            Whether to use ATR for volatility (True) or standard deviation (False).
        """
        parameters = {
            'window': window,
            'volatility_window': volatility_window,
            'volatility_multiplier': volatility_multiplier,
            'use_atr': use_atr
        }
        super().__init__(name="VolatilityBreakout", parameters=parameters)
    
    def generate_signals(self, data):
        """
        Generate trading signals based on volatility breakouts.
        
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
        volatility_window = self.parameters['volatility_window']
        volatility_multiplier = self.parameters['volatility_multiplier']
        use_atr = self.parameters['use_atr']
        
        # Make a copy of the data
        df = data.copy()
        
        # Calculate base price (moving average)
        df['base_price'] = df['close'].rolling(window=window).mean()
        
        # Calculate volatility
        if use_atr:
            # True Range
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Average True Range
            df['volatility'] = tr.rolling(window=volatility_window).mean()
        else:
            # Standard deviation
            df['volatility'] = df['close'].rolling(window=volatility_window).std()
        
        # Calculate breakout levels
        df['upper_level'] = df['base_price'] + (df['volatility'] * volatility_multiplier)
        df['lower_level'] = df['base_price'] - (df['volatility'] * volatility_multiplier)
        
        # Initialize signal columns
        signals = pd.DataFrame(index=df.index)
        signals['enter_long'] = False
        signals['enter_short'] = False
        signals['exit_long'] = False
        signals['exit_short'] = False
        
        # Generate entry signals
        # Long entry: price breaks above upper level
        signals['enter_long'] = (df['close'] > df['upper_level'].shift(1)) & (df['close'].shift(1) <= df['upper_level'].shift(1))
        
        # Short entry: price breaks below lower level
        signals['enter_short'] = (df['close'] < df['lower_level'].shift(1)) & (df['close'].shift(1) >= df['lower_level'].shift(1))
        
        # Generate exit signals
        # Exit long: price crosses below base price
        signals['exit_long'] = (df['close'] < df['base_price']) & (df['close'].shift(1) >= df['base_price'].shift(1))
        
        # Exit short: price crosses above base price
        signals['exit_short'] = (df['close'] > df['base_price']) & (df['close'].shift(1) <= df['base_price'].shift(1))
        
        logger.info(f"Generated Volatility Breakout signals with window={window}, volatility_window={volatility_window}")
        return signals


class RangeBreakoutStrategy(Strategy):
    """
    Range Breakout strategy.
    
    This strategy identifies consolidation ranges and generates signals when price breaks out.
    """
    
    def __init__(self, range_window=20, threshold_pct=0.5, confirmation_bars=2, exit_after_bars=10):
        """
        Initialize the Range Breakout strategy.
        
        Parameters:
        -----------
        range_window : int, optional
            Window period for identifying price ranges.
        threshold_pct : float, optional
            Threshold percentage for identifying consolidation (lower = tighter range).
        confirmation_bars : int, optional
            Number of bars to confirm a breakout.
        exit_after_bars : int, optional
            Number of bars to exit after a breakout if no other exit signal.
        """
        parameters = {
            'range_window': range_window,
            'threshold_pct': threshold_pct,
            'confirmation_bars': confirmation_bars,
            'exit_after_bars': exit_after_bars
        }
        super().__init__(name="RangeBreakout", parameters=parameters)
    
    def generate_signals(self, data):
        """
        Generate trading signals based on range breakouts.
        
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
        range_window = self.parameters['range_window']
        threshold_pct = self.parameters['threshold_pct']
        confirmation_bars = self.parameters['confirmation_bars']
        exit_after_bars = self.parameters['exit_after_bars']
        
        # Make a copy of the data
        df = data.copy()
        
        # Calculate range metrics
        df['range_high'] = df['high'].rolling(window=range_window).max()
        df['range_low'] = df['low'].rolling(window=range_window).min()
        df['range_size'] = df['range_high'] - df['range_low']
        df['range_midpoint'] = (df['range_high'] + df['range_low']) / 2
        
        # Calculate average price and range percentage
        df['avg_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['range_pct'] = df['range_size'] / df['avg_price'] * 100
        
        # Identify consolidation periods (range percentage below threshold)
        df['in_consolidation'] = df['range_pct'] < threshold_pct
        
        # Initialize signal columns
        signals = pd.DataFrame(index=df.index)
        signals['enter_long'] = False
        signals['enter_short'] = False
        signals['exit_long'] = False
        signals['exit_short'] = False
        
        # Track breakout status
        breakout_status = pd.Series(index=df.index, data=0)  # 0: none, 1: upside, -1: downside
        
        # Generate signals
        for i in range(range_window, len(df)):
            # Check if we're in consolidation
            if df['in_consolidation'].iloc[i-1]:
                # Check for breakout
                if df['close'].iloc[i] > df['range_high'].iloc[i-1]:
                    # Potential upside breakout
                    breakout_status.iloc[i] = 1
                elif df['close'].iloc[i] < df['range_low'].iloc[i-1]:
                    # Potential downside breakout
                    breakout_status.iloc[i] = -1
            
            # Check for breakout confirmation
            if i >= confirmation_bars:
                # Upside breakout confirmation
                if (breakout_status.iloc[i-confirmation_bars:i] == 1).all():
                    signals['enter_long'].iloc[i] = True
                
                # Downside breakout confirmation
                if (breakout_status.iloc[i-confirmation_bars:i] == -1).all():
                    signals['enter_short'].iloc[i] = True
            
            # Exit signals
            if i >= exit_after_bars:
                # Exit long after specified bars
                if signals['enter_long'].iloc[i-exit_after_bars]:
                    signals['exit_long'].iloc[i] = True
                
                # Exit short after specified bars
                if signals['enter_short'].iloc[i-exit_after_bars]:
                    signals['exit_short'].iloc[i] = True
            
            # Also exit when price crosses back into the range
            if breakout_status.iloc[i-1] == 1 and df['close'].iloc[i] < df['range_midpoint'].iloc[i-1]:
                signals['exit_long'].iloc[i] = True
            
            if breakout_status.iloc[i-1] == -1 and df['close'].iloc[i] > df['range_midpoint'].iloc[i-1]:
                signals['exit_short'].iloc[i] = True
        
        logger.info(f"Generated Range Breakout signals with window={range_window}, threshold={threshold_pct}%")
        return signals


class SupportResistanceStrategy(Strategy):
    """
    Support and Resistance trading strategy.
    
    This strategy identifies key support and resistance levels and generates signals
    when price bounces off or breaks through these levels.
    """
    
    def __init__(self, window=50, num_levels=3, level_threshold=0.5, bounce_mode=True):
        """
        Initialize the Support and Resistance strategy.
        
        Parameters:
        -----------
        window : int, optional
            Window period for identifying support/resistance levels.
        num_levels : int, optional
            Number of support/resistance levels to track.
        level_threshold : float, optional
            Percentage threshold for level proximity.
        bounce_mode : bool, optional
            If True, trade bounces off levels; if False, trade breakouts.
        """
        parameters = {
            'window': window,
            'num_levels': num_levels,
            'level_threshold': level_threshold,
            'bounce_mode': bounce_mode
        }
        super().__init__(name="SupportResistance", parameters=parameters)
    
    def _find_peaks(self, series, window):
        """Find peaks in a series using a rolling window."""
        # A point is a peak if it's the maximum in the window
        return series == series.rolling(window=window, center=True).max()
    
    def _find_troughs(self, series, window):
        """Find troughs in a series using a rolling window."""
        # A point is a trough if it's the minimum in the window
        return series == series.rolling(window=window, center=True).min()
    
    def generate_signals(self, data):
        """
        Generate trading signals based on support and resistance levels.
        
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
        num_levels = self.parameters['num_levels']
        level_threshold = self.parameters['level_threshold']
        bounce_mode = self.parameters['bounce_mode']
        
        # Make a copy of the data
        df = data.copy()
        
        # Find peaks and troughs
        peaks = self._find_peaks(df['high'], window)
        troughs = self._find_troughs(df['low'], window)
        
        # Extract resistance levels (from peaks)
        resistance_levels = df.loc[peaks, 'high'].sort_values(ascending=False).head(num_levels).values
        
        # Extract support levels (from troughs)
        support_levels = df.loc[troughs, 'low'].sort_values(ascending=True).head(num_levels).values
        
        # Initialize signal columns
        signals = pd.DataFrame(index=df.index)
        signals['enter_long'] = False
        signals['enter_short'] = False
        signals['exit_long'] = False
        signals['exit_short'] = False
        
        # Generate signals
        for i in range(1, len(df)):
            current_price = df['close'].iloc[i]
            previous_price = df['close'].iloc[i-1]
            
            # Calculate price movement
            price_change = (current_price - previous_price) / previous_price * 100
            
            if bounce_mode:
                # Trade bounces off support/resistance
                
                # Check for bounce off support (long entry)
                for level in support_levels:
                    # Price approached support and then moved up
                    level_proximity = abs(df['low'].iloc[i-1] - level) / level * 100
                    if level_proximity < level_threshold and price_change > 0:
                        signals['enter_long'].iloc[i] = True
                        break
                
                # Check for bounce off resistance (short entry)
                for level in resistance_levels:
                    # Price approached resistance and then moved down
                    level_proximity = abs(df['high'].iloc[i-1] - level) / level * 100
                    if level_proximity < level_threshold and price_change < 0:
                        signals['enter_short'].iloc[i] = True
                        break
                
                # Exit signals - when price approaches the opposite level type
                if signals['enter_long'].iloc[i-1]:
                    for level in resistance_levels:
                        level_proximity = abs(df['high'].iloc[i] - level) / level * 100
                        if level_proximity < level_threshold:
                            signals['exit_long'].iloc[i] = True
                            break
                
                if signals['enter_short'].iloc[i-1]:
                    for level in support_levels:
                        level_proximity = abs(df['low'].iloc[i] - level) / level * 100
                        if level_proximity < level_threshold:
                            signals['exit_short'].iloc[i] = True
                            break
            else:
                # Trade breakouts through support/resistance
                
                # Check for breakout above resistance (long entry)
                for level in resistance_levels:
                    # Price broke above resistance
                    if previous_price < level and current_price > level:
                        signals['enter_long'].iloc[i] = True
                        break
                
                # Check for breakdown below support (short entry)
                for level in support_levels:
                    # Price broke below support
                    if previous_price > level and current_price < level:
                        signals['enter_short'].iloc[i] = True
                        break
                
                # Exit signals - when price reverses back through the level
                if signals['enter_long'].iloc[i-1]:
                    for level in resistance_levels:
                        if previous_price > level and current_price < level:
                            signals['exit_long'].iloc[i] = True
                            break
                
                if signals['enter_short'].iloc[i-1]:
                    for level in support_levels:
                        if previous_price < level and current_price > level:
                            signals['exit_short'].iloc[i] = True
                            break
        
        logger.info(f"Generated Support/Resistance signals with window={window}, num_levels={num_levels}, bounce_mode={bounce_mode}")
        return signals


class BreakoutEnsemble(Strategy):
    """
    Ensemble of multiple breakout and range trading strategies.
    """
    
    def __init__(self, use_donchian=True, use_volatility=True, use_range=True, use_support_resistance=True):
        """
        Initialize the breakout ensemble strategy.
        
        Parameters:
        -----------
        use_donchian : bool, optional
            Whether to include Donchian Channel Breakout strategy.
        use_volatility : bool, optional
            Whether to include Volatility Breakout strategy.
        use_range : bool, optional
            Whether to include Range Breakout strategy.
        use_support_resistance : bool, optional
            Whether to include Support/Resistance strategy.
        """
        parameters = {
            'use_donchian': use_donchian,
            'use_volatility': use_volatility,
            'use_range': use_range,
            'use_support_resistance': use_support_resistance
        }
        super().__init__(name="BreakoutEnsemble", parameters=parameters)
        
        # Create individual strategies
        self.strategies = []
        
        if use_donchian:
            self.strategies.append(DonchianChannelBreakoutStrategy())
        
        if use_volatility:
            self.strategies.append(VolatilityBreakoutStrategy())
        
        if use_range:
            self.strategies.append(RangeBreakoutStrategy())
        
        if use_support_resistance:
            self.strategies.append(SupportResistanceStrategy(bounce_mode=False))  # Use breakout mode
        
        logger.info(f"Initialized BreakoutEnsemble with {len(self.strategies)} strategies")
    
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
            logger.warning("BreakoutEnsemble has no strategies")
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
        
        logger.info(f"Generated ensemble signals from {len(self.strategies)} breakout strategies")
        return signals


if __name__ == "__main__":
    # Example usage
    print("Breakout and Range Trading Strategies module ready for use")

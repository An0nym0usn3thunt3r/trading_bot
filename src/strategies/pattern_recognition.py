"""
Pattern Recognition Strategies for Nasdaq-100 E-mini futures trading bot.
This module implements various pattern recognition trading strategies.
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


class CandlestickPatternStrategy(Strategy):
    """
    Candlestick Pattern Recognition strategy.
    
    This strategy identifies common candlestick patterns and generates signals based on them.
    """
    
    def __init__(self, confirmation_window=3, use_volume_filter=True, trend_filter=True, trend_window=50):
        """
        Initialize the Candlestick Pattern strategy.
        
        Parameters:
        -----------
        confirmation_window : int, optional
            Window for confirming the pattern's effectiveness.
        use_volume_filter : bool, optional
            Whether to use volume as a confirmation filter.
        trend_filter : bool, optional
            Whether to use trend direction as a filter.
        trend_window : int, optional
            Window for determining the trend direction.
        """
        parameters = {
            'confirmation_window': confirmation_window,
            'use_volume_filter': use_volume_filter,
            'trend_filter': trend_filter,
            'trend_window': trend_window
        }
        super().__init__(name="CandlestickPattern", parameters=parameters)
    
    def _is_doji(self, df, i, threshold=0.1):
        """Check if the candle is a doji (open and close are very close)."""
        body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])
        candle_range = df['high'].iloc[i] - df['low'].iloc[i]
        return body_size <= (candle_range * threshold)
    
    def _is_hammer(self, df, i, threshold=0.3):
        """Check if the candle is a hammer (small body, long lower shadow, small upper shadow)."""
        body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])
        candle_range = df['high'].iloc[i] - df['low'].iloc[i]
        
        if body_size <= (candle_range * 0.3):  # Small body
            body_low = min(df['open'].iloc[i], df['close'].iloc[i])
            lower_shadow = body_low - df['low'].iloc[i]
            
            body_high = max(df['open'].iloc[i], df['close'].iloc[i])
            upper_shadow = df['high'].iloc[i] - body_high
            
            # Long lower shadow, small upper shadow
            return (lower_shadow >= (candle_range * 0.6)) and (upper_shadow <= (candle_range * 0.1))
        
        return False
    
    def _is_shooting_star(self, df, i, threshold=0.3):
        """Check if the candle is a shooting star (small body, long upper shadow, small lower shadow)."""
        body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])
        candle_range = df['high'].iloc[i] - df['low'].iloc[i]
        
        if body_size <= (candle_range * 0.3):  # Small body
            body_low = min(df['open'].iloc[i], df['close'].iloc[i])
            lower_shadow = body_low - df['low'].iloc[i]
            
            body_high = max(df['open'].iloc[i], df['close'].iloc[i])
            upper_shadow = df['high'].iloc[i] - body_high
            
            # Long upper shadow, small lower shadow
            return (upper_shadow >= (candle_range * 0.6)) and (lower_shadow <= (candle_range * 0.1))
        
        return False
    
    def _is_engulfing_bullish(self, df, i):
        """Check if the pattern is a bullish engulfing (current candle engulfs previous bearish candle)."""
        if i == 0:
            return False
        
        # Previous candle is bearish (close < open)
        prev_bearish = df['close'].iloc[i-1] < df['open'].iloc[i-1]
        
        # Current candle is bullish (close > open)
        curr_bullish = df['close'].iloc[i] > df['open'].iloc[i]
        
        # Current candle body engulfs previous candle body
        engulfing = (df['open'].iloc[i] <= df['close'].iloc[i-1]) and (df['close'].iloc[i] >= df['open'].iloc[i-1])
        
        return prev_bearish and curr_bullish and engulfing
    
    def _is_engulfing_bearish(self, df, i):
        """Check if the pattern is a bearish engulfing (current candle engulfs previous bullish candle)."""
        if i == 0:
            return False
        
        # Previous candle is bullish (close > open)
        prev_bullish = df['close'].iloc[i-1] > df['open'].iloc[i-1]
        
        # Current candle is bearish (close < open)
        curr_bearish = df['close'].iloc[i] < df['open'].iloc[i]
        
        # Current candle body engulfs previous candle body
        engulfing = (df['open'].iloc[i] >= df['close'].iloc[i-1]) and (df['close'].iloc[i] <= df['open'].iloc[i-1])
        
        return prev_bullish and curr_bearish and engulfing
    
    def _is_morning_star(self, df, i):
        """Check if the pattern is a morning star (reversal pattern at bottom of downtrend)."""
        if i < 2:
            return False
        
        # First candle is bearish with large body
        first_bearish = df['close'].iloc[i-2] < df['open'].iloc[i-2]
        first_large = abs(df['close'].iloc[i-2] - df['open'].iloc[i-2]) > (df['high'].iloc[i-2] - df['low'].iloc[i-2]) * 0.6
        
        # Second candle is small (doji or small body)
        second_small = abs(df['close'].iloc[i-1] - df['open'].iloc[i-1]) < (df['high'].iloc[i-1] - df['low'].iloc[i-1]) * 0.3
        
        # Third candle is bullish with large body
        third_bullish = df['close'].iloc[i] > df['open'].iloc[i]
        third_large = abs(df['close'].iloc[i] - df['open'].iloc[i]) > (df['high'].iloc[i] - df['low'].iloc[i]) * 0.6
        
        # Gap down between first and second, gap up between second and third
        gap_down = max(df['open'].iloc[i-1], df['close'].iloc[i-1]) < df['close'].iloc[i-2]
        gap_up = min(df['open'].iloc[i], df['close'].iloc[i]) > df['close'].iloc[i-1]
        
        return first_bearish and first_large and second_small and third_bullish and third_large and gap_down and gap_up
    
    def _is_evening_star(self, df, i):
        """Check if the pattern is an evening star (reversal pattern at top of uptrend)."""
        if i < 2:
            return False
        
        # First candle is bullish with large body
        first_bullish = df['close'].iloc[i-2] > df['open'].iloc[i-2]
        first_large = abs(df['close'].iloc[i-2] - df['open'].iloc[i-2]) > (df['high'].iloc[i-2] - df['low'].iloc[i-2]) * 0.6
        
        # Second candle is small (doji or small body)
        second_small = abs(df['close'].iloc[i-1] - df['open'].iloc[i-1]) < (df['high'].iloc[i-1] - df['low'].iloc[i-1]) * 0.3
        
        # Third candle is bearish with large body
        third_bearish = df['close'].iloc[i] < df['open'].iloc[i]
        third_large = abs(df['close'].iloc[i] - df['open'].iloc[i]) > (df['high'].iloc[i] - df['low'].iloc[i]) * 0.6
        
        # Gap up between first and second, gap down between second and third
        gap_up = min(df['open'].iloc[i-1], df['close'].iloc[i-1]) > df['close'].iloc[i-2]
        gap_down = max(df['open'].iloc[i], df['close'].iloc[i]) < df['close'].iloc[i-1]
        
        return first_bullish and first_large and second_small and third_bearish and third_large and gap_up and gap_down
    
    def generate_signals(self, data):
        """
        Generate trading signals based on candlestick patterns.
        
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
        confirmation_window = self.parameters['confirmation_window']
        use_volume_filter = self.parameters['use_volume_filter']
        trend_filter = self.parameters['trend_filter']
        trend_window = self.parameters['trend_window']
        
        # Make a copy of the data
        df = data.copy()
        
        # Calculate trend if needed
        if trend_filter:
            df['trend_ma'] = df['close'].rolling(window=trend_window).mean()
            df['trend'] = np.where(df['close'] > df['trend_ma'], 1, -1)
        
        # Initialize pattern columns
        df['bullish_pattern'] = False
        df['bearish_pattern'] = False
        
        # Identify patterns
        for i in range(max(3, confirmation_window), len(df)):
            # Bullish patterns
            if self._is_hammer(df, i) or self._is_engulfing_bullish(df, i) or self._is_morning_star(df, i):
                df['bullish_pattern'].iloc[i] = True
            
            # Bearish patterns
            if self._is_shooting_star(df, i) or self._is_engulfing_bearish(df, i) or self._is_evening_star(df, i):
                df['bearish_pattern'].iloc[i] = True
        
        # Apply volume filter if enabled
        if use_volume_filter:
            avg_volume = df['volume'].rolling(window=20).mean()
            high_volume = df['volume'] > avg_volume * 1.5
            
            df['bullish_pattern'] = df['bullish_pattern'] & high_volume
            df['bearish_pattern'] = df['bearish_pattern'] & high_volume
        
        # Apply trend filter if enabled
        if trend_filter:
            df['bullish_pattern'] = df['bullish_pattern'] & (df['trend'] == -1)  # Bullish pattern in downtrend (reversal)
            df['bearish_pattern'] = df['bearish_pattern'] & (df['trend'] == 1)   # Bearish pattern in uptrend (reversal)
        
        # Initialize signal columns
        signals = pd.DataFrame(index=df.index)
        signals['enter_long'] = False
        signals['enter_short'] = False
        signals['exit_long'] = False
        signals['exit_short'] = False
        
        # Generate entry signals
        signals['enter_long'] = df['bullish_pattern']
        signals['enter_short'] = df['bearish_pattern']
        
        # Generate exit signals based on opposite patterns
        signals['exit_long'] = df['bearish_pattern']
        signals['exit_short'] = df['bullish_pattern']
        
        # Add time-based exit
        if confirmation_window > 0:
            # Create rolling window for entry signals
            for i in range(confirmation_window, len(signals)):
                # Exit long if we've been in position for confirmation_window bars
                if signals['enter_long'].iloc[i - confirmation_window]:
                    signals['exit_long'].iloc[i] = True
                
                # Exit short if we've been in position for confirmation_window bars
                if signals['enter_short'].iloc[i - confirmation_window]:
                    signals['exit_short'].iloc[i] = True
        
        logger.info(f"Generated Candlestick Pattern signals with confirmation_window={confirmation_window}")
        return signals


class ChartPatternStrategy(Strategy):
    """
    Chart Pattern Recognition strategy.
    
    This strategy identifies common chart patterns like head and shoulders, double tops/bottoms, etc.
    """
    
    def __init__(self, window=50, threshold=0.03, confirmation_bars=3, volume_filter=True):
        """
        Initialize the Chart Pattern strategy.
        
        Parameters:
        -----------
        window : int, optional
            Window period for identifying patterns.
        threshold : float, optional
            Threshold for pattern similarity (lower = stricter matching).
        confirmation_bars : int, optional
            Number of bars to confirm a pattern breakout.
        volume_filter : bool, optional
            Whether to use volume as a confirmation filter.
        """
        parameters = {
            'window': window,
            'threshold': threshold,
            'confirmation_bars': confirmation_bars,
            'volume_filter': volume_filter
        }
        super().__init__(name="ChartPattern", parameters=parameters)
    
    def _find_peaks_and_troughs(self, series, window=5):
        """Find peaks and troughs in a price series."""
        peaks = []
        troughs = []
        
        for i in range(window, len(series) - window):
            if all(series.iloc[i] > series.iloc[i-j] for j in range(1, window+1)) and \
               all(series.iloc[i] > series.iloc[i+j] for j in range(1, window+1)):
                peaks.append(i)
            
            if all(series.iloc[i] < series.iloc[i-j] for j in range(1, window+1)) and \
               all(series.iloc[i] < series.iloc[i+j] for j in range(1, window+1)):
                troughs.append(i)
        
        return peaks, troughs
    
    def _is_head_and_shoulders(self, df, peaks, i, threshold):
        """Check if there's a head and shoulders pattern ending at index i."""
        if len(peaks) < 3 or i < peaks[-1] + 10:
            return False
        
        # Get the last three peaks
        p1, p2, p3 = peaks[-3], peaks[-2], peaks[-1]
        
        # Check if middle peak (head) is higher than the other two (shoulders)
        if not (df['high'].iloc[p2] > df['high'].iloc[p1] and df['high'].iloc[p2] > df['high'].iloc[p3]):
            return False
        
        # Check if the shoulders are at similar heights
        shoulder_diff = abs(df['high'].iloc[p1] - df['high'].iloc[p3]) / df['high'].iloc[p1]
        if shoulder_diff > threshold:
            return False
        
        # Check if there's a neckline (connecting the troughs between peaks)
        t1 = df['low'].iloc[p1:p2].idxmin()
        t2 = df['low'].iloc[p2:p3].idxmin()
        
        # Neckline should be relatively flat
        neckline_diff = abs(df['low'].iloc[t1] - df['low'].iloc[t2]) / df['low'].iloc[t1]
        if neckline_diff > threshold:
            return False
        
        # Check if price has broken below the neckline
        neckline = min(df['low'].iloc[t1], df['low'].iloc[t2])
        return df['close'].iloc[i] < neckline
    
    def _is_inverse_head_and_shoulders(self, df, troughs, i, threshold):
        """Check if there's an inverse head and shoulders pattern ending at index i."""
        if len(troughs) < 3 or i < troughs[-1] + 10:
            return False
        
        # Get the last three troughs
        t1, t2, t3 = troughs[-3], troughs[-2], troughs[-1]
        
        # Check if middle trough (head) is lower than the other two (shoulders)
        if not (df['low'].iloc[t2] < df['low'].iloc[t1] and df['low'].iloc[t2] < df['low'].iloc[t3]):
            return False
        
        # Check if the shoulders are at similar heights
        shoulder_diff = abs(df['low'].iloc[t1] - df['low'].iloc[t3]) / df['low'].iloc[t1]
        if shoulder_diff > threshold:
            return False
        
        # Check if there's a neckline (connecting the peaks between troughs)
        p1 = df['high'].iloc[t1:t2].idxmax()
        p2 = df['high'].iloc[t2:t3].idxmax()
        
        # Neckline should be relatively flat
        neckline_diff = abs(df['high'].iloc[p1] - df['high'].iloc[p2]) / df['high'].iloc[p1]
        if neckline_diff > threshold:
            return False
        
        # Check if price has broken above the neckline
        neckline = max(df['high'].iloc[p1], df['high'].iloc[p2])
        return df['close'].iloc[i] > neckline
    
    def _is_double_top(self, df, peaks, i, threshold):
        """Check if there's a double top pattern ending at index i."""
        if len(peaks) < 2 or i < peaks[-1] + 10:
            return False
        
        # Get the last two peaks
        p1, p2 = peaks[-2], peaks[-1]
        
        # Peaks should be at similar heights
        peak_diff = abs(df['high'].iloc[p1] - df['high'].iloc[p2]) / df['high'].iloc[p1]
        if peak_diff > threshold:
            return False
        
        # There should be a significant trough between the peaks
        trough = df['low'].iloc[p1:p2].idxmin()
        
        # Check if price has broken below the trough
        return df['close'].iloc[i] < df['low'].iloc[trough]
    
    def _is_double_bottom(self, df, troughs, i, threshold):
        """Check if there's a double bottom pattern ending at index i."""
        if len(troughs) < 2 or i < troughs[-1] + 10:
            return False
        
        # Get the last two troughs
        t1, t2 = troughs[-2], troughs[-1]
        
        # Troughs should be at similar heights
        trough_diff = abs(df['low'].iloc[t1] - df['low'].iloc[t2]) / df['low'].iloc[t1]
        if trough_diff > threshold:
            return False
        
        # There should be a significant peak between the troughs
        peak = df['high'].iloc[t1:t2].idxmax()
        
        # Check if price has broken above the peak
        return df['close'].iloc[i] > df['high'].iloc[peak]
    
    def generate_signals(self, data):
        """
        Generate trading signals based on chart patterns.
        
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
        threshold = self.parameters['threshold']
        confirmation_bars = self.parameters['confirmation_bars']
        volume_filter = self.parameters['volume_filter']
        
        # Make a copy of the data
        df = data.copy()
        
        # Find peaks and troughs
        peaks, troughs = self._find_peaks_and_troughs(df['close'], window=5)
        
        # Initialize pattern columns
        df['head_and_shoulders'] = False
        df['inverse_head_and_shoulders'] = False
        df['double_top'] = False
        df['double_bottom'] = False
        
        # Identify patterns
        for i in range(window, len(df)):
            # Head and Shoulders (bearish)
            if self._is_head_and_shoulders(df, peaks, i, threshold):
                df['head_and_shoulders'].iloc[i] = True
            
            # Inverse Head and Shoulders (bullish)
            if self._is_inverse_head_and_shoulders(df, troughs, i, threshold):
                df['inverse_head_and_shoulders'].iloc[i] = True
            
            # Double Top (bearish)
            if self._is_double_top(df, peaks, i, threshold):
                df['double_top'].iloc[i] = True
            
            # Double Bottom (bullish)
            if self._is_double_bottom(df, troughs, i, threshold):
                df['double_bottom'].iloc[i] = True
        
        # Apply volume filter if enabled
        if volume_filter:
            avg_volume = df['volume'].rolling(window=20).mean()
            high_volume = df['volume'] > avg_volume * 1.5
            
            df['head_and_shoulders'] = df['head_and_shoulders'] & high_volume
            df['inverse_head_and_shoulders'] = df['inverse_head_and_shoulders'] & high_volume
            df['double_top'] = df['double_top'] & high_volume
            df['double_bottom'] = df['double_bottom'] & high_volume
        
        # Initialize signal columns
        signals = pd.DataFrame(index=df.index)
        signals['enter_long'] = False
        signals['enter_short'] = False
        signals['exit_long'] = False
        signals['exit_short'] = False
        
        # Generate entry signals
        signals['enter_long'] = df['inverse_head_and_shoulders'] | df['double_bottom']
        signals['enter_short'] = df['head_and_shoulders'] | df['double_top']
        
        # Generate exit signals based on opposite patterns
        signals['exit_long'] = df['head_and_shoulders'] | df['double_top']
        signals['exit_short'] = df['inverse_head_and_shoulders'] | df['double_bottom']
        
        # Apply confirmation filter
        if confirmation_bars > 0:
            # Confirm entry signals
            for i in range(confirmation_bars, len(signals)):
                # Only keep entry signals that persist for confirmation_bars
                if signals['enter_long'].iloc[i]:
                    signals['enter_long'].iloc[i] = all(signals['enter_long'].iloc[i-j] for j in range(1, confirmation_bars))
                
                if signals['enter_short'].iloc[i]:
                    signals['enter_short'].iloc[i] = all(signals['enter_short'].iloc[i-j] for j in range(1, confirmation_bars))
        
        logger.info(f"Generated Chart Pattern signals with window={window}, threshold={threshold}")
        return signals


class HarmonicPatternStrategy(Strategy):
    """
    Harmonic Pattern Recognition strategy.
    
    This strategy identifies harmonic price patterns like Gartley, Butterfly, Bat, etc.
    """
    
    def __init__(self, pattern_types=None, tolerance=0.1, confirmation_bars=2):
        """
        Initialize the Harmonic Pattern strategy.
        
        Parameters:
        -----------
        pattern_types : list, optional
            List of pattern types to identify. If None, uses all patterns.
        tolerance : float, optional
            Tolerance for Fibonacci ratio matching.
        confirmation_bars : int, optional
            Number of bars to confirm a pattern.
        """
        if pattern_types is None:
            pattern_types = ['gartley', 'butterfly', 'bat', 'crab']
        
        parameters = {
            'pattern_types': pattern_types,
            'tolerance': tolerance,
            'confirmation_bars': confirmation_bars
        }
        super().__init__(name="HarmonicPattern", parameters=parameters)
        
        # Define Fibonacci ratios for each pattern
        self.pattern_ratios = {
            'gartley': {
                'bullish': {'XA': 1.0, 'AB': 0.618, 'BC': 0.382, 'CD': 1.272},
                'bearish': {'XA': 1.0, 'AB': 0.618, 'BC': 0.382, 'CD': 1.272}
            },
            'butterfly': {
                'bullish': {'XA': 1.0, 'AB': 0.786, 'BC': 0.382, 'CD': 1.618},
                'bearish': {'XA': 1.0, 'AB': 0.786, 'BC': 0.382, 'CD': 1.618}
            },
            'bat': {
                'bullish': {'XA': 1.0, 'AB': 0.382, 'BC': 0.382, 'CD': 1.618},
                'bearish': {'XA': 1.0, 'AB': 0.382, 'BC': 0.382, 'CD': 1.618}
            },
            'crab': {
                'bullish': {'XA': 1.0, 'AB': 0.382, 'BC': 0.618, 'CD': 3.618},
                'bearish': {'XA': 1.0, 'AB': 0.382, 'BC': 0.618, 'CD': 3.618}
            }
        }
    
    def _find_swings(self, df, window=5):
        """Find swing highs and lows in the price series."""
        swing_highs = []
        swing_lows = []
        
        for i in range(window, len(df) - window):
            # Swing high
            if all(df['high'].iloc[i] > df['high'].iloc[i-j] for j in range(1, window+1)) and \
               all(df['high'].iloc[i] > df['high'].iloc[i+j] for j in range(1, window+1)):
                swing_highs.append(i)
            
            # Swing low
            if all(df['low'].iloc[i] < df['low'].iloc[i-j] for j in range(1, window+1)) and \
               all(df['low'].iloc[i] < df['low'].iloc[i+j] for j in range(1, window+1)):
                swing_lows.append(i)
        
        return swing_highs, swing_lows
    
    def _is_fibonacci_match(self, actual, target, tolerance):
        """Check if the actual ratio is within tolerance of the target Fibonacci ratio."""
        return abs(actual - target) <= tolerance
    
    def _identify_harmonic_patterns(self, df, swings, pattern_type, direction, tolerance):
        """Identify harmonic patterns in the price series."""
        patterns = []
        
        # Get the ratios for this pattern type and direction
        ratios = self.pattern_ratios[pattern_type][direction]
        
        # Need at least 5 swing points (X, A, B, C, D)
        if len(swings) < 5:
            return patterns
        
        # Check each possible pattern
        for i in range(len(swings) - 4):
            X, A, B, C, D = swings[i:i+5]
            
            # Get price values at swing points
            if direction == 'bullish':
                X_price = df['low'].iloc[X]
                A_price = df['high'].iloc[A]
                B_price = df['low'].iloc[B]
                C_price = df['high'].iloc[C]
                D_price = df['low'].iloc[D]
            else:  # bearish
                X_price = df['high'].iloc[X]
                A_price = df['low'].iloc[A]
                B_price = df['high'].iloc[B]
                C_price = df['low'].iloc[C]
                D_price = df['high'].iloc[D]
            
            # Calculate leg lengths
            XA = abs(A_price - X_price)
            AB = abs(B_price - A_price)
            BC = abs(C_price - B_price)
            CD = abs(D_price - C_price)
            
            # Calculate ratios
            AB_XA_ratio = AB / XA
            BC_AB_ratio = BC / AB
            CD_BC_ratio = CD / BC
            
            # Check if ratios match the pattern
            if self._is_fibonacci_match(AB_XA_ratio, ratios['AB'], tolerance) and \
               self._is_fibonacci_match(BC_AB_ratio, ratios['BC'], tolerance) and \
               self._is_fibonacci_match(CD_BC_ratio, ratios['CD'], tolerance):
                patterns.append((X, A, B, C, D))
        
        return patterns
    
    def generate_signals(self, data):
        """
        Generate trading signals based on harmonic patterns.
        
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
        pattern_types = self.parameters['pattern_types']
        tolerance = self.parameters['tolerance']
        confirmation_bars = self.parameters['confirmation_bars']
        
        # Make a copy of the data
        df = data.copy()
        
        # Find swing highs and lows
        swing_highs, swing_lows = self._find_swings(df)
        
        # Initialize pattern columns
        df['bullish_harmonic'] = False
        df['bearish_harmonic'] = False
        
        # Identify patterns
        for pattern_type in pattern_types:
            # Bullish patterns (from swing lows)
            bullish_patterns = self._identify_harmonic_patterns(df, swing_lows, pattern_type, 'bullish', tolerance)
            
            # Bearish patterns (from swing highs)
            bearish_patterns = self._identify_harmonic_patterns(df, swing_highs, pattern_type, 'bearish', tolerance)
            
            # Mark pattern completion points
            for pattern in bullish_patterns:
                _, _, _, _, D = pattern
                df['bullish_harmonic'].iloc[D] = True
            
            for pattern in bearish_patterns:
                _, _, _, _, D = pattern
                df['bearish_harmonic'].iloc[D] = True
        
        # Initialize signal columns
        signals = pd.DataFrame(index=df.index)
        signals['enter_long'] = False
        signals['enter_short'] = False
        signals['exit_long'] = False
        signals['exit_short'] = False
        
        # Generate entry signals
        signals['enter_long'] = df['bullish_harmonic']
        signals['enter_short'] = df['bearish_harmonic']
        
        # Generate exit signals based on opposite patterns
        signals['exit_long'] = df['bearish_harmonic']
        signals['exit_short'] = df['bullish_harmonic']
        
        # Apply confirmation filter
        if confirmation_bars > 0:
            # Confirm entry signals
            for i in range(confirmation_bars, len(signals)):
                # Only keep entry signals that persist for confirmation_bars
                if signals['enter_long'].iloc[i]:
                    signals['enter_long'].iloc[i] = all(df['close'].iloc[i-j] > df['close'].iloc[i-j-1] for j in range(confirmation_bars))
                
                if signals['enter_short'].iloc[i]:
                    signals['enter_short'].iloc[i] = all(df['close'].iloc[i-j] < df['close'].iloc[i-j-1] for j in range(confirmation_bars))
        
        logger.info(f"Generated Harmonic Pattern signals with pattern_types={pattern_types}, tolerance={tolerance}")
        return signals


class PatternRecognitionEnsemble(Strategy):
    """
    Ensemble of multiple pattern recognition strategies.
    """
    
    def __init__(self, use_candlestick=True, use_chart=True, use_harmonic=True):
        """
        Initialize the pattern recognition ensemble strategy.
        
        Parameters:
        -----------
        use_candlestick : bool, optional
            Whether to include Candlestick Pattern strategy.
        use_chart : bool, optional
            Whether to include Chart Pattern strategy.
        use_harmonic : bool, optional
            Whether to include Harmonic Pattern strategy.
        """
        parameters = {
            'use_candlestick': use_candlestick,
            'use_chart': use_chart,
            'use_harmonic': use_harmonic
        }
        super().__init__(name="PatternRecognitionEnsemble", parameters=parameters)
        
        # Create individual strategies
        self.strategies = []
        
        if use_candlestick:
            self.strategies.append(CandlestickPatternStrategy())
        
        if use_chart:
            self.strategies.append(ChartPatternStrategy())
        
        if use_harmonic:
            self.strategies.append(HarmonicPatternStrategy())
        
        logger.info(f"Initialized PatternRecognitionEnsemble with {len(self.strategies)} strategies")
    
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
            logger.warning("PatternRecognitionEnsemble has no strategies")
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
        
        logger.info(f"Generated ensemble signals from {len(self.strategies)} pattern recognition strategies")
        return signals


if __name__ == "__main__":
    # Example usage
    print("Pattern Recognition Strategies module ready for use")

"""
Market Microstructure Strategies for Nasdaq-100 E-mini futures trading bot.
This module implements various market microstructure-based trading strategies.
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


class VolumeProfileStrategy(Strategy):
    """
    Volume Profile trading strategy.
    
    This strategy identifies high volume nodes and value areas to generate trading signals.
    """
    
    def __init__(self, profile_period=20, value_area_pct=0.70, num_bins=20, volume_threshold=1.5):
        """
        Initialize the Volume Profile strategy.
        
        Parameters:
        -----------
        profile_period : int, optional
            Number of bars to include in the volume profile.
        value_area_pct : float, optional
            Percentage of volume to include in the value area (0.0-1.0).
        num_bins : int, optional
            Number of price bins to use for the volume profile.
        volume_threshold : float, optional
            Volume threshold multiplier for significant volume.
        """
        parameters = {
            'profile_period': profile_period,
            'value_area_pct': value_area_pct,
            'num_bins': num_bins,
            'volume_threshold': volume_threshold
        }
        super().__init__(name="VolumeProfile", parameters=parameters)
    
    def _calculate_volume_profile(self, df, start_idx, end_idx, num_bins):
        """Calculate volume profile for a specific period."""
        # Extract the data for the period
        period_data = df.iloc[start_idx:end_idx+1]
        
        # Determine price range
        price_min = period_data['low'].min()
        price_max = period_data['high'].max()
        
        # Create price bins
        bin_size = (price_max - price_min) / num_bins
        bins = [price_min + i * bin_size for i in range(num_bins + 1)]
        
        # Initialize volume profile
        volume_profile = np.zeros(num_bins)
        
        # Distribute volume across price bins
        for i in range(len(period_data)):
            bar = period_data.iloc[i]
            bar_min = bar['low']
            bar_max = bar['high']
            bar_volume = bar['volume']
            
            # Determine which bins this bar spans
            min_bin = max(0, int((bar_min - price_min) / bin_size))
            max_bin = min(num_bins - 1, int((bar_max - price_min) / bin_size))
            
            # Distribute volume proportionally across bins
            if min_bin == max_bin:
                volume_profile[min_bin] += bar_volume
            else:
                num_bins_spanned = max_bin - min_bin + 1
                volume_per_bin = bar_volume / num_bins_spanned
                for b in range(min_bin, max_bin + 1):
                    volume_profile[b] += volume_per_bin
        
        # Calculate bin prices (center of each bin)
        bin_prices = [price_min + (i + 0.5) * bin_size for i in range(num_bins)]
        
        return volume_profile, bin_prices, bin_size
    
    def _find_value_area(self, volume_profile, bin_prices, value_area_pct):
        """Find the value area (price range containing value_area_pct of volume)."""
        # Sort bins by volume (descending)
        sorted_indices = np.argsort(-volume_profile)
        
        # Calculate total volume
        total_volume = np.sum(volume_profile)
        
        # Calculate value area volume
        value_area_volume = total_volume * value_area_pct
        
        # Find bins in value area
        cumulative_volume = 0
        value_area_bins = []
        
        for idx in sorted_indices:
            value_area_bins.append(idx)
            cumulative_volume += volume_profile[idx]
            
            if cumulative_volume >= value_area_volume:
                break
        
        # Sort bins by price
        value_area_bins.sort()
        
        # Find continuous ranges
        ranges = []
        current_range = [value_area_bins[0]]
        
        for i in range(1, len(value_area_bins)):
            if value_area_bins[i] == value_area_bins[i-1] + 1:
                current_range.append(value_area_bins[i])
            else:
                ranges.append(current_range)
                current_range = [value_area_bins[i]]
        
        ranges.append(current_range)
        
        # Find the largest continuous range
        largest_range = max(ranges, key=len)
        
        # Calculate value area high and low
        value_area_low = bin_prices[largest_range[0]]
        value_area_high = bin_prices[largest_range[-1]]
        
        # Find point of control (highest volume price)
        poc_idx = np.argmax(volume_profile)
        poc_price = bin_prices[poc_idx]
        
        return value_area_low, value_area_high, poc_price
    
    def generate_signals(self, data):
        """
        Generate trading signals based on volume profile analysis.
        
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
        profile_period = self.parameters['profile_period']
        value_area_pct = self.parameters['value_area_pct']
        num_bins = self.parameters['num_bins']
        volume_threshold = self.parameters['volume_threshold']
        
        # Make a copy of the data
        df = data.copy()
        
        # Initialize columns for value area and POC
        df['value_area_low'] = np.nan
        df['value_area_high'] = np.nan
        df['poc'] = np.nan
        
        # Calculate volume profile for each period
        for i in range(profile_period, len(df)):
            start_idx = i - profile_period
            end_idx = i - 1
            
            volume_profile, bin_prices, bin_size = self._calculate_volume_profile(
                df, start_idx, end_idx, num_bins)
            
            value_area_low, value_area_high, poc_price = self._find_value_area(
                volume_profile, bin_prices, value_area_pct)
            
            df['value_area_low'].iloc[i] = value_area_low
            df['value_area_high'].iloc[i] = value_area_high
            df['poc'].iloc[i] = poc_price
        
        # Calculate average volume
        df['avg_volume'] = df['volume'].rolling(window=20).mean()
        
        # Initialize signal columns
        signals = pd.DataFrame(index=df.index)
        signals['enter_long'] = False
        signals['enter_short'] = False
        signals['exit_long'] = False
        signals['exit_short'] = False
        
        # Generate signals
        for i in range(profile_period + 1, len(df)):
            # High volume condition
            high_volume = df['volume'].iloc[i-1] > df['avg_volume'].iloc[i-1] * volume_threshold
            
            # Value area rejection (long)
            if df['low'].iloc[i-1] <= df['value_area_low'].iloc[i-1] and df['close'].iloc[i-1] > df['value_area_low'].iloc[i-1]:
                if high_volume:
                    signals['enter_long'].iloc[i] = True
            
            # Value area rejection (short)
            if df['high'].iloc[i-1] >= df['value_area_high'].iloc[i-1] and df['close'].iloc[i-1] < df['value_area_high'].iloc[i-1]:
                if high_volume:
                    signals['enter_short'].iloc[i] = True
            
            # Exit signals
            # Exit long when price reaches POC or value area high
            if df['close'].iloc[i-1] >= df['poc'].iloc[i-1]:
                signals['exit_long'].iloc[i] = True
            
            # Exit short when price reaches POC or value area low
            if df['close'].iloc[i-1] <= df['poc'].iloc[i-1]:
                signals['exit_short'].iloc[i] = True
        
        logger.info(f"Generated Volume Profile signals with period={profile_period}, value_area_pct={value_area_pct}")
        return signals


class OrderFlowStrategy(Strategy):
    """
    Order Flow Analysis strategy.
    
    This strategy analyzes buying and selling pressure based on volume and price movement.
    """
    
    def __init__(self, window=10, delta_threshold=0.6, volume_threshold=1.2, trend_filter=True):
        """
        Initialize the Order Flow Analysis strategy.
        
        Parameters:
        -----------
        window : int, optional
            Window period for calculating delta and volume metrics.
        delta_threshold : float, optional
            Threshold for delta (buying/selling pressure) significance (0.0-1.0).
        volume_threshold : float, optional
            Volume threshold multiplier for significant volume.
        trend_filter : bool, optional
            Whether to use a trend filter.
        """
        parameters = {
            'window': window,
            'delta_threshold': delta_threshold,
            'volume_threshold': volume_threshold,
            'trend_filter': trend_filter
        }
        super().__init__(name="OrderFlow", parameters=parameters)
    
    def generate_signals(self, data):
        """
        Generate trading signals based on order flow analysis.
        
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
        delta_threshold = self.parameters['delta_threshold']
        volume_threshold = self.parameters['volume_threshold']
        trend_filter = self.parameters['trend_filter']
        
        # Make a copy of the data
        df = data.copy()
        
        # Calculate delta (buying/selling pressure)
        # Delta = (close - open) / (high - low)
        df['delta'] = (df['close'] - df['open']) / (df['high'] - df['low'])
        
        # Handle division by zero
        df['delta'] = df['delta'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Calculate cumulative delta
        df['cum_delta'] = df['delta'].rolling(window=window).sum()
        
        # Calculate average volume
        df['avg_volume'] = df['volume'].rolling(window=20).mean()
        
        # Calculate trend if needed
        if trend_filter:
            df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['trend'] = np.where(df['ema_20'] > df['ema_50'], 1, -1)
        
        # Initialize signal columns
        signals = pd.DataFrame(index=df.index)
        signals['enter_long'] = False
        signals['enter_short'] = False
        signals['exit_long'] = False
        signals['exit_short'] = False
        
        # Generate signals
        for i in range(window + 1, len(df)):
            # High volume condition
            high_volume = df['volume'].iloc[i-1] > df['avg_volume'].iloc[i-1] * volume_threshold
            
            # Strong buying pressure
            strong_buying = df['cum_delta'].iloc[i-1] > delta_threshold
            
            # Strong selling pressure
            strong_selling = df['cum_delta'].iloc[i-1] < -delta_threshold
            
            # Apply trend filter if enabled
            if trend_filter:
                # Long entry: strong buying pressure with high volume in uptrend
                if strong_buying and high_volume and df['trend'].iloc[i-1] == 1:
                    signals['enter_long'].iloc[i] = True
                
                # Short entry: strong selling pressure with high volume in downtrend
                if strong_selling and high_volume and df['trend'].iloc[i-1] == -1:
                    signals['enter_short'].iloc[i] = True
            else:
                # Long entry: strong buying pressure with high volume
                if strong_buying and high_volume:
                    signals['enter_long'].iloc[i] = True
                
                # Short entry: strong selling pressure with high volume
                if strong_selling and high_volume:
                    signals['enter_short'].iloc[i] = True
            
            # Exit signals
            # Exit long when delta turns negative
            if df['delta'].iloc[i-1] < 0 and df['delta'].iloc[i-2] >= 0:
                signals['exit_long'].iloc[i] = True
            
            # Exit short when delta turns positive
            if df['delta'].iloc[i-1] > 0 and df['delta'].iloc[i-2] <= 0:
                signals['exit_short'].iloc[i] = True
        
        logger.info(f"Generated Order Flow signals with window={window}, delta_threshold={delta_threshold}")
        return signals


class MarketDepthStrategy(Strategy):
    """
    Market Depth Analysis strategy.
    
    This strategy simulates market depth analysis using volume and price action.
    Note: Actual market depth data (Level II) would be used in production.
    """
    
    def __init__(self, imbalance_threshold=2.0, volume_threshold=1.5, persistence=3):
        """
        Initialize the Market Depth Analysis strategy.
        
        Parameters:
        -----------
        imbalance_threshold : float, optional
            Threshold for order book imbalance significance.
        volume_threshold : float, optional
            Volume threshold multiplier for significant volume.
        persistence : int, optional
            Number of bars required for persistent imbalance.
        """
        parameters = {
            'imbalance_threshold': imbalance_threshold,
            'volume_threshold': volume_threshold,
            'persistence': persistence
        }
        super().__init__(name="MarketDepth", parameters=parameters)
    
    def generate_signals(self, data):
        """
        Generate trading signals based on simulated market depth analysis.
        
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
        imbalance_threshold = self.parameters['imbalance_threshold']
        volume_threshold = self.parameters['volume_threshold']
        persistence = self.parameters['persistence']
        
        # Make a copy of the data
        df = data.copy()
        
        # Calculate average volume
        df['avg_volume'] = df['volume'].rolling(window=20).mean()
        
        # Simulate order book imbalance using price action and volume
        # In a real implementation, this would use actual market depth data
        
        # Calculate price movement
        df['price_change'] = df['close'] - df['open']
        
        # Calculate normalized price movement
        df['norm_price_change'] = df['price_change'] / (df['high'] - df['low'])
        
        # Simulate buy/sell imbalance
        # Positive values indicate more buying pressure, negative values indicate more selling pressure
        df['imbalance'] = df['norm_price_change'] * (df['volume'] / df['avg_volume'])
        
        # Initialize signal columns
        signals = pd.DataFrame(index=df.index)
        signals['enter_long'] = False
        signals['enter_short'] = False
        signals['exit_long'] = False
        signals['exit_short'] = False
        
        # Generate signals
        for i in range(20 + persistence, len(df)):
            # Check for persistent imbalance
            buy_imbalance = all(df['imbalance'].iloc[i-j] > imbalance_threshold for j in range(1, persistence+1))
            sell_imbalance = all(df['imbalance'].iloc[i-j] < -imbalance_threshold for j in range(1, persistence+1))
            
            # High volume condition
            high_volume = df['volume'].iloc[i-1] > df['avg_volume'].iloc[i-1] * volume_threshold
            
            # Long entry: persistent buy imbalance with high volume
            if buy_imbalance and high_volume:
                signals['enter_long'].iloc[i] = True
            
            # Short entry: persistent sell imbalance with high volume
            if sell_imbalance and high_volume:
                signals['enter_short'].iloc[i] = True
            
            # Exit signals
            # Exit long when imbalance turns negative
            if df['imbalance'].iloc[i-1] < 0 and df['imbalance'].iloc[i-2] >= 0:
                signals['exit_long'].iloc[i] = True
            
            # Exit short when imbalance turns positive
            if df['imbalance'].iloc[i-1] > 0 and df['imbalance'].iloc[i-2] <= 0:
                signals['exit_short'].iloc[i] = True
        
        logger.info(f"Generated Market Depth signals with imbalance_threshold={imbalance_threshold}, persistence={persistence}")
        return signals


class LiquidityStrategy(Strategy):
    """
    Liquidity Analysis strategy.
    
    This strategy identifies and trades around liquidity zones where stop orders may be clustered.
    """
    
    def __init__(self, window=20, threshold=0.5, volume_filter=True, confirmation_bars=2):
        """
        Initialize the Liquidity Analysis strategy.
        
        Parameters:
        -----------
        window : int, optional
            Window period for identifying liquidity zones.
        threshold : float, optional
            Threshold for identifying significant price levels.
        volume_filter : bool, optional
            Whether to use volume as a confirmation filter.
        confirmation_bars : int, optional
            Number of bars to confirm a liquidity sweep.
        """
        parameters = {
            'window': window,
            'threshold': threshold,
            'volume_filter': volume_filter,
            'confirmation_bars': confirmation_bars
        }
        super().__init__(name="Liquidity", parameters=parameters)
    
    def _find_liquidity_zones(self, df, window, threshold):
        """Find liquidity zones (swing highs/lows where stops might be clustered)."""
        liquidity_highs = []
        liquidity_lows = []
        
        for i in range(window, len(df) - window):
            # Check if this is a swing high
            if all(df['high'].iloc[i] > df['high'].iloc[i-j] for j in range(1, window+1)) and \
               all(df['high'].iloc[i] > df['high'].iloc[i+j] for j in range(1, window+1)):
                # Calculate significance
                avg_range = df['high'].iloc[i-window:i].mean() - df['low'].iloc[i-window:i].mean()
                significance = (df['high'].iloc[i] - df['close'].iloc[i-1]) / avg_range
                
                if significance > threshold:
                    liquidity_highs.append((i, df['high'].iloc[i]))
            
            # Check if this is a swing low
            if all(df['low'].iloc[i] < df['low'].iloc[i-j] for j in range(1, window+1)) and \
               all(df['low'].iloc[i] < df['low'].iloc[i+j] for j in range(1, window+1)):
                # Calculate significance
                avg_range = df['high'].iloc[i-window:i].mean() - df['low'].iloc[i-window:i].mean()
                significance = (df['close'].iloc[i-1] - df['low'].iloc[i]) / avg_range
                
                if significance > threshold:
                    liquidity_lows.append((i, df['low'].iloc[i]))
        
        return liquidity_highs, liquidity_lows
    
    def generate_signals(self, data):
        """
        Generate trading signals based on liquidity analysis.
        
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
        volume_filter = self.parameters['volume_filter']
        confirmation_bars = self.parameters['confirmation_bars']
        
        # Make a copy of the data
        df = data.copy()
        
        # Find liquidity zones
        liquidity_highs, liquidity_lows = self._find_liquidity_zones(df, window, threshold)
        
        # Calculate average volume if needed
        if volume_filter:
            df['avg_volume'] = df['volume'].rolling(window=20).mean()
        
        # Initialize signal columns
        signals = pd.DataFrame(index=df.index)
        signals['enter_long'] = False
        signals['enter_short'] = False
        signals['exit_long'] = False
        signals['exit_short'] = False
        
        # Generate signals
        for i in range(window * 2, len(df)):
            # Check for liquidity sweeps
            
            # Sweep of liquidity highs (short entry)
            for idx, level in liquidity_highs:
                if idx < i - window:  # Only consider older levels
                    # Price moved above the liquidity level then back below
                    if df['high'].iloc[i-1] > level and df['close'].iloc[i-1] < level:
                        # Volume confirmation if enabled
                        if not volume_filter or df['volume'].iloc[i-1] > df['avg_volume'].iloc[i-1]:
                            # Price confirmation
                            if confirmation_bars <= 1 or all(df['close'].iloc[i-j] < df['close'].iloc[i-j-1] for j in range(1, confirmation_bars)):
                                signals['enter_short'].iloc[i] = True
                                break
            
            # Sweep of liquidity lows (long entry)
            for idx, level in liquidity_lows:
                if idx < i - window:  # Only consider older levels
                    # Price moved below the liquidity level then back above
                    if df['low'].iloc[i-1] < level and df['close'].iloc[i-1] > level:
                        # Volume confirmation if enabled
                        if not volume_filter or df['volume'].iloc[i-1] > df['avg_volume'].iloc[i-1]:
                            # Price confirmation
                            if confirmation_bars <= 1 or all(df['close'].iloc[i-j] > df['close'].iloc[i-j-1] for j in range(1, confirmation_bars)):
                                signals['enter_long'].iloc[i] = True
                                break
            
            # Exit signals
            # Exit long when approaching a liquidity high
            for idx, level in liquidity_highs:
                if idx < i - window:  # Only consider older levels
                    if df['high'].iloc[i-1] > level * 0.99 and df['high'].iloc[i-1] < level:
                        signals['exit_long'].iloc[i] = True
                        break
            
            # Exit short when approaching a liquidity low
            for idx, level in liquidity_lows:
                if idx < i - window:  # Only consider older levels
                    if df['low'].iloc[i-1] < level * 1.01 and df['low'].iloc[i-1] > level:
                        signals['exit_short'].iloc[i] = True
                        break
        
        logger.info(f"Generated Liquidity signals with window={window}, threshold={threshold}")
        return signals


class MicrostructureEnsemble(Strategy):
    """
    Ensemble of multiple market microstructure strategies.
    """
    
    def __init__(self, use_volume_profile=True, use_order_flow=True, use_market_depth=True, use_liquidity=True):
        """
        Initialize the market microstructure ensemble strategy.
        
        Parameters:
        -----------
        use_volume_profile : bool, optional
            Whether to include Volume Profile strategy.
        use_order_flow : bool, optional
            Whether to include Order Flow strategy.
        use_market_depth : bool, optional
            Whether to include Market Depth strategy.
        use_liquidity : bool, optional
            Whether to include Liquidity strategy.
        """
        parameters = {
            'use_volume_profile': use_volume_profile,
            'use_order_flow': use_order_flow,
            'use_market_depth': use_market_depth,
            'use_liquidity': use_liquidity
        }
        super().__init__(name="MicrostructureEnsemble", parameters=parameters)
        
        # Create individual strategies
        self.strategies = []
        
        if use_volume_profile:
            self.strategies.append(VolumeProfileStrategy())
        
        if use_order_flow:
            self.strategies.append(OrderFlowStrategy())
        
        if use_market_depth:
            self.strategies.append(MarketDepthStrategy())
        
        if use_liquidity:
            self.strategies.append(LiquidityStrategy())
        
        logger.info(f"Initialized MicrostructureEnsemble with {len(self.strategies)} strategies")
    
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
            logger.warning("MicrostructureEnsemble has no strategies")
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
        
        logger.info(f"Generated ensemble signals from {len(self.strategies)} microstructure strategies")
        return signals


if __name__ == "__main__":
    # Example usage
    print("Market Microstructure Strategies module ready for use")

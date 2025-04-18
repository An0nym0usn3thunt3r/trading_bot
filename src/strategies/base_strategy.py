"""
Base strategy module for Nasdaq-100 E-mini futures trading bot.
This module provides the base classes and interfaces for all trading strategies.
"""

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    All strategy implementations must inherit from this class.
    """
    
    def __init__(self, name=None, parameters=None):
        """
        Initialize the strategy.
        
        Parameters:
        -----------
        name : str, optional
            Name of the strategy. If None, uses the class name.
        parameters : dict, optional
            Dictionary of strategy parameters.
        """
        self.name = name or self.__class__.__name__
        self.parameters = parameters or {}
        self._is_trained = False
        logger.info(f"Strategy '{self.name}' initialized with parameters: {self.parameters}")
    
    @abstractmethod
    def generate_signals(self, data):
        """
        Generate trading signals from data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical price data with OHLCV columns and any additional features.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with signal columns (enter_long, enter_short, exit_long, exit_short).
        """
        pass
    
    def train(self, data, **kwargs):
        """
        Train the strategy on historical data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical price data for training.
        **kwargs : dict
            Additional keyword arguments for training.
            
        Returns:
        --------
        self
            Returns self for method chaining.
        """
        logger.info(f"Training strategy '{self.name}'")
        self._train_implementation(data, **kwargs)
        self._is_trained = True
        return self
    
    def _train_implementation(self, data, **kwargs):
        """
        Implementation of the training logic.
        Override this method in subclasses that require training.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical price data for training.
        **kwargs : dict
            Additional keyword arguments for training.
        """
        # Default implementation does nothing
        logger.info(f"Strategy '{self.name}' does not require training")
    
    def is_trained(self):
        """
        Check if the strategy has been trained.
        
        Returns:
        --------
        bool
            True if the strategy has been trained, False otherwise.
        """
        return self._is_trained
    
    def set_parameters(self, parameters):
        """
        Set strategy parameters.
        
        Parameters:
        -----------
        parameters : dict
            Dictionary of strategy parameters.
            
        Returns:
        --------
        self
            Returns self for method chaining.
        """
        self.parameters.update(parameters)
        logger.info(f"Updated parameters for strategy '{self.name}': {parameters}")
        return self
    
    def get_parameters(self):
        """
        Get strategy parameters.
        
        Returns:
        --------
        dict
            Dictionary of strategy parameters.
        """
        return self.parameters.copy()
    
    def optimize(self, data, parameter_grid, metric='sharpe_ratio', **kwargs):
        """
        Optimize strategy parameters using grid search.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical price data for optimization.
        parameter_grid : dict
            Dictionary of parameter names and lists of values to try.
        metric : str, optional
            Metric to optimize for.
        **kwargs : dict
            Additional keyword arguments for optimization.
            
        Returns:
        --------
        dict
            Dictionary with best parameters and optimization results.
        """
        from itertools import product
        import copy
        
        logger.info(f"Optimizing strategy '{self.name}' for {metric}")
        
        # Generate all parameter combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        param_combinations = list(product(*param_values))
        
        best_score = float('-inf')
        best_params = None
        results = []
        
        # Try each parameter combination
        for i, combination in enumerate(param_combinations):
            params = dict(zip(param_names, combination))
            logger.info(f"Testing parameters {i+1}/{len(param_combinations)}: {params}")
            
            # Create a copy of the strategy with these parameters
            strategy_copy = copy.deepcopy(self)
            strategy_copy.set_parameters(params)
            
            # Train if needed
            if kwargs.get('train', True):
                strategy_copy.train(data)
            
            # Generate signals
            signals = strategy_copy.generate_signals(data)
            
            # Evaluate
            from data.backtest_engine import BacktestEngine
            engine = BacktestEngine()
            result = engine.run_backtest(data, strategy_copy)
            
            # Extract metric
            score = self._extract_metric(result, metric)
            
            # Record result
            result_entry = {
                'parameters': params,
                'score': score,
                'result': result
            }
            results.append(result_entry)
            
            # Update best if better
            if score > best_score:
                best_score = score
                best_params = params
        
        # Set the best parameters
        if best_params:
            self.set_parameters(best_params)
            logger.info(f"Best parameters found: {best_params} with {metric}={best_score}")
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'results': results
        }
    
    def _extract_metric(self, backtest_result, metric):
        """
        Extract a specific metric from backtest results.
        
        Parameters:
        -----------
        backtest_result : dict
            Dictionary of backtest results.
        metric : str
            Name of the metric to extract.
            
        Returns:
        --------
        float
            Value of the metric.
        """
        if metric == 'total_return':
            return backtest_result['total_return']
        elif metric == 'sharpe_ratio':
            return backtest_result['stats'].get('sharpe_ratio', 0)
        elif metric == 'win_rate':
            return backtest_result['stats'].get('win_rate', 0)
        elif metric == 'profit_factor':
            return backtest_result['stats'].get('profit_factor', 0)
        elif metric == 'max_drawdown':
            # Negate so that smaller drawdowns are better
            return -backtest_result['stats']['max_drawdown_pct']
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def __str__(self):
        """String representation of the strategy."""
        return f"{self.name} Strategy with parameters: {self.parameters}"


class StrategyEnsemble(Strategy):
    """
    Ensemble of multiple strategies with weighting.
    """
    
    def __init__(self, strategies=None, weights=None, name="StrategyEnsemble"):
        """
        Initialize the strategy ensemble.
        
        Parameters:
        -----------
        strategies : list, optional
            List of Strategy objects.
        weights : list, optional
            List of weights for each strategy. If None, equal weights are used.
        name : str, optional
            Name of the ensemble.
        """
        super().__init__(name=name)
        self.strategies = strategies or []
        
        # Set equal weights if not provided
        if weights is None and strategies:
            weights = [1.0 / len(strategies)] * len(strategies)
        
        self.weights = weights or []
        
        # Validate weights
        if self.strategies and self.weights:
            if len(self.strategies) != len(self.weights):
                raise ValueError("Number of strategies must match number of weights")
            
            if abs(sum(self.weights) - 1.0) > 1e-10:
                logger.warning("Weights do not sum to 1, normalizing")
                total = sum(self.weights)
                self.weights = [w / total for w in self.weights]
        
        logger.info(f"StrategyEnsemble '{name}' initialized with {len(self.strategies)} strategies")
    
    def add_strategy(self, strategy, weight=None):
        """
        Add a strategy to the ensemble.
        
        Parameters:
        -----------
        strategy : Strategy
            Strategy to add.
        weight : float, optional
            Weight for the strategy. If None, adjusts weights to be equal.
            
        Returns:
        --------
        self
            Returns self for method chaining.
        """
        self.strategies.append(strategy)
        
        # Adjust weights
        if weight is None:
            # Equal weights
            self.weights = [1.0 / len(self.strategies)] * len(self.strategies)
        else:
            # Add new weight and normalize
            self.weights.append(weight)
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]
        
        logger.info(f"Added strategy '{strategy.name}' to ensemble '{self.name}'")
        return self
    
    def remove_strategy(self, index):
        """
        Remove a strategy from the ensemble.
        
        Parameters:
        -----------
        index : int
            Index of the strategy to remove.
            
        Returns:
        --------
        Strategy
            The removed strategy.
        """
        if index < 0 or index >= len(self.strategies):
            raise IndexError(f"Strategy index {index} out of range")
        
        strategy = self.strategies.pop(index)
        self.weights.pop(index)
        
        # Normalize remaining weights
        if self.weights:
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]
        
        logger.info(f"Removed strategy '{strategy.name}' from ensemble '{self.name}'")
        return strategy
    
    def set_weights(self, weights):
        """
        Set weights for the strategies.
        
        Parameters:
        -----------
        weights : list
            List of weights for each strategy.
            
        Returns:
        --------
        self
            Returns self for method chaining.
        """
        if len(weights) != len(self.strategies):
            raise ValueError("Number of weights must match number of strategies")
        
        # Normalize weights
        total = sum(weights)
        self.weights = [w / total for w in weights]
        
        logger.info(f"Updated weights for ensemble '{self.name}': {self.weights}")
        return self
    
    def generate_signals(self, data):
        """
        Generate trading signals by combining signals from all strategies.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical price data with OHLCV columns and any additional features.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with signal columns.
        """
        if not self.strategies:
            logger.warning(f"Ensemble '{self.name}' has no strategies")
            return pd.DataFrame(index=data.index)
        
        # Generate signals for each strategy
        all_signals = []
        for i, strategy in enumerate(self.strategies):
            signals = strategy.generate_signals(data)
            weight = self.weights[i]
            
            # Weight the signals
            for col in signals.columns:
                if signals[col].dtype == bool:
                    # Convert boolean to float for weighting
                    signals[col] = signals[col].astype(float)
                signals[col] *= weight
            
            all_signals.append(signals)
        
        # Combine signals
        combined = pd.DataFrame(index=data.index)
        
        # Sum weighted signals
        for signals in all_signals:
            for col in signals.columns:
                if col not in combined:
                    combined[col] = 0.0
                combined[col] += signals[col]
        
        # Threshold for final decision (0.5 for majority)
        threshold = 0.5
        
        # Convert to boolean signals
        for col in combined.columns:
            combined[col] = combined[col] >= threshold
        
        logger.info(f"Generated ensemble signals for '{self.name}'")
        return combined
    
    def _train_implementation(self, data, **kwargs):
        """
        Train all strategies in the ensemble.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical price data for training.
        **kwargs : dict
            Additional keyword arguments for training.
        """
        for strategy in self.strategies:
            strategy.train(data, **kwargs)
        
        logger.info(f"Trained all strategies in ensemble '{self.name}'")


# Utility functions for strategy development

def add_technical_indicators(df):
    """
    Add common technical indicators to a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLCV data.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added technical indicators.
    """
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Simple Moving Averages
    for window in [5, 10, 20, 50, 200]:
        result[f'sma_{window}'] = result['close'].rolling(window=window).mean()
    
    # Exponential Moving Averages
    for window in [5, 10, 20, 50, 200]:
        result[f'ema_{window}'] = result['close'].ewm(span=window, adjust=False).mean()
    
    # Bollinger Bands
    for window in [20]:
        mid = result['close'].rolling(window=window).mean()
        std = result['close'].rolling(window=window).std()
        result[f'bb_upper_{window}'] = mid + 2 * std
        result[f'bb_mid_{window}'] = mid
        result[f'bb_lower_{window}'] = mid - 2 * std
        result[f'bb_width_{window}'] = (result[f'bb_upper_{window}'] - result[f'bb_lower_{window}']) / mid
    
    # RSI (Relative Strength Index)
    for window in [14]:
        delta = result['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        result[f'rsi_{window}'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    fast_ema = result['close'].ewm(span=12, adjust=False).mean()
    slow_ema = result['close'].ewm(span=26, adjust=False).mean()
    result['macd'] = fast_ema - slow_ema
    result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
    result['macd_hist'] = result['macd'] - result['macd_signal']
    
    # Stochastic Oscillator
    for window in [14]:
        low_min = result['low'].rolling(window=window).min()
        high_max = result['high'].rolling(window=window).max()
        
        result[f'stoch_k_{window}'] = 100 * (result['close'] - low_min) / (high_max - low_min)
        result[f'stoch_d_{window}'] = result[f'stoch_k_{window}'].rolling(window=3).mean()
    
    # Average True Range (ATR)
    for window in [14]:
        tr1 = result['high'] - result['low']
        tr2 = abs(result['high'] - result['close'].shift())
        tr3 = abs(result['low'] - result['close'].shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        result[f'atr_{window}'] = tr.rolling(window=window).mean()
    
    # Average Directional Index (ADX)
    for window in [14]:
        # True Range
        tr1 = result['high'] - result['low']
        tr2 = abs(result['high'] - result['close'].shift())
        tr3 = abs(result['low'] - result['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = result['high'] - result['high'].shift()
        down_move = result['low'].shift() - result['low']
        
        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed True Range and Directional Movement
        atr = tr.rolling(window=window).mean()
        pos_di = 100 * (pd.Series(pos_dm).rolling(window=window).mean() / atr)
        neg_di = 100 * (pd.Series(neg_dm).rolling(window=window).mean() / atr)
        
        # ADX
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        result[f'adx_{window}'] = dx.rolling(window=window).mean()
        result[f'pos_di_{window}'] = pos_di
        result[f'neg_di_{window}'] = neg_di
    
    # On-Balance Volume (OBV)
    obv = (np.sign(result['close'].diff()) * result['volume']).fillna(0).cumsum()
    result['obv'] = obv
    
    # Commodity Channel Index (CCI)
    for window in [20]:
        tp = (result['high'] + result['low'] + result['close']) / 3
        tp_ma = tp.rolling(window=window).mean()
        tp_md = tp.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean())
        result[f'cci_{window}'] = (tp - tp_ma) / (0.015 * tp_md)
    
    # Williams %R
    for window in [14]:
        highest_high = result['high'].rolling(window=window).max()
        lowest_low = result['low'].rolling(window=window).min()
        result[f'williams_r_{window}'] = -100 * (highest_high - result['close']) / (highest_high - lowest_low)
    
    # Rate of Change (ROC)
    for window in [10]:
        result[f'roc_{window}'] = result['close'].pct_change(periods=window) * 100
    
    # Money Flow Index (MFI)
    for window in [14]:
        typical_price = (result['high'] + result['low'] + result['close']) / 3
        money_flow = typical_price * result['volume']
        
        positive_flow = np.where(typical_price > typical_price.shift(), money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(), money_flow, 0)
        
        positive_mf = pd.Series(positive_flow).rolling(window=window).sum()
        negative_mf = pd.Series(negative_flow).rolling(window=window).sum()
        
        money_ratio = positive_mf / negative_mf
        result[f'mfi_{window}'] = 100 - (100 / (1 + money_ratio))
    
    # Ichimoku Cloud
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    nine_period_high = result['high'].rolling(window=9).max()
    nine_period_low = result['low'].rolling(window=9).min()
    result['ichimoku_tenkan_sen'] = (nine_period_high + nine_period_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = result['high'].rolling(window=26).max()
    period26_low = result['low'].rolling(window=26).min()
    result['ichimoku_kijun_sen'] = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
    result['ichimoku_senkou_span_a'] = ((result['ichimoku_tenkan_sen'] + result['ichimoku_kijun_sen']) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = result['high'].rolling(window=52).max()
    period52_low = result['low'].rolling(window=52).min()
    result['ichimoku_senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)
    
    # Chikou Span (Lagging Span): Close price shifted back 26 periods
    result['ichimoku_chikou_span'] = result['close'].shift(-26)
    
    return result


if __name__ == "__main__":
    # Example usage
    print("Strategy base module ready for use")

"""
Data ingestion module for Nasdaq-100 E-mini futures trading bot.
This module handles loading, preprocessing, and feature engineering for historical data.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Class for loading and preprocessing historical OHLCV data for NQ futures.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize the DataLoader.
        
        Parameters:
        -----------
        data_dir : str, optional
            Directory containing historical data files. If None, uses default.
        """
        self.data_dir = data_dir or os.path.join(os.path.dirname(os.path.dirname(
                                               os.path.dirname(__file__))), 'data')
        logger.info(f"DataLoader initialized with data directory: {self.data_dir}")
    
    def load_csv_data(self, file_path):
        """
        Load data from a CSV file.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the loaded data.
        """
        try:
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            
            # Check if the timestamp column exists
            if 'timestamp' not in df.columns:
                raise ValueError("CSV file must contain a 'timestamp' column")
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            logger.info(f"Successfully loaded {len(df)} rows of data")
            return df
        
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def preprocess_data(self, df):
        """
        Preprocess the data by handling missing values, outliers, etc.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the raw data.
            
        Returns:
        --------
        pandas.DataFrame
            Preprocessed DataFrame.
        """
        logger.info("Preprocessing data")
        
        # Make a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Handle missing values
        if processed_df.isnull().sum().sum() > 0:
            logger.warning(f"Found {processed_df.isnull().sum().sum()} missing values")
            # Forward fill missing values
            processed_df.fillna(method='ffill', inplace=True)
            # If there are still missing values (at the beginning), backward fill
            processed_df.fillna(method='bfill', inplace=True)
        
        # Check for negative or zero prices
        for col in ['open', 'high', 'low', 'close']:
            if (processed_df[col] <= 0).any():
                logger.warning(f"Found non-positive values in {col} column")
                # Replace with NaN and then forward fill
                processed_df.loc[processed_df[col] <= 0, col] = np.nan
                processed_df[col].fillna(method='ffill', inplace=True)
                processed_df[col].fillna(method='bfill', inplace=True)
        
        # Ensure high >= low
        if (processed_df['high'] < processed_df['low']).any():
            logger.warning("Found high < low, swapping values")
            mask = processed_df['high'] < processed_df['low']
            processed_df.loc[mask, ['high', 'low']] = processed_df.loc[mask, ['low', 'high']].values
        
        # Ensure high >= open, high >= close
        processed_df['high'] = processed_df[['high', 'open', 'close']].max(axis=1)
        
        # Ensure low <= open, low <= close
        processed_df['low'] = processed_df[['low', 'open', 'close']].min(axis=1)
        
        # Handle negative volume
        if (processed_df['volume'] < 0).any():
            logger.warning("Found negative volume values")
            processed_df.loc[processed_df['volume'] < 0, 'volume'] = 0
        
        logger.info("Data preprocessing completed")
        return processed_df
    
    def engineer_features(self, df):
        """
        Engineer features from the preprocessed data.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Preprocessed DataFrame.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with engineered features.
        """
        logger.info("Engineering features")
        
        # Make a copy to avoid modifying the original
        feature_df = df.copy()
        
        # Calculate returns
        feature_df['returns'] = feature_df['close'].pct_change()
        
        # Calculate log returns
        feature_df['log_returns'] = np.log(feature_df['close'] / feature_df['close'].shift(1))
        
        # Calculate rolling statistics
        for window in [5, 10, 20]:
            # Rolling mean of close price
            feature_df[f'close_ma_{window}'] = feature_df['close'].rolling(window=window).mean()
            
            # Rolling standard deviation of returns
            feature_df[f'returns_std_{window}'] = feature_df['returns'].rolling(window=window).std()
            
            # Rolling mean of volume
            feature_df[f'volume_ma_{window}'] = feature_df['volume'].rolling(window=window).mean()
        
        # Calculate price momentum
        for period in [1, 5, 10, 20]:
            feature_df[f'momentum_{period}'] = feature_df['close'].pct_change(periods=period)
        
        # Calculate high-low range
        feature_df['hl_range'] = feature_df['high'] - feature_df['low']
        feature_df['hl_range_pct'] = feature_df['hl_range'] / feature_df['close']
        
        # Calculate open-close range
        feature_df['oc_range'] = abs(feature_df['open'] - feature_df['close'])
        feature_df['oc_range_pct'] = feature_df['oc_range'] / feature_df['close']
        
        # Calculate if close is higher than open (bullish candle)
        feature_df['bullish'] = (feature_df['close'] > feature_df['open']).astype(int)
        
        # Calculate if current close is higher than previous close
        feature_df['higher_close'] = (feature_df['close'] > feature_df['close'].shift(1)).astype(int)
        
        # Calculate if volume is higher than previous volume
        feature_df['higher_volume'] = (feature_df['volume'] > feature_df['volume'].shift(1)).astype(int)
        
        # Drop rows with NaN values resulting from calculations
        feature_df.dropna(inplace=True)
        
        logger.info(f"Feature engineering completed, {len(feature_df)} rows remaining")
        return feature_df
    
    def resample_timeframe(self, df, timeframe='5min'):
        """
        Resample data to a different timeframe.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to resample.
        timeframe : str, optional
            Target timeframe for resampling (e.g., '5min', '1H', '1D').
            
        Returns:
        --------
        pandas.DataFrame
            Resampled DataFrame.
        """
        logger.info(f"Resampling data to {timeframe} timeframe")
        
        # Define how to resample each column
        resampled = df.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        logger.info(f"Resampling completed, {len(resampled)} rows in resampled data")
        return resampled
    
    def split_data(self, df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Split data into training, validation, and test sets.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to split.
        train_ratio : float, optional
            Ratio of data to use for training.
        val_ratio : float, optional
            Ratio of data to use for validation.
        test_ratio : float, optional
            Ratio of data to use for testing.
            
        Returns:
        --------
        tuple
            (train_df, val_df, test_df)
        """
        logger.info("Splitting data into train, validation, and test sets")
        
        # Ensure ratios sum to 1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
        
        # Calculate split indices
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Split the data
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        logger.info(f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        return train_df, val_df, test_df
    
    def process_pipeline(self, file_path, resample_timeframe=None):
        """
        Run the complete data processing pipeline.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file.
        resample_timeframe : str, optional
            Target timeframe for resampling. If None, no resampling is performed.
            
        Returns:
        --------
        tuple
            (train_df, val_df, test_df) with engineered features.
        """
        logger.info(f"Running complete data processing pipeline for {file_path}")
        
        # Load data
        df = self.load_csv_data(file_path)
        
        # Preprocess data
        df = self.preprocess_data(df)
        
        # Resample if needed
        if resample_timeframe:
            df = self.resample_timeframe(df, resample_timeframe)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Split data
        train_df, val_df, test_df = self.split_data(df)
        
        logger.info("Data processing pipeline completed successfully")
        return train_df, val_df, test_df


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    # This would be replaced with actual file path
    # train_df, val_df, test_df = loader.process_pipeline('path/to/data.csv')
    print("DataLoader module ready for use")

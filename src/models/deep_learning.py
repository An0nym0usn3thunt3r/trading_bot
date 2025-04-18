"""
Deep Learning Models for Nasdaq-100 E-mini futures trading bot.
This module implements various deep learning models for prediction and decision making.
"""

import pandas as pd
import numpy as np
import logging
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score

# Try to import TensorFlow and Keras with error handling for environments with limited resources
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model, save_model
    from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization, Input, Concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.utils import plot_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow/Keras not available. Deep learning models will not function.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeepLearningModel:
    """
    Base class for deep learning models.
    """
    
    def __init__(self, name, model_type='classifier', feature_names=None, target_name=None, 
                 sequence_length=10, batch_size=32, epochs=100, validation_split=0.2, 
                 early_stopping_patience=10, learning_rate=0.001):
        """
        Initialize the deep learning model.
        
        Parameters:
        -----------
        name : str
            Name of the model.
        model_type : str, optional
            Type of model ('classifier' or 'regressor').
        feature_names : list, optional
            List of feature names to use.
        target_name : str, optional
            Name of the target variable.
        sequence_length : int, optional
            Length of input sequences for recurrent models.
        batch_size : int, optional
            Batch size for training.
        epochs : int, optional
            Maximum number of epochs for training.
        validation_split : float, optional
            Fraction of training data to use for validation.
        early_stopping_patience : int, optional
            Number of epochs with no improvement after which training will be stopped.
        learning_rate : float, optional
            Learning rate for the optimizer.
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow/Keras is not available. Cannot create deep learning model.")
        
        self.name = name
        self.model_type = model_type
        self.feature_names = feature_names
        self.target_name = target_name
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.learning_rate = learning_rate
        
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.is_fitted = False
        self.history = None
    
    def _create_model(self):
        """Create the deep learning model (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement _create_model method")
    
    def _prepare_data(self, X, y=None, is_training=True):
        """
        Prepare data for the model (scaling, sequence creation, etc.).
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Input features.
        y : pandas.Series or numpy.ndarray, optional
            Target variable.
        is_training : bool, optional
            Whether this is for training (True) or prediction (False).
        
        Returns:
        --------
        tuple
            Prepared X and y data.
        """
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if y is not None and isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
        
        # Scale features
        if is_training:
            self.scaler_X = StandardScaler()
            X_scaled = self.scaler_X.fit_transform(X)
            
            if y is not None:
                if self.model_type == 'regressor':
                    self.scaler_y = StandardScaler()
                    y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
                else:  # classifier
                    y_scaled = y
        else:
            if self.scaler_X is None:
                raise ValueError("Model has not been trained yet (scaler_X is None)")
            
            X_scaled = self.scaler_X.transform(X)
            
            if y is not None:
                if self.model_type == 'regressor' and self.scaler_y is not None:
                    y_scaled = self.scaler_y.transform(y.reshape(-1, 1)).flatten()
                else:  # classifier or scaler_y is None
                    y_scaled = y
        
        # Create sequences for recurrent models
        if hasattr(self, 'is_recurrent') and self.is_recurrent:
            X_seq = self._create_sequences(X_scaled)
            
            if y is not None:
                # For sequences, we need to adjust y to match the sequence structure
                y_seq = y_scaled[self.sequence_length-1:]
                return X_seq, y_seq
            else:
                return X_seq
        
        # Return scaled data
        if y is not None:
            return X_scaled, y_scaled
        else:
            return X_scaled
    
    def _create_sequences(self, data):
        """
        Create sequences for recurrent models.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Input data.
        
        Returns:
        --------
        numpy.ndarray
            Sequences of data.
        """
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i+self.sequence_length])
        return np.array(sequences)
    
    def fit(self, X, y, **kwargs):
        """
        Fit the model to the data.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Input features.
        y : pandas.Series or numpy.ndarray
            Target variable.
        **kwargs : dict
            Additional arguments to pass to the model's fit method.
        
        Returns:
        --------
        self
        """
        if self.model is None:
            self._create_model()
        
        # Prepare data
        X_prepared, y_prepared = self._prepare_data(X, y, is_training=True)
        
        # Create callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.early_stopping_patience // 2,
                min_lr=1e-6
            )
        ]
        
        # Add model checkpoint if a directory is provided
        if 'checkpoint_dir' in kwargs:
            checkpoint_path = os.path.join(kwargs.pop('checkpoint_dir'), f"{self.name}_best_model.h5")
            callbacks.append(
                ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor='val_loss',
                    save_best_only=True
                )
            )
        
        # Fit the model
        logger.info(f"Fitting {self.name} model...")
        self.history = self.model.fit(
            X_prepared, y_prepared,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=1,
            **kwargs
        )
        
        self.is_fitted = True
        logger.info(f"Fitted {self.name} model")
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the fitted model.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Input features.
        
        Returns:
        --------
        numpy.ndarray
            Predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet")
        
        # Prepare data
        X_prepared = self._prepare_data(X, is_training=False)
        
        # Make predictions
        predictions = self.model.predict(X_prepared)
        
        # Inverse transform for regression
        if self.model_type == 'regressor' and self.scaler_y is not None:
            predictions = self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Input features.
        
        Returns:
        --------
        numpy.ndarray
            Class probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet")
        
        if self.model_type != 'classifier':
            raise ValueError("predict_proba is only available for classifiers")
        
        # Prepare data
        X_prepared = self._prepare_data(X, is_training=False)
        
        # Make predictions
        return self.model.predict(X_prepared)
    
    def evaluate(self, X, y):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Input features.
        y : pandas.Series or numpy.ndarray
            True target values.
        
        Returns:
        --------
        dict
            Evaluation metrics.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet")
        
        # Prepare data
        X_prepared, y_prepared = self._prepare_data(X, y, is_training=False)
        
        # Evaluate model
        loss, *metrics = self.model.evaluate(X_prepared, y_prepared, verbose=0)
        
        # Get predictions for additional metrics
        y_pred = self.predict(X)
        
        # Calculate metrics
        if self.model_type == 'classifier':
            metrics_dict = {
                'loss': loss,
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted'),
                'recall': recall_score(y, y_pred, average='weighted'),
                'f1': f1_score(y, y_pred, average='weighted')
            }
        else:  # regressor
            metrics_dict = {
                'loss': loss,
                'mse': mean_squared_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'mae': np.mean(np.abs(y - y_pred))
            }
        
        logger.info(f"Evaluation metrics for {self.name}: {metrics_dict}")
        return metrics_dict
    
    def plot_training_history(self, save_path=None):
        """
        Plot the training history.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot. If None, the plot is displayed.
        
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure.
        """
        if self.history is None:
            raise ValueError("Model has not been trained yet (history is None)")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot metrics
        for metric in self.history.history:
            if metric not in ['loss', 'val_loss']:
                ax2.plot(self.history.history[metric], label=f'Training {metric}')
                val_metric = f'val_{metric}'
                if val_metric in self.history.history:
                    ax2.plot(self.history.history[val_metric], label=f'Validation {metric}')
        
        ax2.set_title('Metrics')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Value')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved training history plot to {save_path}")
        
        return fig
    
    def save(self, filepath):
        """
        Save the model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model.
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save Keras model
        model_path = f"{filepath}_keras_model.h5"
        self.model.save(model_path)
        
        # Save scalers and other attributes
        attrs = {
            'name': self.name,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'sequence_length': self.sequence_length,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'validation_split': self.validation_split,
            'early_stopping_patience': self.early_stopping_patience,
            'learning_rate': self.learning_rate,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'is_fitted': self.is_fitted,
            'model_path': model_path
        }
        
        joblib.dump(attrs, f"{filepath}_attributes.joblib")
        logger.info(f"Saved {self.name} model to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load a model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model.
        
        Returns:
        --------
        DeepLearningModel
            Loaded model.
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow/Keras is not available. Cannot load deep learning model.")
        
        # Load attributes
        attrs = joblib.load(f"{filepath}_attributes.joblib")
        
        # Create instance
        instance = cls(
            name=attrs['name'],
            model_type=attrs['model_type'],
            feature_names=attrs['feature_names'],
            target_name=attrs['target_name'],
            sequence_length=attrs['sequence_length'],
            batch_size=attrs['batch_size'],
            epochs=attrs['epochs'],
            validation_split=attrs['validation_split'],
            early_stopping_patience=attrs['early_stopping_patience'],
            learning_rate=attrs['learning_rate']
        )
        
        # Load Keras model
        instance.model = load_model(attrs['model_path'])
        
        # Set other attributes
        instance.scaler_X = attrs['scaler_X']
        instance.scaler_y = attrs['scaler_y']
        instance.is_fitted = attrs['is_fitted']
        
        logger.info(f"Loaded model from {filepath}")
        return instance


class LSTMModel(DeepLearningModel):
    """
    Long Short-Term Memory (LSTM) model implementation.
    """
    
    def __init__(self, name, model_type='classifier', feature_names=None, target_name=None, 
                 sequence_length=10, lstm_units=[64, 32], dropout_rate=0.2, 
                 batch_size=32, epochs=100, validation_split=0.2, 
                 early_stopping_patience=10, learning_rate=0.001):
        """
        Initialize the LSTM model.
        
        Parameters:
        -----------
        name : str
            Name of the model.
        model_type : str, optional
            Type of model ('classifier' or 'regressor').
        feature_names : list, optional
            List of feature names to use.
        target_name : str, optional
            Name of the target variable.
        sequence_length : int, optional
            Length of input sequences.
        lstm_units : list, optional
            Number of units in each LSTM layer.
        dropout_rate : float, optional
            Dropout rate for regularization.
        batch_size : int, optional
            Batch size for training.
        epochs : int, optional
            Maximum number of epochs for training.
        validation_split : float, optional
            Fraction of training data to use for validation.
        early_stopping_patience : int, optional
            Number of epochs with no improvement after which training will be stopped.
        learning_rate : float, optional
            Learning rate for the optimizer.
        """
        super().__init__(
            name=name,
            model_type=model_type,
            feature_names=feature_names,
            target_name=target_name,
            sequence_length=sequence_length,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            early_stopping_patience=early_stopping_patience,
            learning_rate=learning_rate
        )
        
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.is_recurrent = True
    
    def _create_model(self):
        """Create the LSTM model."""
        # Determine input shape
        if self.feature_names:
            input_dim = len(self.feature_names)
        else:
            # Assume input_dim will be determined during fit
            input_dim = None
        
        # Create model
        model = Sequential()
        
        # Add LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            
            if i == 0:
                model.add(LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    input_shape=(self.sequence_length, input_dim)
                ))
            else:
                model.add(LSTM(
                    units=units,
                    return_sequences=return_sequences
                ))
            
            model.add(Dropout(self.dropout_rate))
        
        # Add output layer
        if self.model_type == 'classifier':
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:  # regressor
            model.add(Dense(1, activation='linear'))
            loss = 'mse'
            metrics = ['mae']
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        self.model = model
        logger.info(f"Created {self.model_type} LSTM model with {len(self.lstm_units)} LSTM layers")
        
        return model


class GRUModel(DeepLearningModel):
    """
    Gated Recurrent Unit (GRU) model implementation.
    """
    
    def __init__(self, name, model_type='classifier', feature_names=None, target_name=None, 
                 sequence_length=10, gru_units=[64, 32], dropout_rate=0.2, 
                 batch_size=32, epochs=100, validation_split=0.2, 
                 early_stopping_patience=10, learning_rate=0.001):
        """
        Initialize the GRU model.
        
        Parameters:
        -----------
        name : str
            Name of the model.
        model_type : str, optional
            Type of model ('classifier' or 'regressor').
        feature_names : list, optional
            List of feature names to use.
        target_name : str, optional
            Name of the target variable.
        sequence_length : int, optional
            Length of input sequences.
        gru_units : list, optional
            Number of units in each GRU layer.
        dropout_rate : float, optional
            Dropout rate for regularization.
        batch_size : int, optional
            Batch size for training.
        epochs : int, optional
            Maximum number of epochs for training.
        validation_split : float, optional
            Fraction of training data to use for validation.
        early_stopping_patience : int, optional
            Number of epochs with no improvement after which training will be stopped.
        learning_rate : float, optional
            Learning rate for the optimizer.
        """
        super().__init__(
            name=name,
            model_type=model_type,
            feature_names=feature_names,
            target_name=target_name,
            sequence_length=sequence_length,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            early_stopping_patience=early_stopping_patience,
            learning_rate=learning_rate
        )
        
        self.gru_units = gru_units
        self.dropout_rate = dropout_rate
        self.is_recurrent = True
    
    def _create_model(self):
        """Create the GRU model."""
        # Determine input shape
        if self.feature_names:
            input_dim = len(self.feature_names)
        else:
            # Assume input_dim will be determined during fit
            input_dim = None
        
        # Create model
        model = Sequential()
        
        # Add GRU layers
        for i, units in enumerate(self.gru_units):
            return_sequences = i < len(self.gru_units) - 1
            
            if i == 0:
                model.add(GRU(
                    units=units,
                    return_sequences=return_sequences,
                    input_shape=(self.sequence_length, input_dim)
                ))
            else:
                model.add(GRU(
                    units=units,
                    return_sequences=return_sequences
                ))
            
            model.add(Dropout(self.dropout_rate))
        
        # Add output layer
        if self.model_type == 'classifier':
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:  # regressor
            model.add(Dense(1, activation='linear'))
            loss = 'mse'
            metrics = ['mae']
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        self.model = model
        logger.info(f"Created {self.model_type} GRU model with {len(self.gru_units)} GRU layers")
        
        return model


class DeepNeuralNetworkModel(DeepLearningModel):
    """
    Deep Neural Network model implementation.
    """
    
    def __init__(self, name, model_type='classifier', feature_names=None, target_name=None, 
                 hidden_layers=[128, 64, 32], dropout_rate=0.2, use_batch_norm=True, 
                 batch_size=32, epochs=100, validation_split=0.2, 
                 early_stopping_patience=10, learning_rate=0.001):
        """
        Initialize the Deep Neural Network model.
        
        Parameters:
        -----------
        name : str
            Name of the model.
        model_type : str, optional
            Type of model ('classifier' or 'regressor').
        feature_names : list, optional
            List of feature names to use.
        target_name : str, optional
            Name of the target variable.
        hidden_layers : list, optional
            Number of units in each hidden layer.
        dropout_rate : float, optional
            Dropout rate for regularization.
        use_batch_norm : bool, optional
            Whether to use batch normalization.
        batch_size : int, optional
            Batch size for training.
        epochs : int, optional
            Maximum number of epochs for training.
        validation_split : float, optional
            Fraction of training data to use for validation.
        early_stopping_patience : int, optional
            Number of epochs with no improvement after which training will be stopped.
        learning_rate : float, optional
            Learning rate for the optimizer.
        """
        super().__init__(
            name=name,
            model_type=model_type,
            feature_names=feature_names,
            target_name=target_name,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            early_stopping_patience=early_stopping_patience,
            learning_rate=learning_rate
        )
        
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.is_recurrent = False
    
    def _create_model(self):
        """Create the Deep Neural Network model."""
        # Create model
        model = Sequential()
        
        # Add input layer
        model.add(Dense(self.hidden_layers[0], activation='relu', input_dim=len(self.feature_names)))
        
        if self.use_batch_norm:
            model.add(BatchNormalization())
        
        model.add(Dropout(self.dropout_rate))
        
        # Add hidden layers
        for units in self.hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            
            if self.use_batch_norm:
                model.add(BatchNormalization())
            
            model.add(Dropout(self.dropout_rate))
        
        # Add output layer
        if self.model_type == 'classifier':
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:  # regressor
            model.add(Dense(1, activation='linear'))
            loss = 'mse'
            metrics = ['mae']
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        self.model = model
        logger.info(f"Created {self.model_type} DNN model with {len(self.hidden_layers)} hidden layers")
        
        return model


class DeepEnsembleModel(DeepLearningModel):
    """
    Deep Ensemble model that combines predictions from multiple deep learning models.
    """
    
    def __init__(self, name, base_models, ensemble_method='averaging', weights=None):
        """
        Initialize the Deep Ensemble model.
        
        Parameters:
        -----------
        name : str
            Name of the model.
        base_models : list
            List of base deep learning models to ensemble.
        ensemble_method : str, optional
            Method for combining predictions ('averaging', 'stacking').
        weights : list, optional
            Weights for each base model (used in 'averaging' method).
        """
        if not base_models:
            raise ValueError("base_models cannot be empty")
        
        # Determine model type from base models
        model_type = base_models[0].model_type
        
        # Initialize with parameters from first base model
        super().__init__(
            name=name,
            model_type=model_type,
            feature_names=base_models[0].feature_names,
            target_name=base_models[0].target_name,
            batch_size=base_models[0].batch_size,
            epochs=base_models[0].epochs,
            validation_split=base_models[0].validation_split,
            early_stopping_patience=base_models[0].early_stopping_patience,
            learning_rate=base_models[0].learning_rate
        )
        
        self.base_models = base_models
        self.ensemble_method = ensemble_method
        self.weights = weights
        
        # Validate weights if provided
        if weights is not None and len(weights) != len(base_models):
            raise ValueError("Number of weights must match number of base models")
        
        # Determine if this is a recurrent model ensemble
        self.is_recurrent = all(hasattr(model, 'is_recurrent') and model.is_recurrent for model in base_models)
        
        if self.is_recurrent:
            # Use sequence length from first model
            self.sequence_length = base_models[0].sequence_length
    
    def _create_model(self):
        """Create the ensemble model."""
        if self.ensemble_method == 'stacking':
            # Create a stacking ensemble model
            
            # Ensure all base models are fitted
            if not all(model.is_fitted for model in self.base_models):
                raise ValueError("All base models must be fitted before creating a stacking ensemble")
            
            # Create input layer
            inputs = Input(shape=(len(self.feature_names),))
            
            # Create a dense layer for each base model's output
            model_outputs = []
            for i, model in enumerate(self.base_models):
                # We'll use the base model's predictions as features
                # This is a simplified version; in practice, you'd use the base model's architecture
                dense = Dense(32, activation='relu', name=f'model_{i}_dense')(inputs)
                output = Dense(1, activation='linear' if self.model_type == 'regressor' else 'sigmoid', 
                               name=f'model_{i}_output')(dense)
                model_outputs.append(output)
            
            # Combine outputs
            if len(model_outputs) > 1:
                combined = Concatenate()(model_outputs)
                
                # Add a few more layers to learn the optimal combination
                x = Dense(16, activation='relu')(combined)
                x = BatchNormalization()(x)
                x = Dropout(0.2)(x)
                
                # Final output
                if self.model_type == 'classifier':
                    output = Dense(1, activation='sigmoid')(x)
                    loss = 'binary_crossentropy'
                    metrics = ['accuracy']
                else:  # regressor
                    output = Dense(1, activation='linear')(x)
                    loss = 'mse'
                    metrics = ['mae']
            else:
                # Only one base model, use its output directly
                output = model_outputs[0]
                
                if self.model_type == 'classifier':
                    loss = 'binary_crossentropy'
                    metrics = ['accuracy']
                else:  # regressor
                    loss = 'mse'
                    metrics = ['mae']
            
            # Create model
            model = Model(inputs=inputs, outputs=output)
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss=loss,
                metrics=metrics
            )
            
            self.model = model
            logger.info(f"Created {self.model_type} stacking ensemble model with {len(self.base_models)} base models")
        else:
            # For averaging ensemble, we don't create a new model
            # We'll implement the averaging in the predict method
            self.model = None
            logger.info(f"Created {self.model_type} averaging ensemble model with {len(self.base_models)} base models")
    
    def fit(self, X, y, **kwargs):
        """
        Fit the model to the data.
        
        For averaging ensemble, this method does nothing as the base models are already fitted.
        For stacking ensemble, this method fits the stacking model.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Input features.
        y : pandas.Series or numpy.ndarray
            Target variable.
        **kwargs : dict
            Additional arguments to pass to the model's fit method.
        
        Returns:
        --------
        self
        """
        if self.ensemble_method == 'stacking':
            # Create and fit the stacking model
            if self.model is None:
                self._create_model()
            
            # Prepare data
            X_prepared, y_prepared = self._prepare_data(X, y, is_training=True)
            
            # Create callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.early_stopping_patience,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=self.early_stopping_patience // 2,
                    min_lr=1e-6
                )
            ]
            
            # Add model checkpoint if a directory is provided
            if 'checkpoint_dir' in kwargs:
                checkpoint_path = os.path.join(kwargs.pop('checkpoint_dir'), f"{self.name}_best_model.h5")
                callbacks.append(
                    ModelCheckpoint(
                        filepath=checkpoint_path,
                        monitor='val_loss',
                        save_best_only=True
                    )
                )
            
            # Fit the model
            logger.info(f"Fitting {self.name} stacking ensemble model...")
            self.history = self.model.fit(
                X_prepared, y_prepared,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=self.validation_split,
                callbacks=callbacks,
                verbose=1,
                **kwargs
            )
        else:
            # For averaging ensemble, we don't need to fit anything
            # Just check that all base models are fitted
            if not all(model.is_fitted for model in self.base_models):
                raise ValueError("All base models must be fitted before using an averaging ensemble")
            
            logger.info(f"Using pre-fitted base models for {self.name} averaging ensemble model")
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Make predictions using the ensemble of base models.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Input features.
        
        Returns:
        --------
        numpy.ndarray
            Predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet")
        
        if self.ensemble_method == 'stacking' and self.model is not None:
            # Use the stacking model for predictions
            X_prepared = self._prepare_data(X, is_training=False)
            predictions = self.model.predict(X_prepared)
            
            # Inverse transform for regression
            if self.model_type == 'regressor' and self.scaler_y is not None:
                predictions = self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
            
            return predictions
        else:
            # Use averaging for predictions
            all_predictions = []
            
            for model in self.base_models:
                predictions = model.predict(X)
                all_predictions.append(predictions)
            
            # Convert to array for easier manipulation
            all_predictions = np.array(all_predictions)
            
            # Apply weights if provided
            if self.weights is not None:
                weights = np.array(self.weights).reshape(-1, 1)
                weighted_predictions = np.sum(all_predictions * weights, axis=0) / np.sum(weights)
                return weighted_predictions
            else:
                # Simple averaging
                return np.mean(all_predictions, axis=0)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X using the ensemble of base models.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Input features.
        
        Returns:
        --------
        numpy.ndarray
            Class probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet")
        
        if self.model_type != 'classifier':
            raise ValueError("predict_proba is only available for classifiers")
        
        if self.ensemble_method == 'stacking' and self.model is not None:
            # Use the stacking model for predictions
            X_prepared = self._prepare_data(X, is_training=False)
            return self.model.predict(X_prepared)
        else:
            # Use averaging for predictions
            all_probas = []
            
            for model in self.base_models:
                probas = model.predict_proba(X)
                all_probas.append(probas)
            
            # Convert to array for easier manipulation
            all_probas = np.array(all_probas)
            
            # Apply weights if provided
            if self.weights is not None:
                weights = np.array(self.weights).reshape(-1, 1, 1)
                weighted_probas = np.sum(all_probas * weights, axis=0) / np.sum(weights)
                return weighted_probas
            else:
                # Simple averaging
                return np.mean(all_probas, axis=0)


class DeepModelTrainer:
    """
    Class for training and evaluating deep learning models.
    """
    
    def __init__(self, data, target_name, feature_names=None, test_size=0.2, random_state=42):
        """
        Initialize the deep model trainer.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data containing features and target.
        target_name : str
            Name of the target variable.
        feature_names : list, optional
            List of feature names to use. If None, uses all columns except target.
        test_size : float, optional
            Proportion of the data to use for testing.
        random_state : int, optional
            Random seed for reproducibility.
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow/Keras is not available. Cannot create deep model trainer.")
        
        self.data = data
        self.target_name = target_name
        
        # Set feature names if not provided
        if feature_names is None:
            self.feature_names = [col for col in data.columns if col != target_name]
        else:
            self.feature_names = feature_names
        
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        
        # Split data into features and target
        self.X = data[self.feature_names]
        self.y = data[target_name]
        
        # Split data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Initialized DeepModelTrainer with {len(self.feature_names)} features and {len(self.data)} samples")
        logger.info(f"Train set: {len(self.X_train)} samples, Test set: {len(self.X_test)} samples")
    
    def add_model(self, model):
        """
        Add a model to the trainer.
        
        Parameters:
        -----------
        model : DeepLearningModel
            Model to add.
        """
        self.models[model.name] = model
        logger.info(f"Added model: {model.name}")
    
    def train_all_models(self, **kwargs):
        """
        Train all added models.
        
        Parameters:
        -----------
        **kwargs : dict
            Additional arguments to pass to the models' fit methods.
        """
        for name, model in self.models.items():
            logger.info(f"Training model: {name}")
            model.fit(self.X_train, self.y_train, **kwargs)
    
    def evaluate_all_models(self):
        """
        Evaluate all trained models.
        
        Returns:
        --------
        dict
            Evaluation metrics for all models.
        """
        metrics = {}
        
        for name, model in self.models.items():
            if model.is_fitted:
                logger.info(f"Evaluating model: {name}")
                metrics[name] = model.evaluate(self.X_test, self.y_test)
            else:
                logger.warning(f"Model {name} is not fitted yet")
        
        return metrics
    
    def create_ensemble(self, name, models_to_include=None, ensemble_method='averaging', weights=None):
        """
        Create an ensemble model from trained models.
        
        Parameters:
        -----------
        name : str
            Name of the ensemble model.
        models_to_include : list, optional
            List of model names to include in the ensemble. If None, includes all models.
        ensemble_method : str, optional
            Method for combining predictions ('averaging', 'stacking').
        weights : list, optional
            Weights for each base model.
        
        Returns:
        --------
        DeepEnsembleModel
            Created ensemble model.
        """
        # Select models to include
        if models_to_include is None:
            base_models = list(self.models.values())
        else:
            base_models = [self.models[name] for name in models_to_include if name in self.models]
        
        # Check if we have enough models
        if len(base_models) < 2:
            raise ValueError("Need at least 2 models to create an ensemble")
        
        # Create ensemble model
        ensemble = DeepEnsembleModel(name, base_models, ensemble_method, weights)
        
        # Add to models dictionary
        self.models[name] = ensemble
        
        logger.info(f"Created ensemble model {name} with {len(base_models)} base models")
        return ensemble
    
    def save_models(self, directory):
        """
        Save all trained models to files.
        
        Parameters:
        -----------
        directory : str
            Directory to save the models.
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.models.items():
            if model.is_fitted:
                filepath = os.path.join(directory, name)
                model.save(filepath)
            else:
                logger.warning(f"Model {name} is not fitted yet, skipping save")


if __name__ == "__main__":
    # Example usage
    if TENSORFLOW_AVAILABLE:
        print("Deep Learning Models module ready for use")
    else:
        print("TensorFlow/Keras not available. Deep learning models will not function.")

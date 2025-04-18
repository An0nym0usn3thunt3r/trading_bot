"""
Base AI/ML Models for Nasdaq-100 E-mini futures trading bot.
This module implements various AI/ML models for prediction and decision making.
"""

import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer for selecting features from a DataFrame.
    """
    
    def __init__(self, feature_names=None):
        """
        Initialize the feature selector.
        
        Parameters:
        -----------
        feature_names : list, optional
            List of feature names to select. If None, selects all features.
        """
        self.feature_names = feature_names
    
    def fit(self, X, y=None):
        """Fit method (does nothing but required for scikit-learn API)."""
        return self
    
    def transform(self, X):
        """
        Select features from X.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data.
            
        Returns:
        --------
        pandas.DataFrame
            Selected features.
        """
        if self.feature_names is None:
            return X
        
        return X[self.feature_names]


class BaseMLModel:
    """
    Base class for machine learning models.
    """
    
    def __init__(self, name, model_type='classifier', feature_names=None, target_name=None):
        """
        Initialize the base ML model.
        
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
        """
        self.name = name
        self.model_type = model_type
        self.feature_names = feature_names
        self.target_name = target_name
        self.model = None
        self.pipeline = None
        self.is_fitted = False
    
    def build_pipeline(self, scaler_type='standard'):
        """
        Build the model pipeline.
        
        Parameters:
        -----------
        scaler_type : str, optional
            Type of scaler to use ('standard' or 'minmax').
        """
        # Create feature selector
        feature_selector = FeatureSelector(self.feature_names)
        
        # Create scaler
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        # Create model
        if self.model is None:
            self._create_model()
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('selector', feature_selector),
            ('scaler', scaler),
            ('model', self.model)
        ])
        
        logger.info(f"Built pipeline for {self.name} model")
    
    def _create_model(self):
        """Create the underlying model (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement _create_model method")
    
    def fit(self, X, y, **kwargs):
        """
        Fit the model to the data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input features.
        y : pandas.Series or numpy.ndarray
            Target variable.
        **kwargs : dict
            Additional arguments to pass to the model's fit method.
        
        Returns:
        --------
        self
        """
        if self.pipeline is None:
            self.build_pipeline()
        
        logger.info(f"Fitting {self.name} model...")
        self.pipeline.fit(X, y, **kwargs)
        self.is_fitted = True
        logger.info(f"Fitted {self.name} model")
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the fitted model.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input features.
        
        Returns:
        --------
        numpy.ndarray
            Predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet")
        
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters:
        -----------
        X : pandas.DataFrame
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
        
        return self.pipeline.predict_proba(X)
    
    def evaluate(self, X, y):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
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
        
        y_pred = self.predict(X)
        
        if self.model_type == 'classifier':
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted'),
                'recall': recall_score(y, y_pred, average='weighted'),
                'f1': f1_score(y, y_pred, average='weighted')
            }
        else:  # regressor
            metrics = {
                'mse': mean_squared_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'mae': np.mean(np.abs(y - y_pred))
            }
        
        logger.info(f"Evaluation metrics for {self.name}: {metrics}")
        return metrics
    
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
        
        joblib.dump(self, filepath)
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
        BaseMLModel
            Loaded model.
        """
        model = joblib.load(filepath)
        logger.info(f"Loaded model from {filepath}")
        return model


class RandomForestModel(BaseMLModel):
    """
    Random Forest model implementation.
    """
    
    def __init__(self, name, model_type='classifier', feature_names=None, target_name=None, 
                 n_estimators=100, max_depth=None, random_state=42, **kwargs):
        """
        Initialize the Random Forest model.
        
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
        n_estimators : int, optional
            Number of trees in the forest.
        max_depth : int, optional
            Maximum depth of the trees.
        random_state : int, optional
            Random seed for reproducibility.
        **kwargs : dict
            Additional arguments to pass to the Random Forest constructor.
        """
        super().__init__(name, model_type, feature_names, target_name)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.kwargs = kwargs
    
    def _create_model(self):
        """Create the Random Forest model."""
        if self.model_type == 'classifier':
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                **self.kwargs
            )
        else:  # regressor
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                **self.kwargs
            )
        
        logger.info(f"Created {self.model_type} Random Forest model with {self.n_estimators} estimators")


class GradientBoostingModel(BaseMLModel):
    """
    Gradient Boosting model implementation.
    """
    
    def __init__(self, name, model_type='classifier', feature_names=None, target_name=None, 
                 n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, **kwargs):
        """
        Initialize the Gradient Boosting model.
        
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
        n_estimators : int, optional
            Number of boosting stages.
        learning_rate : float, optional
            Learning rate shrinks the contribution of each tree.
        max_depth : int, optional
            Maximum depth of the trees.
        random_state : int, optional
            Random seed for reproducibility.
        **kwargs : dict
            Additional arguments to pass to the Gradient Boosting constructor.
        """
        super().__init__(name, model_type, feature_names, target_name)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.kwargs = kwargs
    
    def _create_model(self):
        """Create the Gradient Boosting model."""
        if self.model_type == 'classifier':
            self.model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                **self.kwargs
            )
        else:  # regressor
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                **self.kwargs
            )
        
        logger.info(f"Created {self.model_type} Gradient Boosting model with {self.n_estimators} estimators")


class SVMModel(BaseMLModel):
    """
    Support Vector Machine model implementation.
    """
    
    def __init__(self, name, model_type='classifier', feature_names=None, target_name=None, 
                 kernel='rbf', C=1.0, gamma='scale', random_state=42, **kwargs):
        """
        Initialize the SVM model.
        
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
        kernel : str, optional
            Kernel type to be used in the algorithm.
        C : float, optional
            Regularization parameter.
        gamma : str or float, optional
            Kernel coefficient.
        random_state : int, optional
            Random seed for reproducibility.
        **kwargs : dict
            Additional arguments to pass to the SVM constructor.
        """
        super().__init__(name, model_type, feature_names, target_name)
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.random_state = random_state
        self.kwargs = kwargs
    
    def _create_model(self):
        """Create the SVM model."""
        if self.model_type == 'classifier':
            self.model = SVC(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                random_state=self.random_state,
                probability=True,
                **self.kwargs
            )
        else:  # regressor
            self.model = SVR(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                **self.kwargs
            )
        
        logger.info(f"Created {self.model_type} SVM model with {self.kernel} kernel")


class NeuralNetworkModel(BaseMLModel):
    """
    Neural Network model implementation using scikit-learn's MLPClassifier/MLPRegressor.
    """
    
    def __init__(self, name, model_type='classifier', feature_names=None, target_name=None, 
                 hidden_layer_sizes=(100,), activation='relu', solver='adam', 
                 alpha=0.0001, learning_rate='constant', random_state=42, **kwargs):
        """
        Initialize the Neural Network model.
        
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
        hidden_layer_sizes : tuple, optional
            The ith element represents the number of neurons in the ith hidden layer.
        activation : str, optional
            Activation function for the hidden layer.
        solver : str, optional
            The solver for weight optimization.
        alpha : float, optional
            L2 penalty (regularization term) parameter.
        learning_rate : str, optional
            Learning rate schedule for weight updates.
        random_state : int, optional
            Random seed for reproducibility.
        **kwargs : dict
            Additional arguments to pass to the Neural Network constructor.
        """
        super().__init__(name, model_type, feature_names, target_name)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.kwargs = kwargs
    
    def _create_model(self):
        """Create the Neural Network model."""
        if self.model_type == 'classifier':
            self.model = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                alpha=self.alpha,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                **self.kwargs
            )
        else:  # regressor
            self.model = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                alpha=self.alpha,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                **self.kwargs
            )
        
        logger.info(f"Created {self.model_type} Neural Network model with {self.hidden_layer_sizes} hidden layers")


class EnsembleModel(BaseMLModel):
    """
    Ensemble model that combines predictions from multiple base models.
    """
    
    def __init__(self, name, base_models, ensemble_method='voting', weights=None):
        """
        Initialize the Ensemble model.
        
        Parameters:
        -----------
        name : str
            Name of the model.
        base_models : list
            List of base models to ensemble.
        ensemble_method : str, optional
            Method for combining predictions ('voting', 'averaging', 'stacking').
        weights : list, optional
            Weights for each base model (used in 'voting' and 'averaging' methods).
        """
        # Determine model type from base models
        model_type = base_models[0].model_type if base_models else 'classifier'
        super().__init__(name, model_type)
        
        self.base_models = base_models
        self.ensemble_method = ensemble_method
        self.weights = weights
        
        # Validate weights if provided
        if weights is not None and len(weights) != len(base_models):
            raise ValueError("Number of weights must match number of base models")
    
    def _create_model(self):
        """Create the ensemble model (not used in this class)."""
        pass  # No single model to create
    
    def fit(self, X, y, **kwargs):
        """
        Fit all base models to the data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input features.
        y : pandas.Series or numpy.ndarray
            Target variable.
        **kwargs : dict
            Additional arguments to pass to the base models' fit methods.
        
        Returns:
        --------
        self
        """
        logger.info(f"Fitting {self.name} ensemble model with {len(self.base_models)} base models...")
        
        for model in self.base_models:
            model.fit(X, y, **kwargs)
        
        self.is_fitted = True
        logger.info(f"Fitted {self.name} ensemble model")
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the ensemble of base models.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input features.
        
        Returns:
        --------
        numpy.ndarray
            Predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet")
        
        # Get predictions from all base models
        predictions = [model.predict(X) for model in self.base_models]
        
        if self.ensemble_method == 'voting':
            # For classification, use majority voting
            if self.model_type == 'classifier':
                # Convert to array for easier manipulation
                predictions_array = np.array(predictions)
                
                # Apply weights if provided
                if self.weights is not None:
                    # Create a weighted vote count for each class
                    unique_classes = np.unique(np.concatenate(predictions))
                    weighted_votes = np.zeros((len(X), len(unique_classes)))
                    
                    for i, pred in enumerate(predictions):
                        for j, cls in enumerate(unique_classes):
                            weighted_votes[:, j] += (pred == cls) * self.weights[i]
                    
                    # Return the class with the highest weighted vote
                    return unique_classes[np.argmax(weighted_votes, axis=1)]
                else:
                    # Simple majority voting
                    return np.apply_along_axis(
                        lambda x: np.bincount(x).argmax(),
                        axis=0,
                        arr=predictions_array
                    )
            else:  # regressor
                # For regression, use weighted average
                if self.weights is not None:
                    return np.average(predictions, axis=0, weights=self.weights)
                else:
                    return np.mean(predictions, axis=0)
        
        elif self.ensemble_method == 'averaging':
            # For classification, average the probabilities
            if self.model_type == 'classifier':
                # Get probability predictions from all base models
                proba_predictions = [model.predict_proba(X) for model in self.base_models]
                
                # Apply weights if provided
                if self.weights is not None:
                    avg_proba = np.zeros_like(proba_predictions[0])
                    for i, proba in enumerate(proba_predictions):
                        avg_proba += proba * self.weights[i]
                    avg_proba /= sum(self.weights)
                else:
                    avg_proba = np.mean(proba_predictions, axis=0)
                
                # Return the class with the highest probability
                return np.argmax(avg_proba, axis=1)
            else:  # regressor
                # For regression, use weighted average
                if self.weights is not None:
                    return np.average(predictions, axis=0, weights=self.weights)
                else:
                    return np.mean(predictions, axis=0)
        
        elif self.ensemble_method == 'stacking':
            # Stacking is not implemented here as it requires a meta-learner
            raise NotImplementedError("Stacking ensemble method is not implemented yet")
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X using the ensemble of base models.
        
        Parameters:
        -----------
        X : pandas.DataFrame
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
        
        # Get probability predictions from all base models
        proba_predictions = [model.predict_proba(X) for model in self.base_models]
        
        # Apply weights if provided
        if self.weights is not None:
            avg_proba = np.zeros_like(proba_predictions[0])
            for i, proba in enumerate(proba_predictions):
                avg_proba += proba * self.weights[i]
            avg_proba /= sum(self.weights)
        else:
            avg_proba = np.mean(proba_predictions, axis=0)
        
        return avg_proba
    
    def evaluate(self, X, y):
        """
        Evaluate the ensemble model on test data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input features.
        y : pandas.Series or numpy.ndarray
            True target values.
        
        Returns:
        --------
        dict
            Evaluation metrics for the ensemble and each base model.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet")
        
        # Evaluate ensemble model
        ensemble_metrics = super().evaluate(X, y)
        
        # Evaluate each base model
        base_model_metrics = {}
        for model in self.base_models:
            base_model_metrics[model.name] = model.evaluate(X, y)
        
        # Combine metrics
        metrics = {
            'ensemble': ensemble_metrics,
            'base_models': base_model_metrics
        }
        
        return metrics


class ModelTrainer:
    """
    Class for training and evaluating machine learning models.
    """
    
    def __init__(self, data, target_name, feature_names=None, test_size=0.2, random_state=42):
        """
        Initialize the model trainer.
        
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
        
        logger.info(f"Initialized ModelTrainer with {len(self.feature_names)} features and {len(self.data)} samples")
        logger.info(f"Train set: {len(self.X_train)} samples, Test set: {len(self.X_test)} samples")
    
    def add_model(self, model):
        """
        Add a model to the trainer.
        
        Parameters:
        -----------
        model : BaseMLModel
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
    
    def create_ensemble(self, name, models_to_include=None, ensemble_method='voting', weights=None):
        """
        Create an ensemble model from trained models.
        
        Parameters:
        -----------
        name : str
            Name of the ensemble model.
        models_to_include : list, optional
            List of model names to include in the ensemble. If None, includes all models.
        ensemble_method : str, optional
            Method for combining predictions ('voting', 'averaging', 'stacking').
        weights : list, optional
            Weights for each base model.
        
        Returns:
        --------
        EnsembleModel
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
        ensemble = EnsembleModel(name, base_models, ensemble_method, weights)
        
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
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.models.items():
            if model.is_fitted:
                filepath = os.path.join(directory, f"{name}.joblib")
                model.save(filepath)
            else:
                logger.warning(f"Model {name} is not fitted yet, skipping save")


if __name__ == "__main__":
    # Example usage
    print("Base AI/ML Models module ready for use")

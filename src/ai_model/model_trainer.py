import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple, Callable
import logging
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import joblib
import os
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Trains machine learning models on processed data."""
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the ModelTrainer.
        
        Args:
            model_dir (str): Directory to save trained models
        """
        self.model_dir = model_dir
        self.models = {}
        self.metrics = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        logger.info(f"Initialized ModelTrainer with model directory: {self.model_dir}")
    
    def train(self, X: pd.DataFrame, y: pd.Series, model_type: str = "random_forest", 
              problem_type: str = "regression", model_params: Dict[str, Any] = None, 
              test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Train a machine learning model.
        
        Args:
            X (pd.DataFrame): Feature data
            y (pd.Series): Target data
            model_type (str): Type of model to train ('random_forest', 'linear', 'decision_tree')
            problem_type (str): Type of problem ('regression' or 'classification')
            model_params (Dict[str, Any]): Parameters for the model
            test_size (float): Fraction of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            Dict[str, Any]: Training results and metrics
        """
        if X.empty or len(y) == 0:
            logger.error("Empty features or target data provided for training")
            return {"status": "error", "message": "Empty data"}
        
        logger.info(f"Training {model_type} {problem_type} model on {len(X)} samples")
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Get the appropriate model
        model = self._get_model(model_type, problem_type, model_params)
        
        if model is None:
            return {"status": "error", "message": f"Invalid model type: {model_type} for {problem_type}"}
        
        try:
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, problem_type)
            
            # Create a unique model ID
            model_id = f"{model_type}_{problem_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save the model
            model_path = os.path.join(self.model_dir, f"{model_id}.joblib")
            joblib.dump(model, model_path)
            
            # Store the model and metrics
            self.models[model_id] = model
            self.metrics[model_id] = metrics
            
            # Feature importance (if available)
            feature_importance = self._get_feature_importance(model, X.columns)
            
            # Prepare result
            result = {
                "status": "success",
                "model_id": model_id,
                "model_type": model_type,
                "problem_type": problem_type,
                "metrics": metrics,
                "feature_importance": feature_importance,
                "model_path": model_path,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_count": X.shape[1],
                "training_date": datetime.datetime.now().isoformat()
            }
            
            logger.info(f"Successfully trained model {model_id} with {metrics}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return {"status": "error", "message": str(e)}
    
    def _get_model(self, model_type: str, problem_type: str, params: Dict[str, Any] = None) -> Any:
        """
        Get an instance of the specified model type.
        
        Args:
            model_type (str): Type of model
            problem_type (str): Type of problem ('regression' or 'classification')
            params (Dict[str, Any]): Model parameters
            
        Returns:
            object: Model instance
        """
        params = params or {}
        
        if problem_type == "regression":
            if model_type == "linear":
                return LinearRegression(**params)
            elif model_type == "random_forest":
                default_params = {"n_estimators": 100, "random_state": 42}
                return RandomForestRegressor(**{**default_params, **params})
            elif model_type == "decision_tree":
                default_params = {"random_state": 42}
                return DecisionTreeRegressor(**{**default_params, **params})
            else:
                logger.warning(f"Unknown regression model type: {model_type}")
                return None
                
        elif problem_type == "classification":
            if model_type == "logistic":
                default_params = {"random_state": 42, "max_iter": 1000}
                return LogisticRegression(**{**default_params, **params})
            elif model_type == "random_forest":
                default_params = {"n_estimators": 100, "random_state": 42}
                return RandomForestClassifier(**{**default_params, **params})
            elif model_type == "decision_tree":
                default_params = {"random_state": 42}
                return DecisionTreeClassifier(**{**default_params, **params})
            else:
                logger.warning(f"Unknown classification model type: {model_type}")
                return None
        else:
            logger.warning(f"Unknown problem type: {problem_type}")
            return None
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, problem_type: str) -> Dict[str, float]:
        """
        Calculate performance metrics based on problem type.
        
        Args:
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted values
            problem_type (str): Type of problem ('regression' or 'classification')
            
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        if problem_type == "regression":
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            mae = np.mean(np.abs(y_true - y_pred))
            
            return {
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
                "mae": mae
            }
            
        elif problem_type == "classification":
            accuracy = accuracy_score(y_true, y_pred)
            
            # For multi-class, we need to specify averaging method
            if len(np.unique(y_true)) > 2:
                precision = precision_score(y_true, y_pred, average='weighted')
                recall = recall_score(y_true, y_pred, average='weighted')
                f1 = f1_score(y_true, y_pred, average='weighted')
            else:
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
            
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        
        else:
            logger.warning(f"Unknown problem type for metrics: {problem_type}")
            return {}
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """
        Extract feature importance from the model if available.
        
        Args:
            model: Trained model
            feature_names (List[str]): List of feature names
            
        Returns:
            Dict[str, float]: Feature importance dictionary
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                return {name: float(importance) for name, importance in zip(feature_names, importances)}
            elif hasattr(model, 'coef_'):
                # For linear models
                coefficients = model.coef_
                if coefficients.ndim > 1:
                    # For multi-class, take average of absolute values
                    coefficients = np.mean(np.abs(coefficients), axis=0)
                return {name: float(coef) for name, coef in zip(feature_names, coefficients)}
            else:
                return {}
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return {}
    
    def load_model(self, model_id: str) -> Any:
        """
        Load a trained model by ID.
        
        Args:
            model_id (str): Model identifier
            
        Returns:
            object: Loaded model
        """
        if model_id in self.models:
            return self.models[model_id]
        
        try:
            model_path = os.path.join(self.model_dir, f"{model_id}.joblib")
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return None
            
            model = joblib.load(model_path)
            self.models[model_id] = model
            logger.info(f"Successfully loaded model {model_id}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            return None
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, model_type: str, problem_type: str,
                                param_grid: Dict[str, List[Any]], cv: int = 5) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization using grid search.
        
        Args:
            X (pd.DataFrame): Feature data
            y (pd.Series): Target data
            model_type (str): Type of model
            problem_type (str): Type of problem ('regression' or 'classification')
            param_grid (Dict[str, List[Any]]): Grid of parameters to search
            cv (int): Number of cross-validation folds
            
        Returns:
            Dict[str, Any]: Best parameters and results
        """
        if X.empty or len(y) == 0:
            logger.error("Empty features or target data provided for hyperparameter optimization")
            return {"status": "error", "message": "Empty data"}
        
        logger.info(f"Optimizing hyperparameters for {model_type} {problem_type} model with {cv}-fold CV")
        
        # Get base model (with default parameters)
        base_model = self._get_model(model_type, problem_type)
        
        if base_model is None:
            return {"status": "error", "message": f"Invalid model type: {model_type} for {problem_type}"}
        
        try:
            # Set up scoring metric based on problem type
            if problem_type == "regression":
                scoring = "neg_mean_squared_error"
            else:
                scoring = "f1_weighted"
            
            # Create grid search
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
            
            # Perform grid search
            grid_search.fit(X, y)
            
            # Get best parameters and model
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            # Train a new model with the best parameters
            optimized_result = self.train(
                X, y,
                model_type=model_type,
                problem_type=problem_type,
                model_params=best_params
            )
            
            # Add grid search information to result
            optimized_result["optimization"] = {
                "best_params": best_params,
                "best_cv_score": best_score,
                "param_grid": param_grid,
                "cv_folds": cv
            }
            
            logger.info(f"Hyperparameter optimization completed. Best parameters: {best_params}")
            
            return optimized_result
            
        except Exception as e:
            logger.error(f"Error during hyperparameter optimization: {e}")
            return {"status": "error", "message": str(e)}
    
    def evaluate_model(self, model_id: str, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Evaluate a trained model on new data.
        
        Args:
            model_id (str): Model identifier
            X (pd.DataFrame): Feature data for evaluation
            y (pd.Series): Target data for evaluation
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        model = self.load_model(model_id)
        
        if model is None:
            return {"status": "error", "message": f"Model {model_id} not found"}
        
        if X.empty or len(y) == 0:
            logger.error("Empty evaluation data provided")
            return {"status": "error", "message": "Empty evaluation data"}
        
        try:
            # Get the problem type based on the model
            if hasattr(model, 'predict_proba'):
                problem_type = "classification"
            else:
                problem_type = "regression"
            
            # Make predictions
            y_pred = model.predict(X)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y, y_pred, problem_type)
            
            logger.info(f"Model {model_id} evaluation complete with metrics: {metrics}")
            
            return {
                "status": "success",
                "model_id": model_id,
                "evaluation_metrics": metrics,
                "samples": len(X)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_id}: {e}")
            return {"status": "error", "message": str(e)}
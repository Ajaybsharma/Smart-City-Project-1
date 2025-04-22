import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple, Callable
import logging
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AIOptimizer:
    """Applies machine learning to optimize sustainability."""
    
    def __init__(self, model_dir: str = "optimization_models"):
        """
        Initialize the AIOptimizer.
        
        Args:
            model_dir (str): Directory to save optimization models
        """
        self.model_dir = model_dir
        self.models = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        logger.info(f"Initialized AIOptimizer with model directory: {self.model_dir}")
    
    def train_optimization_model(self, data: pd.DataFrame, target_column: str, 
                                feature_columns: List[str] = None, 
                                model_type: str = "random_forest",
                                model_name: str = "optimization_model") -> Dict[str, Any]:
        """
        Train a machine learning model for optimization.
        
        Args:
            data (pd.DataFrame): Training data
            target_column (str): Column containing the target variable to optimize
            feature_columns (List[str], optional): Columns to use as features. If None, uses all columns except target.
            model_type (str): Type of model to train ('random_forest', 'gradient_boosting', etc.)
            model_name (str): Name for the saved model
            
        Returns:
            Dict[str, Any]: Training results and metrics
        """
        if data.empty:
            logger.error("Empty dataset provided for optimization model training")
            return {"status": "error", "message": "Empty dataset"}
        
        if target_column not in data.columns:
            logger.error(f"Target column '{target_column}' not found in data")
            return {"status": "error", "message": f"Column not found: {target_column}"}
        
        logger.info(f"Training optimization model for target: {target_column}")
        
        # If feature columns not specified, use all columns except target
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        else:
            # Check if all feature columns exist
            missing_columns = [col for col in feature_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"Feature columns not found: {', '.join(missing_columns)}")
                return {"status": "error", "message": f"Columns not found: {', '.join(missing_columns)}"}
        
        try:
            # Prepare data
            X = data[feature_columns]
            y = data[target_column]
            
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create and train the model
            if model_type == "random_forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                logger.warning(f"Unknown model type: {model_type}. Using RandomForestRegressor instead.")
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X_train, y_train)
            
            # Evaluate the model
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Get feature importances
            feature_importances = {
                feature: float(importance)
                for feature, importance in zip(feature_columns, model.feature_importances_)
            }
            
            # Save the model
            model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
            joblib.dump(model, model_path)
            
            # Save model in memory
            self.models[model_name] = {
                "model": model,
                "feature_columns": feature_columns,
                "target_column": target_column
            }
            
            logger.info(f"Successfully trained and saved optimization model '{model_name}'")
            
            return {
                "status": "success",
                "model_name": model_name,
                "model_type": model_type,
                "target_column": target_column,
                "feature_columns": feature_columns,
                "train_score": float(train_score),
                "test_score": float(test_score),
                "feature_importances": feature_importances,
                "model_path": model_path
            }
            
        except Exception as e:
            logger.error(f"Error training optimization model: {e}")
            return {"status": "error", "message": str(e)}
    
    def optimize(self, model_name: str, constraints: Dict[str, Any], 
               optimization_goal: str = "maximize", 
               num_simulations: int = 1000) -> Dict[str, Any]:
        """
        Find optimal parameters to maximize or minimize a target variable.
        
        Args:
            model_name (str): Name of the trained optimization model
            constraints (Dict[str, Any]): Constraints for each feature (min, max, fixed values)
            optimization_goal (str): Whether to 'maximize' or 'minimize' the target
            num_simulations (int): Number of simulations to run
            
        Returns:
            Dict[str, Any]: Optimization results
        """
        # Load model
        if model_name in self.models:
            model_info = self.models[model_name]
            model = model_info["model"]
            feature_columns = model_info["feature_columns"]
            target_column = model_info["target_column"]
        else:
            model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
            if not os.path.exists(model_path):
                logger.error(f"Optimization model '{model_name}' not found")
                return {"status": "error", "message": f"Model not found: {model_name}"}
            
            try:
                model = joblib.load(model_path)
                
                # We need feature columns, but if not in memory, need to infer from model
                if hasattr(model, 'feature_names_in_'):
                    feature_columns = model.feature_names_in_.tolist()
                else:
                    logger.error("Loaded model does not contain feature column information")
                    return {"status": "error", "message": "Model missing feature information"}
                
                # Assume target column is unknown if not in memory
                target_column = "unknown_target"
                
            except Exception as e:
                logger.error(f"Error loading optimization model: {e}")
                return {"status": "error", "message": f"Error loading model: {e}"}
        
        logger.info(f"Running optimization using model '{model_name}' with {num_simulations} simulations")
        
        try:
            # Validate constraints against feature columns
            for feature in constraints:
                if feature not in feature_columns:
                    logger.warning(f"Constraint provided for unknown feature: {feature}")
            
            # Generate random samples within constraints
            simulation_data = {}
            
            for feature in feature_columns:
                if feature in constraints:
                    constraint = constraints[feature]
                    
                    if "fixed" in constraint:
                        # Use fixed value
                        simulation_data[feature] = np.full(num_simulations, constraint["fixed"])
                    else:
                        # Use random values within min/max range
                        min_val = constraint.get("min", 0)
                        max_val = constraint.get("max", 1)
                        
                        if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                            if isinstance(min_val, int) and isinstance(max_val, int):
                                # For integers, use randint
                                simulation_data[feature] = np.random.randint(min_val, max_val + 1, size=num_simulations)
                            else:
                                # For floats, use uniform
                                simulation_data[feature] = np.random.uniform(min_val, max_val, size=num_simulations)
                        else:
                            logger.warning(f"Invalid min/max values for feature {feature}")
                            simulation_data[feature] = np.zeros(num_simulations)
                else:
                    # No constraints, use reasonable defaults
                    # In a real system, these would be based on the feature's distribution in the training data
                    simulation_data[feature] = np.random.uniform(0, 1, size=num_simulations)
            
            # Create DataFrame from simulation data
            simulation_df = pd.DataFrame(simulation_data)
            
            # Make predictions
            predictions = model.predict(simulation_df)
            
            # Find optimal result
            if optimization_goal == "maximize":
                best_idx = np.argmax(predictions)
            else:  # minimize
                best_idx = np.argmin(predictions)
            
            best_params = {
                feature: float(simulation_df.iloc[best_idx][feature])
                for feature in feature_columns
            }
            
            best_value = float(predictions[best_idx])
            
            # Get top 5 results
            if optimization_goal == "maximize":
                top_indices = np.argsort(predictions)[-5:][::-1]
            else:  # minimize
                top_indices = np.argsort(predictions)[:5]
            
            top_results = [
                {
                    "parameters": {
                        feature: float(simulation_df.iloc[idx][feature])
                        for feature in feature_columns
                    },
                    "predicted_value": float(predictions[idx])
                }
                for idx in top_indices
            ]
            
            # Additional analysis
            prediction_range = {
                "min": float(np.min(predictions)),
                "max": float(np.max(predictions)),
                "mean": float(np.mean(predictions)),
                "median": float(np.median(predictions)),
                "std": float(np.std(predictions))
            }
            
            # Calculate improvement percentage compared to average
            improvement_pct = ((best_value - prediction_range["mean"]) / prediction_range["mean"]) * 100
            if optimization_goal == "minimize":
                improvement_pct = -improvement_pct
            
            logger.info(f"Optimization completed. Best {target_column}: {best_value}")
            
            return {
                "status": "success",
                "optimization_goal": optimization_goal,
                "target_column": target_column,
                "best_parameters": best_params,
                "best_value": best_value,
                "top_results": top_results,
                "prediction_range": prediction_range,
                "improvement_percentage": float(improvement_pct),
                "simulation_count": num_simulations
            }
            
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            return {"status": "error", "message": str(e)}
    
    def sensitivity_analysis(self, model_name: str, baseline_values: Dict[str, float], 
                           feature_ranges: Dict[str, Tuple[float, float]] = None,
                           steps: int = 10) -> Dict[str, Any]:
        """
        Perform sensitivity analysis to see how changes in input features affect the target.
        
        Args:
            model_name (str): Name of the trained optimization model
            baseline_values (Dict[str, float]): Baseline values for all features
            feature_ranges (Dict[str, Tuple[float, float]], optional): Min/max ranges for features to analyze
            steps (int): Number of steps for each feature range
            
        Returns:
            Dict[str, Any]: Sensitivity analysis results
        """
        # Load model
        if model_name in self.models:
            model_info = self.models[model_name]
            model = model_info["model"]
            feature_columns = model_info["feature_columns"]
            target_column = model_info["target_column"]
        else:
            model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
            if not os.path.exists(model_path):
                logger.error(f"Optimization model '{model_name}' not found")
                return {"status": "error", "message": f"Model not found: {model_name}"}
            
            try:
                model = joblib.load(model_path)
                
                if hasattr(model, 'feature_names_in_'):
                    feature_columns = model.feature_names_in_.tolist()
                else:
                    logger.error("Loaded model does not contain feature column information")
                    return {"status": "error", "message": "Model missing feature information"}
                
                target_column = "unknown_target"
                
            except Exception as e:
                logger.error(f"Error loading optimization model: {e}")
                return {"status": "error", "message": f"Error loading model: {e}"}
        
        logger.info(f"Running sensitivity analysis using model '{model_name}'")
        
        try:
            # Validate baseline values
            missing_baselines = [f for f in feature_columns if f not in baseline_values]
            if missing_baselines:
                logger.error(f"Missing baseline values for features: {', '.join(missing_baselines)}")
                return {"status": "error", "message": f"Missing baseline values for features: {', '.join(missing_baselines)}"}
            
            # Use all features if no specific ranges provided
            if feature_ranges is None:
                feature_ranges = {}
                for feature in feature_columns:
                    # Default range is ±20% from baseline
                    baseline = baseline_values[feature]
                    feature_ranges[feature] = (baseline * 0.8, baseline * 1.2)
            
            # Predict baseline
            baseline_df = pd.DataFrame({feature: [baseline_values[feature]] for feature in feature_columns})
            baseline_prediction = float(model.predict(baseline_df)[0])
            
            # Analyze each feature's sensitivity
            sensitivity_results = {}
            
            for feature in feature_columns:
                if feature in feature_ranges:
                    min_val, max_val = feature_ranges[feature]
                    
                    # Create range of values
                    feature_values = np.linspace(min_val, max_val, steps)
                    
                    # Create test dataset with varying feature
                    test_data = {}
                    for f in feature_columns:
                        if f == feature:
                            test_data[f] = feature_values
                        else:
                            test_data[f] = np.full(steps, baseline_values[f])
                    
                    test_df = pd.DataFrame(test_data)
                    
                    # Make predictions
                    predictions = model.predict(test_df)
                    
                    # Calculate sensitivity metrics
                    sensitivity = {}
                    sensitivity["feature_values"] = feature_values.tolist()
                    sensitivity["predictions"] = predictions.tolist()
                    
                    # Calculate percent change in prediction for percent change in feature
                    baseline_feature_val = baseline_values[feature]
                    if baseline_feature_val != 0:
                        feature_pct_changes = [(val - baseline_feature_val) / baseline_feature_val * 100 for val in feature_values]
                    else:
                        feature_pct_changes = [0] + [(val - min_val) / (max_val - min_val) * 100 for val in feature_values[1:]]
                    
                    if baseline_prediction != 0:
                        prediction_pct_changes = [(pred - baseline_prediction) / baseline_prediction * 100 for pred in predictions]
                    else:
                        prediction_pct_changes = [0] * len(predictions)
                    
                    # Calculate elasticity (% change in output / % change in input)
                    elasticities = []
                    for i in range(len(feature_pct_changes)):
                        if feature_pct_changes[i] != 0:
                            elasticity = prediction_pct_changes[i] / feature_pct_changes[i]
                        else:
                            elasticity = 0
                        elasticities.append(elasticity)
                    
                    # Calculate average elasticity
                    avg_elasticity = np.mean([abs(e) for e in elasticities if not np.isnan(e) and not np.isinf(e)])
                    
                    sensitivity["elasticity"] = float(avg_elasticity)
                    sensitivity["max_impact"] = float(max(abs(p - baseline_prediction) for p in predictions))
                    sensitivity["feature_range"] = [float(min_val), float(max_val)]
                    
                    sensitivity_results[feature] = sensitivity
            
            # Rank features by sensitivity
            ranked_features = sorted(
                sensitivity_results.items(),
                key=lambda x: x[1]["elasticity"],
                reverse=True
            )
            
            ranked_result = {
                feature: {
                    "elasticity": sensitivity_results[feature]["elasticity"],
                    "max_impact": sensitivity_results[feature]["max_impact"]
                }
                for feature, _ in ranked_features
            }
            
            logger.info(f"Sensitivity analysis completed for {len(sensitivity_results)} features")
            
            return {
                "status": "success",
                "baseline_values": baseline_values,
                "baseline_prediction": baseline_prediction,
                "sensitivity_by_feature": sensitivity_results,
                "ranked_sensitivity": ranked_result
            }
            
        except Exception as e:
            logger.error(f"Error during sensitivity analysis: {e}")
            return {"status": "error", "message": str(e)}
    
    def scenario_analysis(self, model_name: str, scenarios: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Analyze different scenarios to predict outcomes.
        
        Args:
            model_name (str): Name of the trained optimization model
            scenarios (Dict[str, Dict[str, float]]): Dictionary of named scenarios with feature values
            
        Returns:
            Dict[str, Any]: Scenario analysis results
        """
        # Load model
        if model_name in self.models:
            model_info = self.models[model_name]
            model = model_info["model"]
            feature_columns = model_info["feature_columns"]
            target_column = model_info["target_column"]
        else:
            model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
            if not os.path.exists(model_path):
                logger.error(f"Optimization model '{model_name}' not found")
                return {"status": "error", "message": f"Model not found: {model_name}"}
            
            try:
                model = joblib.load(model_path)
                
                if hasattr(model, 'feature_names_in_'):
                    feature_columns = model.feature_names_in_.tolist()
                else:
                    logger.error("Loaded model does not contain feature column information")
                    return {"status": "error", "message": "Model missing feature information"}
                
                target_column = "unknown_target"
                
            except Exception as e:
                logger.error(f"Error loading optimization model: {e}")
                return {"status": "error", "message": f"Error loading model: {e}"}
        
        logger.info(f"Running scenario analysis for {len(scenarios)} scenarios using model '{model_name}'")
        
        try:
            scenario_results = {}
            baseline_scenario = None
            
            for scenario_name, scenario_values in scenarios.items():
                # Check if this scenario contains all required features
                missing_features = [f for f in feature_columns if f not in scenario_values]
                if missing_features:
                    logger.warning(f"Scenario '{scenario_name}' is missing features: {', '.join(missing_features)}")
                    continue
                
                # Create DataFrame for prediction
                scenario_df = pd.DataFrame({feature: [scenario_values[feature]] for feature in feature_columns})
                
                # Make prediction
                prediction = float(model.predict(scenario_df)[0])
                
                # Store result
                scenario_results[scenario_name] = {
                    "parameters": {f: float(v) for f, v in scenario_values.items()},
                    "prediction": prediction
                }
                
                # Identify baseline scenario if it exists
                if scenario_name.lower() == "baseline" or scenario_name.lower() == "current":
                    baseline_scenario = scenario_name
            
            # Calculate comparisons if baseline exists
            comparisons = {}
            if baseline_scenario and baseline_scenario in scenario_results:
                baseline_prediction = scenario_results[baseline_scenario]["prediction"]
                
                for scenario_name, result in scenario_results.items():
                    if scenario_name != baseline_scenario:
                        prediction = result["prediction"]
                        absolute_change = prediction - baseline_prediction
                        
                        percent_change = 0
                        if baseline_prediction != 0:
                            percent_change = (absolute_change / baseline_prediction) * 100
                        
                        comparisons[scenario_name] = {
                            "absolute_change": float(absolute_change),
                            "percent_change": float(percent_change),
                            "baseline": baseline_scenario
                        }
            
            # Rank scenarios
            ranked_scenarios = sorted(
                scenario_results.items(),
                key=lambda x: x[1]["prediction"],
                reverse=True
            )
            
            ranked_result = {
                scenario: scenario_results[scenario]["prediction"]
                for scenario, _ in ranked_scenarios
            }
            
            logger.info(f"Scenario analysis completed for {len(scenario_results)} scenarios")
            
            return {
                "status": "success",
                "target_column": target_column,
                "scenarios": scenario_results,
                "comparisons": comparisons,
                "ranked_scenarios": ranked_result
            }
            
        except Exception as e:
            logger.error(f"Error during scenario analysis: {e}")
            return {"status": "error", "message": str(e)}
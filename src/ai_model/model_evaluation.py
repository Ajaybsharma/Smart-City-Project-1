import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple, Callable
import logging
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Assesses model performance with various metrics."""
    
    def __init__(self, evaluation_dir: str = "evaluations"):
        """
        Initialize the ModelEvaluator.
        
        Args:
            evaluation_dir (str): Directory to save evaluation results
        """
        self.evaluation_dir = evaluation_dir
        
        # Create evaluation directory if it doesn't exist
        os.makedirs(self.evaluation_dir, exist_ok=True)
        
        logger.info(f"Initialized ModelEvaluator with evaluation directory: {self.evaluation_dir}")
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, problem_type: str = "regression",
                model_name: str = "model", detailed: bool = False) -> Dict[str, Any]:
        """
        Evaluate model predictions.
        
        Args:
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted values
            problem_type (str): Type of problem ('regression' or 'classification')
            model_name (str): Name of the model for reporting
            detailed (bool): Whether to include detailed metrics
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            logger.error("Empty arrays provided for evaluation")
            return {"status": "error", "message": "Empty evaluation data"}
        
        evaluation_id = f"{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Calculate metrics based on problem type
            if problem_type == "regression":
                metrics = self._evaluate_regression(y_true, y_pred, detailed)
            elif problem_type == "classification":
                # For classification, we might need class probabilities for some metrics
                metrics = self._evaluate_classification(y_true, y_pred, detailed)
            else:
                logger.warning(f"Unknown problem type: {problem_type}")
                return {"status": "error", "message": f"Unknown problem type: {problem_type}"}
            
            # Prepare evaluation results
            evaluation_results = {
                "evaluation_id": evaluation_id,
                "model_name": model_name,
                "problem_type": problem_type,
                "metrics": metrics,
                "sample_count": len(y_true),
                "evaluation_time": datetime.datetime.now().isoformat()
            }
            
            # Save evaluation results
            self._save_evaluation(evaluation_results, evaluation_id)
            
            logger.info(f"Completed evaluation {evaluation_id} for {model_name}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {"status": "error", "message": str(e)}
    
    def _evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray, detailed: bool = False) -> Dict[str, Any]:
        """
        Evaluate regression model.
        
        Args:
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted values
            detailed (bool): Whether to include detailed metrics
            
        Returns:
            Dict[str, Any]: Regression metrics
        """
        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
            "mae": float(mae)
        }
        
        # Additional metrics for detailed evaluation
        if detailed:
            # Calculate MAPE (Mean Absolute Percentage Error)
            # Avoid division by zero by adding small epsilon
            epsilon = 1e-10
            mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
            
            # Calculate explained variance
            explained_variance = 1 - np.var(y_true - y_pred) / np.var(y_true)
            
            # Add detailed metrics
            metrics.update({
                "mape": float(mape),
                "explained_variance": float(explained_variance),
                "median_absolute_error": float(np.median(np.abs(y_true - y_pred))),
                "max_error": float(np.max(np.abs(y_true - y_pred)))
            })
            
            # Residual analysis
            residuals = y_true - y_pred
            metrics["residuals"] = {
                "mean": float(np.mean(residuals)),
                "std": float(np.std(residuals)),
                "min": float(np.min(residuals)),
                "25%": float(np.percentile(residuals, 25)),
                "50%": float(np.median(residuals)),
                "75%": float(np.percentile(residuals, 75)),
                "max": float(np.max(residuals)),
                "distribution": "normal" if self._is_normal(residuals) else "non-normal"
            }
        
        return metrics
    
    def _evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               detailed: bool = False, y_prob: np.ndarray = None) -> Dict[str, Any]:
        """
        Evaluate classification model.
        
        Args:
            y_true (np.ndarray): True class labels
            y_pred (np.ndarray): Predicted class labels
            detailed (bool): Whether to include detailed metrics
            y_prob (np.ndarray, optional): Predicted class probabilities for ROC analysis
            
        Returns:
            Dict[str, Any]: Classification metrics
        """
        # Get number of classes
        classes = np.unique(np.concatenate((y_true, y_pred)))
        n_classes = len(classes)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # For multi-class, we need to specify averaging method
        if n_classes > 2:
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
        else:
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
        
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }
        
        # Additional metrics for detailed evaluation
        if detailed:
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Convert confusion matrix to list for JSON serialization
            cm_list = cm.tolist()
            
            # Class-wise metrics
            class_report = classification_report(y_true, y_pred, output_dict=True)
            
            # Add metrics to the results
            metrics.update({
                "confusion_matrix": cm_list,
                "class_report": class_report
            })
            
            # If probabilities are provided, calculate ROC AUC
            if y_prob is not None and n_classes == 2:
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                roc_auc = auc(fpr, tpr)
                metrics["roc_auc"] = float(roc_auc)
                
                # Precision-Recall curve
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob[:, 1])
                avg_precision = average_precision_score(y_true, y_prob[:, 1])
                metrics["average_precision"] = float(avg_precision)
        
        return metrics
    
    def _is_normal(self, data: np.ndarray, alpha: float = 0.05) -> bool:
        """
        Check if data follows a normal distribution using Shapiro-Wilk test.
        
        Args:
            data (np.ndarray): Data to test
            alpha (float): Significance level
            
        Returns:
            bool: True if data appears to be normally distributed
        """
        from scipy import stats
        
        # Sample data if there are too many points (Shapiro-Wilk limited to 5000 samples)
        if len(data) > 5000:
            data = np.random.choice(data, size=5000, replace=False)
        
        _, p_value = stats.shapiro(data)
        return p_value > alpha
    
    def _save_evaluation(self, evaluation_results: Dict[str, Any], evaluation_id: str) -> str:
        """
        Save evaluation results to file.
        
        Args:
            evaluation_results (Dict[str, Any]): Evaluation results
            evaluation_id (str): Unique identifier for the evaluation
            
        Returns:
            str: Path to the saved file
        """
        file_path = os.path.join(self.evaluation_dir, f"{evaluation_id}.json")
        
        try:
            with open(file_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
                
            logger.info(f"Saved evaluation results to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
            return ""
    
    def compare_models(self, evaluations: List[str], output_format: str = "text") -> Dict[str, Any]:
        """
        Compare multiple model evaluations.
        
        Args:
            evaluations (List[str]): List of evaluation IDs to compare
            output_format (str): Format for comparison ('text', 'json', 'html')
            
        Returns:
            Dict[str, Any]: Comparison results
        """
        if not evaluations:
            logger.error("No evaluations provided for comparison")
            return {"status": "error", "message": "No evaluations to compare"}
        
        comparison_id = f"comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Load evaluation results
            evaluation_data = []
            
            for eval_id in evaluations:
                file_path = os.path.join(self.evaluation_dir, f"{eval_id}.json")
                
                if not os.path.exists(file_path):
                    logger.warning(f"Evaluation file not found: {file_path}")
                    continue
                
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    evaluation_data.append(data)
            
            if not evaluation_data:
                logger.error("No valid evaluations found for comparison")
                return {"status": "error", "message": "No valid evaluations to compare"}
            
            # Extract metrics for comparison
            comparison_table = self._create_comparison_table(evaluation_data)
            
            # Format the comparison result based on output format
            if output_format == "text":
                comparison_result = self._format_comparison_text(comparison_table)
            elif output_format == "html":
                comparison_result = self._format_comparison_html(comparison_table)
            else:  # json
                comparison_result = comparison_table
            
            # Prepare and save the comparison result
            comparison_data = {
                "comparison_id": comparison_id,
                "evaluations": evaluations,
                "comparison_table": comparison_table,
                "comparison_result": comparison_result,
                "comparison_time": datetime.datetime.now().isoformat()
            }
            
            # Save comparison result
            comparison_file = os.path.join(self.evaluation_dir, f"{comparison_id}.json")
            with open(comparison_file, 'w') as f:
                json.dump(comparison_data, f, indent=2)
            
            logger.info(f"Completed model comparison {comparison_id}")
            
            return comparison_data
            
        except Exception as e:
            logger.error(f"Error during model comparison: {e}")
            return {"status": "error", "message": str(e)}
    
    def _create_comparison_table(self, evaluation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a table comparing metrics from different evaluations.
        
        Args:
            evaluation_data (List[Dict[str, Any]]): List of evaluation results
            
        Returns:
            Dict[str, Any]: Comparison table
        """
        comparison = {
            "models": [],
            "problem_type": [],
            "sample_count": [],
            "metrics": {}
        }
        
        # Extract common metrics
        all_metrics = set()
        for data in evaluation_data:
            comparison["models"].append(data["model_name"])
            comparison["problem_type"].append(data["problem_type"])
            comparison["sample_count"].append(data["sample_count"])
            
            for metric in data["metrics"]:
                if isinstance(data["metrics"][metric], (int, float)):
                    all_metrics.add(metric)
        
        # Initialize metric lists
        for metric in all_metrics:
            comparison["metrics"][metric] = []
        
        # Fill in metric values
        for data in evaluation_data:
            for metric in all_metrics:
                value = data["metrics"].get(metric, "N/A")
                comparison["metrics"][metric].append(value)
        
        return comparison
    
    def _format_comparison_text(self, comparison_table: Dict[str, Any]) -> str:
        """
        Format comparison table as text.
        
        Args:
            comparison_table (Dict[str, Any]): Comparison table data
            
        Returns:
            str: Formatted text
        """
        lines = ["Model Comparison"]
        lines.append("=" * 80)
        
        # Header row with model names
        header = "Metric".ljust(20)
        for model in comparison_table["models"]:
            header += model.ljust(15)
        lines.append(header)
        lines.append("-" * 80)
        
        # Problem type row
        prob_type_row = "Problem Type".ljust(20)
        for pt in comparison_table["problem_type"]:
            prob_type_row += pt.ljust(15)
        lines.append(prob_type_row)
        
        # Sample count row
        sample_row = "Sample Count".ljust(20)
        for count in comparison_table["sample_count"]:
            sample_row += str(count).ljust(15)
        lines.append(sample_row)
        lines.append("-" * 80)
        
        # Metrics rows
        for metric, values in comparison_table["metrics"].items():
            metric_row = metric.ljust(20)
            for value in values:
                if isinstance(value, float):
                    metric_row += f"{value:.4f}".ljust(15)
                else:
                    metric_row += str(value).ljust(15)
            lines.append(metric_row)
        
        return "\n".join(lines)
    
    def _format_comparison_html(self, comparison_table: Dict[str, Any]) -> str:
        """
        Format comparison table as HTML.
        
        Args:
            comparison_table (Dict[str, Any]): Comparison table data
            
        Returns:
            str: HTML formatted comparison
        """
        html = ["<table border='1'>"]
        
        # Header row
        html.append("<tr><th>Metric</th>")
        for model in comparison_table["models"]:
            html.append(f"<th>{model}</th>")
        html.append("</tr>")
        
        # Problem type row
        html.append("<tr><td>Problem Type</td>")
        for pt in comparison_table["problem_type"]:
            html.append(f"<td>{pt}</td>")
        html.append("</tr>")
        
        # Sample count row
        html.append("<tr><td>Sample Count</td>")
        for count in comparison_table["sample_count"]:
            html.append(f"<td>{count}</td>")
        html.append("</tr>")
        
        # Metrics rows
        for metric, values in comparison_table["metrics"].items():
            html.append(f"<tr><td>{metric}</td>")
            for value in values:
                if isinstance(value, float):
                    html.append(f"<td>{value:.4f}</td>")
                else:
                    html.append(f"<td>{value}</td>")
            html.append("</tr>")
        
        html.append("</table>")
        
        return "".join(html)
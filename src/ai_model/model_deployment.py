import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple, Callable
import logging
import joblib
import os
import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelDeployer:
    """Fine-tunes model parameters and prepares models for deployment."""
    
    def __init__(self, models_dir: str = "models", deployments_dir: str = "deployments"):
        """
        Initialize the ModelDeployer.
        
        Args:
            models_dir (str): Directory containing trained models
            deployments_dir (str): Directory for deployed models
        """
        self.models_dir = models_dir
        self.deployments_dir = deployments_dir
        self.deployed_models = {}
        
        # Create deployment directory if it doesn't exist
        os.makedirs(self.deployments_dir, exist_ok=True)
        
        logger.info(f"Initialized ModelDeployer with deployments directory: {self.deployments_dir}")
    
    def deploy_model(self, model_id: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Deploy a model for production use.
        
        Args:
            model_id (str): ID of the model to deploy
            metadata (Dict[str, Any], optional): Additional metadata for the deployment
            
        Returns:
            Dict[str, Any]: Deployment information
        """
        # Load the model
        model_path = os.path.join(self.models_dir, f"{model_id}.joblib")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return {"status": "error", "message": f"Model {model_id} not found"}
        
        try:
            # Load the model
            model = joblib.load(model_path)
            
            # Create a deployment ID
            deployment_id = f"deployment_{model_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create deployment metadata
            deployment_info = {
                "deployment_id": deployment_id,
                "model_id": model_id,
                "deployment_time": datetime.datetime.now().isoformat(),
                "model_type": model.__class__.__name__,
                "model_parameters": model.get_params(),
                "metadata": metadata or {}
            }
            
            # Save deployment information
            deployment_info_path = os.path.join(self.deployments_dir, f"{deployment_id}_info.json")
            with open(deployment_info_path, 'w') as f:
                json.dump(deployment_info, f, indent=2)
            
            # Copy the model to deployments directory
            deployment_model_path = os.path.join(self.deployments_dir, f"{deployment_id}.joblib")
            joblib.dump(model, deployment_model_path)
            
            # Add to deployed models
            self.deployed_models[deployment_id] = {
                "model": model,
                "info": deployment_info
            }
            
            logger.info(f"Successfully deployed model {model_id} as {deployment_id}")
            
            return {
                "status": "success",
                "deployment_id": deployment_id,
                "model_id": model_id,
                "deployment_time": deployment_info["deployment_time"],
                "info_path": deployment_info_path,
                "model_path": deployment_model_path
            }
            
        except Exception as e:
            logger.error(f"Error deploying model {model_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    def list_deployments(self) -> List[Dict[str, Any]]:
        """
        List all deployed models.
        
        Returns:
            List[Dict[str, Any]]: List of deployment information
        """
        deployments = []
        
        try:
            for filename in os.listdir(self.deployments_dir):
                if filename.endswith("_info.json"):
                    with open(os.path.join(self.deployments_dir, filename), 'r') as f:
                        deployment_info = json.load(f)
                    
                    deployments.append(deployment_info)
            
            return deployments
            
        except Exception as e:
            logger.error(f"Error listing deployments: {e}")
            return []
    
    def get_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """
        Get information about a specific deployment.
        
        Args:
            deployment_id (str): Deployment identifier
            
        Returns:
            Dict[str, Any]: Deployment information
        """
        if deployment_id in self.deployed_models:
            return self.deployed_models[deployment_id]["info"]
        
        try:
            info_path = os.path.join(self.deployments_dir, f"{deployment_id}_info.json")
            
            if not os.path.exists(info_path):
                logger.error(f"Deployment info not found: {info_path}")
                return {"status": "error", "message": f"Deployment {deployment_id} not found"}
            
            with open(info_path, 'r') as f:
                deployment_info = json.load(f)
            
            return deployment_info
            
        except Exception as e:
            logger.error(f"Error getting deployment info for {deployment_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    def load_deployed_model(self, deployment_id: str) -> Any:
        """
        Load a deployed model by ID.
        
        Args:
            deployment_id (str): Deployment identifier
            
        Returns:
            object: Loaded model
        """
        if deployment_id in self.deployed_models:
            return self.deployed_models[deployment_id]["model"]
        
        try:
            model_path = os.path.join(self.deployments_dir, f"{deployment_id}.joblib")
            
            if not os.path.exists(model_path):
                logger.error(f"Deployed model file not found: {model_path}")
                return None
            
            model = joblib.load(model_path)
            
            # Load deployment info
            info_path = os.path.join(self.deployments_dir, f"{deployment_id}_info.json")
            with open(info_path, 'r') as f:
                deployment_info = json.load(f)
            
            # Cache the model and info
            self.deployed_models[deployment_id] = {
                "model": model,
                "info": deployment_info
            }
            
            logger.info(f"Successfully loaded deployed model {deployment_id}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading deployed model {deployment_id}: {e}")
            return None
    
    def predict(self, deployment_id: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions using a deployed model.
        
        Args:
            deployment_id (str): Deployment identifier
            data (pd.DataFrame): Data for predictions
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        model = self.load_deployed_model(deployment_id)
        
        if model is None:
            return {"status": "error", "message": f"Deployed model {deployment_id} not found"}
        
        if data.empty:
            logger.error("Empty data provided for prediction")
            return {"status": "error", "message": "Empty data for prediction"}
        
        try:
            # Make predictions
            predictions = model.predict(data)
            
            # Check if the model has probability prediction capability
            has_probabilities = hasattr(model, 'predict_proba')
            
            result = {
                "status": "success",
                "deployment_id": deployment_id,
                "sample_count": len(data),
                "predictions": predictions.tolist(),
                "prediction_time": datetime.datetime.now().isoformat()
            }
            
            # Add probabilities if available
            if has_probabilities:
                probabilities = model.predict_proba(data)
                result["probabilities"] = probabilities.tolist()
            
            logger.info(f"Successfully made predictions with model {deployment_id} on {len(data)} samples")
            
            return result
            
        except Exception as e:
            logger.error(f"Error making predictions with model {deployment_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    def update_deployment_metadata(self, deployment_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update metadata for a deployed model.
        
        Args:
            deployment_id (str): Deployment identifier
            metadata (Dict[str, Any]): New metadata to update
            
        Returns:
            Dict[str, Any]: Updated deployment information
        """
        try:
            info_path = os.path.join(self.deployments_dir, f"{deployment_id}_info.json")
            
            if not os.path.exists(info_path):
                logger.error(f"Deployment info not found: {info_path}")
                return {"status": "error", "message": f"Deployment {deployment_id} not found"}
            
            # Load existing info
            with open(info_path, 'r') as f:
                deployment_info = json.load(f)
            
            # Update metadata
            if "metadata" not in deployment_info:
                deployment_info["metadata"] = {}
            
            deployment_info["metadata"].update(metadata)
            deployment_info["last_updated"] = datetime.datetime.now().isoformat()
            
            # Save updated info
            with open(info_path, 'w') as f:
                json.dump(deployment_info, f, indent=2)
            
            # Update cached info if exists
            if deployment_id in self.deployed_models:
                self.deployed_models[deployment_id]["info"] = deployment_info
            
            logger.info(f"Updated metadata for deployment {deployment_id}")
            
            return deployment_info
            
        except Exception as e:
            logger.error(f"Error updating metadata for deployment {deployment_id}: {e}")
            return {"status": "error", "message": str(e)}
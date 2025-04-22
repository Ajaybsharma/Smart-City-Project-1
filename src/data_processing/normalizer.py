import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataNormalizer:
    """Standardizes data from diverse input sources."""
    
    def __init__(self):
        """Initialize the DataNormalizer."""
        self.scalers = {}
        logger.info("Initialized DataNormalizer")
    
    def normalize(self, df: pd.DataFrame, method: str = "standard", columns: List[str] = None, save_scaler: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Normalize data using specified method.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            method (str): Normalization method ('standard', 'minmax', 'robust', 'quantile')
            columns (List[str], optional): Specific columns to normalize. If None, all numeric columns are used.
            save_scaler (bool): Whether to save the fitted scaler for later use
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Normalized DataFrame and metadata
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for normalization")
            return df, {"status": "error", "message": "Empty DataFrame"}
        
        # Select columns to normalize
        if columns is None:
            # Use all numeric columns
            columns = df.select_dtypes(include=np.number).columns.tolist()
        else:
            # Filter to only include existing numeric columns
            columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if not columns:
            logger.warning("No numeric columns found for normalization")
            return df, {"status": "warning", "message": "No numeric columns to normalize"}
        
        logger.info(f"Normalizing {len(columns)} columns using {method} method")
        
        result_df = df.copy()
        metadata = {
            "method": method,
            "columns": columns,
            "stats": {}
        }
        
        # Create a unique key for this normalization
        scaler_key = f"{method}_{'_'.join(sorted(columns))}"
        
        try:
            # Initialize the appropriate scaler
            scaler = self._get_scaler(method)
            
            # Save original data information for interpretability
            for col in columns:
                metadata["stats"][col] = {
                    "original_mean": df[col].mean(),
                    "original_std": df[col].std(),
                    "original_min": df[col].min(),
                    "original_max": df[col].max(),
                    "original_median": df[col].median()
                }
            
            # Extract the data for scaling
            data_to_scale = df[columns].values
            
            # Check if we have a saved scaler
            if scaler_key in self.scalers:
                logger.info("Using existing scaler")
                scaler = self.scalers[scaler_key]
                scaled_data = scaler.transform(data_to_scale)
            else:
                # Fit and transform the data
                scaled_data = scaler.fit_transform(data_to_scale)
                
                # Save the scaler if requested
                if save_scaler:
                    self.scalers[scaler_key] = scaler
            
            # Update the DataFrame with normalized values
            for i, col in enumerate(columns):
                result_df[col] = scaled_data[:, i]
                
                # Update metadata with normalized stats
                metadata["stats"][col].update({
                    "normalized_mean": result_df[col].mean(),
                    "normalized_std": result_df[col].std(),
                    "normalized_min": result_df[col].min(),
                    "normalized_max": result_df[col].max()
                })
            
            metadata["status"] = "success"
            logger.info("Normalization completed successfully")
            
            return result_df, metadata
        
        except Exception as e:
            logger.error(f"Error during normalization: {e}")
            return df, {"status": "error", "message": str(e)}
    
    def _get_scaler(self, method: str):
        """
        Get the appropriate scaler based on the normalization method.
        
        Args:
            method (str): Normalization method
            
        Returns:
            object: Scaler object
        """
        if method == "standard":
            return StandardScaler()
        elif method == "minmax":
            return MinMaxScaler()
        elif method == "robust":
            return RobustScaler()
        elif method == "quantile":
            return QuantileTransformer(output_distribution='normal')
        else:
            logger.warning(f"Unknown normalization method: {method}. Using standard scaling.")
            return StandardScaler()
    
    def inverse_normalize(self, df: pd.DataFrame, method: str, columns: List[str]) -> pd.DataFrame:
        """
        Reverse the normalization to get back original scale data.
        
        Args:
            df (pd.DataFrame): Normalized DataFrame
            method (str): Normalization method used
            columns (List[str]): Columns to inverse normalize
            
        Returns:
            pd.DataFrame: DataFrame with original scale values
        """
        if df.empty:
            return df
        
        result_df = df.copy()
        
        # Create scaler key
        scaler_key = f"{method}_{'_'.join(sorted(columns))}"
        
        if scaler_key not in self.scalers:
            logger.warning(f"No saved scaler found for {scaler_key}")
            return df
        
        try:
            # Get the saved scaler
            scaler = self.scalers[scaler_key]
            
            # Extract normalized data
            normalized_data = df[columns].values
            
            # Inverse transform
            original_data = scaler.inverse_transform(normalized_data)
            
            # Update the DataFrame
            for i, col in enumerate(columns):
                result_df[col] = original_data[:, i]
            
            logger.info(f"Successfully inverse normalized {len(columns)} columns")
            
            return result_df
        
        except Exception as e:
            logger.error(f"Error during inverse normalization: {e}")
            return df
    
    def normalize_multiple_datasets(self, dataframes: List[pd.DataFrame], method: str = "standard", 
                                   columns: List[str] = None) -> List[pd.DataFrame]:
        """
        Normalize multiple datasets using the same scaling parameters.
        
        Args:
            dataframes (List[pd.DataFrame]): List of DataFrames to normalize
            method (str): Normalization method
            columns (List[str], optional): Columns to normalize
            
        Returns:
            List[pd.DataFrame]: List of normalized DataFrames
        """
        if not dataframes:
            logger.warning("No dataframes provided for normalization")
            return []
        
        # Use the first DataFrame to fit the scaler
        first_df, metadata = self.normalize(dataframes[0], method, columns)
        
        # Apply the same normalization to other DataFrames
        normalized_dfs = [first_df]
        
        for i, df in enumerate(dataframes[1:], 1):
            logger.info(f"Normalizing dataframe {i+1} using scaler from first dataframe")
            
            if columns is None:
                # Use the same columns as the first normalization
                norm_columns = metadata["columns"]
            else:
                norm_columns = columns
            
            # Filter to columns that exist in this DataFrame
            valid_columns = [col for col in norm_columns if col in df.columns]
            
            if not valid_columns:
                logger.warning(f"No valid columns found in dataframe {i+1}")
                normalized_dfs.append(df)
                continue
            
            # Create the scaler key used for the first DataFrame
            scaler_key = f"{method}_{'_'.join(sorted(norm_columns))}"
            
            if scaler_key not in self.scalers:
                logger.warning(f"No saved scaler found for {scaler_key}")
                normalized_dfs.append(df)
                continue
            
            try:
                # Get the scaler from the first normalization
                scaler = self.scalers[scaler_key]
                
                # Apply to this DataFrame
                result_df = df.copy()
                scaled_data = scaler.transform(df[valid_columns].values)
                
                for j, col in enumerate(valid_columns):
                    result_df[col] = scaled_data[:, j]
                
                normalized_dfs.append(result_df)
                
            except Exception as e:
                logger.error(f"Error normalizing dataframe {i+1}: {e}")
                normalized_dfs.append(df)
        
        return normalized_dfs
    
    def save_normalization_metadata(self) -> Dict[str, Any]:
        """
        Save metadata about all normalization operations.
        
        Returns:
            Dict[str, Any]: Dictionary of normalization metadata
        """
        metadata = {
            "saved_scalers": list(self.scalers.keys()),
            "normalization_methods": {},
            "scaler_details": {}
        }
        
        for scaler_key in self.scalers:
            parts = scaler_key.split('_')
            method = parts[0]
            columns = parts[1:]
            
            metadata["normalization_methods"][scaler_key] = method
            metadata["scaler_details"][scaler_key] = {
                "method": method,
                "columns": columns,
                "feature_count": len(columns)
            }
        
        return metadata
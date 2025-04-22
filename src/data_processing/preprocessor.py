import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Cleans and prepares raw data for further processing."""
    
    def __init__(self):
        """Initialize the DataPreprocessor."""
        logger.info("Initialized DataPreprocessor")
    
    def preprocess(self, df: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Apply preprocessing steps to a DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            config (Dict[str, Any], optional): Configuration for preprocessing steps
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for preprocessing")
            return df
        
        processed_df = df.copy()
        
        # Apply default preprocessing if no config is provided
        if config is None:
            logger.info("Using default preprocessing configuration")
            config = {
                "handle_missing": True,
                "remove_duplicates": True,
                "convert_dtypes": True
            }
        
        # Track preprocessing steps for logging
        preprocessing_steps = []
        
        # Handle missing values
        if config.get("handle_missing", False):
            processed_df, missing_info = self._handle_missing_values(
                processed_df, 
                strategy=config.get("missing_strategy", "mean")
            )
            preprocessing_steps.append(f"Handled missing values using {config.get('missing_strategy', 'mean')} strategy")
        
        # Remove duplicate rows
        if config.get("remove_duplicates", False):
            initial_rows = len(processed_df)
            processed_df = processed_df.drop_duplicates()
            removed_rows = initial_rows - len(processed_df)
            if removed_rows > 0:
                preprocessing_steps.append(f"Removed {removed_rows} duplicate rows")
        
        # Convert data types
        if config.get("convert_dtypes", False):
            processed_df = self._convert_dtypes(processed_df)
            preprocessing_steps.append("Converted data types")
        
        # Handle outliers
        if config.get("handle_outliers", False):
            outlier_columns = config.get("outlier_columns", processed_df.select_dtypes(include=np.number).columns)
            processed_df = self._handle_outliers(
                processed_df, 
                columns=outlier_columns,
                method=config.get("outlier_method", "iqr")
            )
            preprocessing_steps.append(f"Handled outliers in {len(outlier_columns)} columns using {config.get('outlier_method', 'iqr')} method")
        
        # Apply custom transformations
        if config.get("custom_transformations", False):
            transformations = config.get("transformations", {})
            processed_df = self._apply_transformations(processed_df, transformations)
            preprocessing_steps.append(f"Applied {len(transformations)} custom transformations")
        
        # Log preprocessing summary
        logger.info(f"Preprocessing completed: {'; '.join(preprocessing_steps)}")
        
        return processed_df
    
    def _handle_missing_values(self, df: pd.DataFrame, strategy: str = "mean") -> tuple:
        """
        Handle missing values in the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop', 'fill_zero')
            
        Returns:
            tuple: (processed DataFrame, missing values info)
        """
        missing_info = {
            "total_missing": df.isnull().sum().sum(),
            "missing_by_column": df.isnull().sum().to_dict()
        }
        
        if missing_info["total_missing"] == 0:
            return df, missing_info
        
        logger.info(f"Handling {missing_info['total_missing']} missing values using {strategy} strategy")
        
        if strategy == "drop":
            # Drop rows with any missing values
            return df.dropna(), missing_info
        
        # Process numerical and categorical columns separately
        num_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(exclude=np.number).columns
        
        result_df = df.copy()
        
        # Handle numerical columns
        for col in num_cols:
            if df[col].isnull().any():
                if strategy == "mean":
                    fill_value = df[col].mean()
                elif strategy == "median":
                    fill_value = df[col].median()
                elif strategy == "fill_zero":
                    fill_value = 0
                else:  # Default to mean
                    fill_value = df[col].mean()
                
                result_df[col] = df[col].fillna(fill_value)
        
        # Handle categorical columns
        for col in cat_cols:
            if df[col].isnull().any():
                if strategy == "mode":
                    fill_value = df[col].mode()[0]
                else:  # Default to mode for categorical
                    fill_value = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                
                result_df[col] = df[col].fillna(fill_value)
        
        return result_df, missing_info
    
    def _convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert data types to appropriate types.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with converted data types
        """
        result_df = df.copy()
        
        # Try to convert to more efficient dtypes
        try:
            for col in df.columns:
                # Try to convert string columns to categorical if they have few unique values
                if df[col].dtype == 'object':
                    num_unique = df[col].nunique()
                    if num_unique < len(df) * 0.5:  # If less than 50% unique values
                        result_df[col] = df[col].astype('category')
            
            # Use pandas' convert_dtypes for other automatic conversions
            result_df = result_df.convert_dtypes()
            logger.info("Successfully converted data types")
            
        except Exception as e:
            logger.warning(f"Error during data type conversion: {e}")
        
        return result_df
    
    def _handle_outliers(self, df: pd.DataFrame, columns: List[str] = None, method: str = "iqr") -> pd.DataFrame:
        """
        Handle outliers in specified columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (List[str], optional): Columns to check for outliers. If None, uses all numeric columns.
            method (str): Method for detecting outliers ('iqr', 'zscore', 'cap')
            
        Returns:
            pd.DataFrame: DataFrame with handled outliers
        """
        result_df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=np.number).columns
        
        for col in columns:
            if col not in df.columns or not np.issubdtype(df[col].dtype, np.number):
                continue
            
            if method == "iqr":
                # IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap the outliers
                result_df.loc[result_df[col] < lower_bound, col] = lower_bound
                result_df.loc[result_df[col] > upper_bound, col] = upper_bound
                
            elif method == "zscore":
                # Z-score method
                mean = df[col].mean()
                std = df[col].std()
                threshold = 3  # Standard threshold for z-score
                
                # Cap the outliers
                result_df.loc[np.abs((df[col] - mean) / std) > threshold, col] = np.nan
                # Fill NaN values with mean
                result_df[col] = result_df[col].fillna(mean)
                
            elif method == "cap":
                # Percentile capping
                lower_bound = df[col].quantile(0.01)  # 1st percentile
                upper_bound = df[col].quantile(0.99)  # 99th percentile
                
                # Cap the outliers
                result_df.loc[result_df[col] < lower_bound, col] = lower_bound
                result_df.loc[result_df[col] > upper_bound, col] = upper_bound
        
        logger.info(f"Handled outliers in {len(columns)} columns using {method} method")
        return result_df
    
    def _apply_transformations(self, df: pd.DataFrame, transformations: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply custom transformations to columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            transformations (Dict[str, Any]): Dictionary of transformations
            
        Returns:
            pd.DataFrame: Transformed DataFrame
        """
        result_df = df.copy()
        
        for col, transform in transformations.items():
            if col not in df.columns:
                logger.warning(f"Column {col} not found for transformation")
                continue
            
            if transform == "log":
                # Log transformation for positive data
                if (df[col] <= 0).any():
                    logger.warning(f"Cannot apply log transformation to column {col} with non-positive values")
                else:
                    result_df[col] = np.log(df[col])
            
            elif transform == "sqrt":
                # Square root transformation for non-negative data
                if (df[col] < 0).any():
                    logger.warning(f"Cannot apply sqrt transformation to column {col} with negative values")
                else:
                    result_df[col] = np.sqrt(df[col])
            
            elif transform == "normalize":
                # Min-max scaling to [0, 1]
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val - min_val > 0:
                    result_df[col] = (df[col] - min_val) / (max_val - min_val)
            
            elif transform == "standardize":
                # Z-score standardization
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    result_df[col] = (df[col] - mean) / std
            
            elif callable(transform):
                # Custom function
                try:
                    result_df[col] = transform(df[col])
                except Exception as e:
                    logger.error(f"Error applying custom transformation to {col}: {e}")
        
        return result_df
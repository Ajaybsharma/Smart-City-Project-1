import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional, Callable
import logging
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Creates meaningful features from raw data for machine learning."""
    
    def __init__(self):
        """Initialize the FeatureEngineer."""
        self.encoders = {}
        self.scalers = {}
        logger.info("Initialized FeatureEngineer")
    
    def create_features(self, df: pd.DataFrame, feature_config: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Generate new features based on configuration.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            feature_config (Dict[str, Any], optional): Configuration for feature generation
            
        Returns:
            pd.DataFrame: DataFrame with new features
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for feature engineering")
            return df
        
        result_df = df.copy()
        
        # Use default feature engineering if no config is provided
        if feature_config is None:
            logger.info("Using default feature engineering configuration")
            feature_config = {
                "categorical_encoding": "default",
                "numerical_scaling": False,
                "date_features": "auto",
                "text_features": False,
                "interaction_features": False
            }
        
        # Process categorical features
        if feature_config.get("categorical_encoding", False):
            categorical_cols = feature_config.get("categorical_columns", None)
            encoding_method = feature_config.get("categorical_encoding", "default")
            
            if categorical_cols is None:
                # Auto-detect categorical columns
                categorical_cols = result_df.select_dtypes(include=['object', 'category']).columns
            
            result_df = self._encode_categorical_features(result_df, categorical_cols, method=encoding_method)
        
        # Scale numerical features
        if feature_config.get("numerical_scaling", False):
            numerical_cols = feature_config.get("numerical_columns", None)
            scaling_method = feature_config.get("scaling_method", "standard")
            
            if numerical_cols is None:
                # Auto-detect numerical columns
                numerical_cols = result_df.select_dtypes(include=np.number).columns
            
            result_df = self._scale_numerical_features(result_df, numerical_cols, method=scaling_method)
        
        # Generate date/time features
        if feature_config.get("date_features", False):
            date_cols = feature_config.get("date_columns", None)
            result_df = self._create_datetime_features(result_df, date_cols)
        
        # Generate text features
        if feature_config.get("text_features", False):
            text_cols = feature_config.get("text_columns", [])
            max_features = feature_config.get("text_max_features", 100)
            
            for col in text_cols:
                if col in df.columns:
                    result_df = self._create_text_features(result_df, col, max_features=max_features)
        
        # Generate interaction features
        if feature_config.get("interaction_features", False):
            interaction_cols = feature_config.get("interaction_columns", [])
            if interaction_cols:
                result_df = self._create_interaction_features(result_df, interaction_cols)
        
        # Generate geographical features
        if feature_config.get("geo_features", False):
            lat_col = feature_config.get("latitude_column")
            lon_col = feature_config.get("longitude_column")
            
            if lat_col and lon_col and lat_col in df.columns and lon_col in df.columns:
                result_df = self._create_geographical_features(result_df, lat_col, lon_col)
        
        # Apply custom features
        if feature_config.get("custom_features", False):
            custom_features = feature_config.get("custom_feature_functions", {})
            result_df = self._apply_custom_features(result_df, custom_features)
        
        # Log feature engineering summary
        original_cols = set(df.columns)
        new_cols = set(result_df.columns) - original_cols
        
        logger.info(f"Feature engineering completed: Added {len(new_cols)} new features")
        logger.debug(f"New features: {new_cols}")
        
        return result_df
    
    def _encode_categorical_features(self, df: pd.DataFrame, columns: List[str], method: str = "onehot") -> pd.DataFrame:
        """
        Encode categorical features using various methods.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (List[str]): Categorical columns to encode
            method (str): Encoding method ('onehot', 'label', 'ordinal', 'target')
            
        Returns:
            pd.DataFrame: DataFrame with encoded features
        """
        result_df = df.copy()
        
        # Filter out columns that don't exist
        columns = [col for col in columns if col in df.columns]
        
        if not columns:
            logger.warning("No valid categorical columns found for encoding")
            return result_df
        
        logger.info(f"Encoding {len(columns)} categorical features using {method} method")
        
        if method == "onehot" or method == "default":
            # One-hot encoding
            for col in columns:
                # Skip if already numeric
                if pd.api.types.is_numeric_dtype(df[col]):
                    continue
                
                # For columns with many categories, consider other encoding methods
                if df[col].nunique() > 10:
                    logger.warning(f"Column {col} has {df[col].nunique()} unique values. Consider using a different encoding method.")
                
                # Create a unique key for this column's encoder
                encoder_key = f"onehot_{col}"
                
                if encoder_key not in self.encoders:
                    encoder = OneHotEncoder(sparse_output=False, drop='first')
                    encoded_data = encoder.fit_transform(df[[col]])
                    self.encoders[encoder_key] = encoder
                else:
                    encoder = self.encoders[encoder_key]
                    encoded_data = encoder.transform(df[[col]])
                
                # Get the feature names from the encoder
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
                
                # Create a DataFrame with the encoded data
                encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df.index)
                
                # Concatenate the encoded features to the result DataFrame
                result_df = pd.concat([result_df, encoded_df], axis=1)
                
                # Drop the original column
                result_df = result_df.drop(col, axis=1)
        
        elif method == "label":
            # Label encoding
            for col in columns:
                # Skip if already numeric
                if pd.api.types.is_numeric_dtype(df[col]):
                    continue
                
                # Create a unique key for this column's encoder
                encoder_key = f"label_{col}"
                
                if encoder_key not in self.encoders:
                    encoder = LabelEncoder()
                    result_df[col] = encoder.fit_transform(df[col].astype(str))
                    self.encoders[encoder_key] = encoder
                else:
                    encoder = self.encoders[encoder_key]
                    result_df[col] = encoder.transform(df[col].astype(str))
        
        elif method == "ordinal":
            # Ordinal encoding (assumes categories are provided in the right order)
            for col in columns:
                # Skip if already numeric
                if pd.api.types.is_numeric_dtype(df[col]):
                    continue
                
                # Get unique categories in the data
                categories = df[col].unique()
                
                # Create mapping dictionary (0 to n-1)
                mapping = {cat: i for i, cat in enumerate(categories)}
                
                # Apply mapping
                result_df[col] = df[col].map(mapping)
        
        return result_df
    
    def _scale_numerical_features(self, df: pd.DataFrame, columns: List[str], method: str = "standard") -> pd.DataFrame:
        """
        Scale numerical features using various methods.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (List[str]): Numerical columns to scale
            method (str): Scaling method ('standard', 'minmax', 'robust')
            
        Returns:
            pd.DataFrame: DataFrame with scaled features
        """
        result_df = df.copy()
        
        # Filter out columns that don't exist or aren't numeric
        columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if not columns:
            logger.warning("No valid numerical columns found for scaling")
            return result_df
        
        logger.info(f"Scaling {len(columns)} numerical features using {method} method")
        
        if method == "standard":
            # Standard scaling (z-score normalization)
            if "standard_scaler" not in self.scalers:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df[columns])
                self.scalers["standard_scaler"] = scaler
            else:
                scaler = self.scalers["standard_scaler"]
                scaled_data = scaler.transform(df[columns])
            
            # Update the DataFrame with scaled values
            for i, col in enumerate(columns):
                result_df[col] = scaled_data[:, i]
        
        elif method == "minmax":
            # Min-max scaling
            for col in columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val - min_val > 0:
                    result_df[col] = (df[col] - min_val) / (max_val - min_val)
        
        elif method == "robust":
            # Robust scaling using median and IQR
            for col in columns:
                median = df[col].median()
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    result_df[col] = (df[col] - median) / iqr
        
        return result_df
    
    def _create_datetime_features(self, df: pd.DataFrame, date_columns: List[str] = None) -> pd.DataFrame:
        """
        Create features from datetime columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            date_columns (List[str], optional): Datetime columns to process
            
        Returns:
            pd.DataFrame: DataFrame with datetime features
        """
        result_df = df.copy()
        
        # Auto-detect date columns if none provided
        if date_columns is None:
            # Try to find datetime columns
            date_columns = []
            for col in df.columns:
                # Check if the column is already a datetime type
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    date_columns.append(col)
                # Try to convert string columns to datetime
                elif df[col].dtype == 'object':
                    try:
                        pd.to_datetime(df[col])
                        date_columns.append(col)
                    except:
                        pass
        
        if not date_columns:
            logger.warning("No valid datetime columns found for feature extraction")
            return result_df
        
        logger.info(f"Creating date features for {len(date_columns)} columns")
        
        for col in date_columns:
            if col not in df.columns:
                continue
            
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    result_df[col] = pd.to_datetime(df[col])
                except:
                    logger.warning(f"Could not convert column {col} to datetime")
                    continue
            
            # Extract date features
            result_df[f"{col}_year"] = result_df[col].dt.year
            result_df[f"{col}_month"] = result_df[col].dt.month
            result_df[f"{col}_day"] = result_df[col].dt.day
            result_df[f"{col}_dayofweek"] = result_df[col].dt.dayofweek
            result_df[f"{col}_quarter"] = result_df[col].dt.quarter
            result_df[f"{col}_is_weekend"] = result_df[col].dt.dayofweek >= 5
            
            # Extract time features if time data is available
            if (result_df[col].dt.hour != 0).any():
                result_df[f"{col}_hour"] = result_df[col].dt.hour
                result_df[f"{col}_minute"] = result_df[col].dt.minute
                
                # Create time period features
                result_df[f"{col}_time_period"] = pd.cut(
                    result_df[col].dt.hour,
                    bins=[0, 6, 12, 18, 24],
                    labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                    right=False
                )
            
            # Calculate days from reference date (e.g., beginning of the year)
            try:
                min_date = result_df[col].min()
                result_df[f"{col}_days_from_min"] = (result_df[col] - min_date).dt.days
            except:
                logger.warning(f"Could not calculate days from reference for {col}")
        
        return result_df
    
    def _create_text_features(self, df: pd.DataFrame, text_column: str, max_features: int = 100) -> pd.DataFrame:
        """
        Create features from text data using TF-IDF.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_column (str): Column containing text data
            max_features (int): Maximum number of features to extract
            
        Returns:
            pd.DataFrame: DataFrame with text features
        """
        result_df = df.copy()
        
        if text_column not in df.columns:
            logger.warning(f"Text column {text_column} not found in DataFrame")
            return result_df
        
        logger.info(f"Creating text features for column {text_column}")
        
        # Create TF-IDF features
        encoder_key = f"tfidf_{text_column}"
        
        if encoder_key not in self.encoders:
            # Fill missing values
            texts = df[text_column].fillna("").astype(str)
            
            # Create and fit the vectorizer
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                min_df=2
            )
            
            # Transform the text data
            text_features = vectorizer.fit_transform(texts)
            self.encoders[encoder_key] = vectorizer
        else:
            # Use existing vectorizer
            vectorizer = self.encoders[encoder_key]
            texts = df[text_column].fillna("").astype(str)
            text_features = vectorizer.transform(texts)
        
        # Create feature names
        feature_names = [f"{text_column}_tfidf_{i}" for i in range(text_features.shape[1])]
        
        # Convert to DataFrame
        text_df = pd.DataFrame(
            text_features.toarray(),
            columns=feature_names,
            index=df.index
        )
        
        # Concatenate with result DataFrame
        result_df = pd.concat([result_df, text_df], axis=1)
        
        return result_df
    
    def _create_interaction_features(self, df: pd.DataFrame, column_pairs: List[List[str]]) -> pd.DataFrame:
        """
        Create interaction features between pairs of numerical columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            column_pairs (List[List[str]]): Pairs of columns to create interactions for
            
        Returns:
            pd.DataFrame: DataFrame with interaction features
        """
        result_df = df.copy()
        
        if not column_pairs:
            # Auto-generate pairs from numerical columns (up to a reasonable limit)
            numerical_cols = df.select_dtypes(include=np.number).columns
            if len(numerical_cols) > 10:
                logger.warning("Too many numerical columns for auto-interaction. Provide specific pairs.")
                return result_df
            
            column_pairs = []
            for i, col1 in enumerate(numerical_cols):
                for col2 in numerical_cols[i+1:]:
                    column_pairs.append([col1, col2])
        
        logger.info(f"Creating interaction features for {len(column_pairs)} column pairs")
        
        for pair in column_pairs:
            if len(pair) != 2:
                logger.warning(f"Interaction pair must contain exactly 2 columns: {pair}")
                continue
            
            col1, col2 = pair
            
            if col1 not in df.columns or col2 not in df.columns:
                logger.warning(f"One or both columns not found: {col1}, {col2}")
                continue
            
            # Skip if either column is not numeric
            if not pd.api.types.is_numeric_dtype(df[col1]) or not pd.api.types.is_numeric_dtype(df[col2]):
                logger.warning(f"Both columns must be numeric for interactions: {col1}, {col2}")
                continue
            
            # Create interaction features
            result_df[f"{col1}_multiply_{col2}"] = df[col1] * df[col2]
            result_df[f"{col1}_divide_{col2}"] = df[col1] / df[col2].replace(0, np.nan)
            result_df[f"{col1}_plus_{col2}"] = df[col1] + df[col2]
            result_df[f"{col1}_minus_{col2}"] = df[col1] - df[col2]
            
            # Fill NaN values that might result from division
            result_df = result_df.fillna(df.mean())
        
        return result_df
    
    def _create_geographical_features(self, df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
        """
        Create features from geographical coordinates.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            lat_col (str): Column containing latitude values
            lon_col (str): Column containing longitude values
            
        Returns:
            pd.DataFrame: DataFrame with geographical features
        """
        result_df = df.copy()
        
        if lat_col not in df.columns or lon_col not in df.columns:
            logger.warning(f"Latitude or longitude column not found: {lat_col}, {lon_col}")
            return result_df
        
        logger.info(f"Creating geographical features from {lat_col} and {lon_col}")
        
        # Calculate distance from the center of the dataset
        center_lat = df[lat_col].mean()
        center_lon = df[lon_col].mean()
        
        # Haversine distance calculation
        R = 6371  # Earth's radius in km
        
        # Convert to radians
        lat_rad = np.radians(df[lat_col])
        lon_rad = np.radians(df[lon_col])
        center_lat_rad = np.radians(center_lat)
        center_lon_rad = np.radians(center_lon)
        
        # Haversine formula
        dlon = lon_rad - center_lon_rad
        dlat = lat_rad - center_lat_rad
        a = np.sin(dlat/2)**2 + np.cos(lat_rad) * np.cos(center_lat_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        
        # Add distance feature
        result_df[f"distance_from_center"] = distance
        
        return result_df
    
    def _apply_custom_features(self, df: pd.DataFrame, custom_features: Dict[str, Callable]) -> pd.DataFrame:
        """
        Apply custom feature engineering functions.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            custom_features (Dict[str, Callable]): Dictionary mapping feature names to functions
            
        Returns:
            pd.DataFrame: DataFrame with custom features
        """
        result_df = df.copy()
        
        logger.info(f"Applying {len(custom_features)} custom feature functions")
        
        for feature_name, feature_func in custom_features.items():
            try:
                result_df[feature_name] = feature_func(df)
                logger.debug(f"Created custom feature: {feature_name}")
            except Exception as e:
                logger.error(f"Error creating custom feature {feature_name}: {e}")
        
        return result_df
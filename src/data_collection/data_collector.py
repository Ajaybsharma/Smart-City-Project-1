import pandas as pd
import logging
from typing import List, Dict, Any
from .input_sources import DataSource

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCollector:
    """
    Aggregates and validates data from multiple sources.
    """
    
    def __init__(self, data_sources: List[DataSource] = None):
        """
        Initialize DataCollector with a list of data sources.
        
        Args:
            data_sources (List[DataSource]): List of data source objects
        """
        self.data_sources = data_sources or []
        self.collected_data = {}
        logger.info(f"Initialized DataCollector with {len(self.data_sources)} data sources")
    
    def add_data_source(self, data_source: DataSource):
        """
        Add a data source to the collector.
        
        Args:
            data_source (DataSource): Data source object
        """
        self.data_sources.append(data_source)
        logger.info(f"Added data source: {data_source.__class__.__name__}")
    
    def collect_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Collect data from all registered data sources.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping source names to collected data
        """
        for source in self.data_sources:
            source_name = source.get_metadata()["source_name"]
            logger.info(f"Collecting data from {source_name}")
            
            try:
                data = source.fetch_data()
                if not data.empty:
                    self.collected_data[source_name] = data
                    logger.info(f"Successfully collected {len(data)} records from {source_name}")
                else:
                    logger.warning(f"No data collected from {source_name}")
            except Exception as e:
                logger.error(f"Error collecting data from {source_name}: {e}")
        
        return self.collected_data
    
    def validate_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Validate collected data for quality and consistency.
        
        Returns:
            Dict[str, Dict[str, Any]]: Validation results for each data source
        """
        validation_results = {}
        
        for source_name, data in self.collected_data.items():
            logger.info(f"Validating data from {source_name}")
            
            try:
                # Basic validation checks
                validation = {
                    "row_count": len(data),
                    "column_count": len(data.columns),
                    "missing_values": data.isnull().sum().to_dict(),
                    "duplicate_rows": data.duplicated().sum(),
                    "validation_passed": True
                }
                
                # Check for critical issues
                if validation["row_count"] == 0:
                    validation["validation_passed"] = False
                    validation["issues"] = ["No data available"]
                
                elif validation["duplicate_rows"] > 0:
                    logger.warning(f"Found {validation['duplicate_rows']} duplicate rows in {source_name}")
                
                # Check for high percentage of missing values
                missing_pct = data.isnull().mean().max() * 100
                if missing_pct > 20:  # If any column has more than 20% missing values
                    logger.warning(f"High percentage of missing values ({missing_pct:.2f}%) in {source_name}")
                    validation["high_missing_values"] = True
                
                validation_results[source_name] = validation
                logger.info(f"Validation completed for {source_name}")
                
            except Exception as e:
                logger.error(f"Error validating data from {source_name}: {e}")
                validation_results[source_name] = {
                    "validation_passed": False,
                    "error": str(e)
                }
        
        return validation_results
    
    def get_data(self, source_name: str = None) -> pd.DataFrame:
        """
        Get collected data from a specific source or merged data from all sources.
        
        Args:
            source_name (str, optional): Name of the source to get data from.
                                        If None, returns a dictionary of all data.
        
        Returns:
            pd.DataFrame: Collected data
        """
        if source_name:
            if source_name in self.collected_data:
                return self.collected_data[source_name]
            else:
                logger.warning(f"No data found for source: {source_name}")
                return pd.DataFrame()
        else:
            return self.collected_data
    
    def get_collection_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a summary of all collected data.
        
        Returns:
            Dict[str, Dict[str, Any]]: Summary statistics for each data source
        """
        summary = {}
        
        for source_name, data in self.collected_data.items():
            try:
                summary[source_name] = {
                    "row_count": len(data),
                    "column_count": len(data.columns),
                    "columns": list(data.columns),
                    "data_types": {col: str(dtype) for col, dtype in data.dtypes.items()},
                    "memory_usage": data.memory_usage(deep=True).sum() / (1024 * 1024),  # in MB
                }
            except Exception as e:
                logger.error(f"Error generating summary for {source_name}: {e}")
                summary[source_name] = {"error": str(e)}
        
        return summary
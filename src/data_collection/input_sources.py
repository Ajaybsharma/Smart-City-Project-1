from abc import ABC, abstractmethod
import requests
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataSource(ABC):
    """Base abstract class for all data sources."""
    
    @abstractmethod
    def fetch_data(self):
        """Fetch data from the source."""
        pass
    
    @abstractmethod
    def get_metadata(self):
        """Get metadata about the data source."""
        pass


class OpenStreetMapSource(DataSource):
    """Data source for OpenStreetMap API."""
    
    def __init__(self, api_endpoint="https://nominatim.openstreetmap.org/search"):
        self.api_endpoint = api_endpoint
        logger.info("Initialized OpenStreetMap data source")
    
    def fetch_data(self, query, limit=10):
        """
        Fetch geographical data from OpenStreetMap.
        
        Args:
            query (str): Search query (e.g., city name)
            limit (int): Maximum number of results
            
        Returns:
            pandas.DataFrame: Geographical data
        """
        try:
            params = {
                'q': query,
                'format': 'json',
                'limit': limit
            }
            
            response = requests.get(self.api_endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Successfully fetched {len(data)} records from OpenStreetMap")
            
            return pd.DataFrame(data)
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from OpenStreetMap: {e}")
            return pd.DataFrame()
    
    def get_metadata(self):
        """Get metadata about the OpenStreetMap data source."""
        return {
            "source_name": "OpenStreetMap",
            "source_type": "API",
            "data_type": "Geographical",
            "api_endpoint": self.api_endpoint
        }


class RealEstateDataSource(DataSource):
    """Data source for real estate datasets."""
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        logger.info("Initialized Real Estate data source")
    
    def fetch_data(self):
        """
        Fetch real estate data from CSV file.
        
        Returns:
            pandas.DataFrame: Real estate data
        """
        try:
            if not self.data_path or not os.path.exists(self.data_path):
                logger.warning("Data path not specified or does not exist")
                # Return sample data for demonstration
                return pd.DataFrame({
                    'property_id': range(1, 11),
                    'price': [250000, 300000, 350000, 400000, 450000, 500000, 550000, 600000, 650000, 700000],
                    'size_sqft': [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800],
                    'bedrooms': [2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
                    'bathrooms': [1, 1.5, 2, 2, 2.5, 2.5, 3, 3, 3.5, 4],
                    'year_built': [1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2022],
                    'location': ['Downtown', 'Suburb', 'Downtown', 'Suburb', 'Downtown', 'Suburb', 'Downtown', 'Suburb', 'Downtown', 'Suburb']
                })
            
            data = pd.read_csv(self.data_path)
            logger.info(f"Successfully loaded {len(data)} records from real estate dataset")
            return data
            
        except Exception as e:
            logger.error(f"Error loading real estate data: {e}")
            return pd.DataFrame()
    
    def get_metadata(self):
        """Get metadata about the real estate data source."""
        return {
            "source_name": "Real Estate Dataset",
            "source_type": "CSV",
            "data_type": "Housing",
            "data_path": self.data_path
        }


class IoTEnvironmentalSource(DataSource):
    """Data source for IoT environmental sensors."""
    
    def __init__(self, api_endpoint=None, api_key=None):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        logger.info("Initialized IoT Environmental data source")
    
    def fetch_data(self, sensor_ids=None):
        """
        Fetch environmental data from IoT sensors.
        
        Args:
            sensor_ids (list): List of sensor IDs to fetch data from
            
        Returns:
            pandas.DataFrame: Environmental sensor data
        """
        try:
            if not self.api_endpoint or not self.api_key:
                logger.warning("API endpoint or key not specified")
                # Return sample data for demonstration
                return pd.DataFrame({
                    'sensor_id': ['S001', 'S002', 'S003', 'S004', 'S005', 'S006', 'S007', 'S008', 'S009', 'S010'],
                    'temperature': [22.5, 23.1, 21.8, 24.2, 22.9, 21.5, 23.8, 22.2, 24.5, 23.3],
                    'humidity': [45, 48, 52, 40, 47, 53, 46, 50, 42, 49],
                    'air_quality_index': [65, 70, 85, 60, 75, 90, 68, 72, 58, 80],
                    'noise_level': [45, 50, 60, 40, 55, 65, 48, 52, 38, 58],
                    'timestamp': pd.date_range(start='2023-01-01', periods=10, freq='H')
                })
            
            # TODO: Implement actual API call with authentication
            # For now, return sample data
            logger.info("Returning sample IoT sensor data")
            return pd.DataFrame({
                'sensor_id': ['S001', 'S002', 'S003', 'S004', 'S005'],
                'temperature': [22.5, 23.1, 21.8, 24.2, 22.9],
                'humidity': [45, 48, 52, 40, 47],
                'air_quality_index': [65, 70, 85, 60, 75],
                'noise_level': [45, 50, 60, 40, 55],
                'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='H')
            })
            
        except Exception as e:
            logger.error(f"Error fetching IoT environmental data: {e}")
            return pd.DataFrame()
    
    def get_metadata(self):
        """Get metadata about the IoT environmental data source."""
        return {
            "source_name": "IoT Environmental Sensors",
            "source_type": "API",
            "data_type": "Environmental",
            "api_endpoint": self.api_endpoint
        }
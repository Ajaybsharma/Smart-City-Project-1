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

class WaterMetricsAnalyzer:
    """Analyzes water availability and usage patterns."""
    
    def __init__(self):
        """Initialize the WaterMetricsAnalyzer."""
        logger.info("Initialized WaterMetricsAnalyzer")
    
    def analyze_water_consumption(self, data: pd.DataFrame, 
                                consumption_column: str, 
                                location_column: Optional[str] = None,
                                time_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze water consumption patterns.
        
        Args:
            data (pd.DataFrame): Water consumption data
            consumption_column (str): Column containing consumption values
            location_column (str, optional): Column containing location information
            time_column (str, optional): Column containing time information
            
        Returns:
            Dict[str, Any]: Water consumption analysis results
        """
        if data.empty:
            logger.warning("Empty dataset provided for water consumption analysis")
            return {"status": "error", "message": "Empty dataset"}
        
        if consumption_column not in data.columns:
            logger.error(f"Consumption column '{consumption_column}' not found in data")
            return {"status": "error", "message": f"Column not found: {consumption_column}"}
        
        logger.info(f"Analyzing water consumption from column: {consumption_column}")
        
        try:
            analysis = {
                "status": "success",
                "total_records": len(data),
                "total_consumption": float(data[consumption_column].sum()),
                "average_consumption": float(data[consumption_column].mean()),
                "consumption_stats": {
                    "min": float(data[consumption_column].min()),
                    "25%": float(data[consumption_column].quantile(0.25)),
                    "median": float(data[consumption_column].median()),
                    "75%": float(data[consumption_column].quantile(0.75)),
                    "max": float(data[consumption_column].max()),
                    "std_dev": float(data[consumption_column].std())
                }
            }
            
            # Add location-based analysis if location column is provided
            if location_column and location_column in data.columns:
                location_analysis = self._analyze_by_category(data, consumption_column, location_column)
                analysis["location_analysis"] = location_analysis
            
            # Add time-based analysis if time column is provided
            if time_column and time_column in data.columns:
                time_analysis = self._analyze_time_patterns(data, consumption_column, time_column)
                analysis["time_analysis"] = time_analysis
            
            # Calculate sustainability metrics
            sustainability_metrics = self._calculate_sustainability_metrics(data, consumption_column)
            analysis["sustainability_metrics"] = sustainability_metrics
            
            logger.info(f"Completed water consumption analysis with {len(data)} records")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error during water consumption analysis: {e}")
            return {"status": "error", "message": str(e)}
    
    def _analyze_by_category(self, data: pd.DataFrame, value_column: str, category_column: str) -> Dict[str, Any]:
        """
        Analyze water consumption by category (e.g., location, building type).
        
        Args:
            data (pd.DataFrame): Water consumption data
            value_column (str): Column containing consumption values
            category_column (str): Column containing category information
            
        Returns:
            Dict[str, Any]: Analysis by category
        """
        # Group by category and calculate statistics
        grouped = data.groupby(category_column)[value_column].agg(['sum', 'mean', 'median', 'std', 'count'])
        
        # Convert to dictionary for JSON serialization
        category_analysis = {}
        
        for category, stats in grouped.iterrows():
            category_analysis[str(category)] = {
                "total": float(stats['sum']),
                "average": float(stats['mean']),
                "median": float(stats['median']),
                "std_dev": float(stats['std']) if not pd.isna(stats['std']) else 0,
                "count": int(stats['count']),
                "percentage_of_total": float(stats['sum'] / data[value_column].sum() * 100)
            }
        
        # Get top and bottom categories by consumption
        sorted_categories = sorted(category_analysis.items(), key=lambda x: x[1]['total'], reverse=True)
        
        top_categories = {k: v for k, v in sorted_categories[:5]}
        bottom_categories = {k: v for k, v in sorted_categories[-5:] if len(sorted_categories) >= 5}
        
        return {
            "by_category": category_analysis,
            "top_consumers": top_categories,
            "bottom_consumers": bottom_categories,
            "category_count": len(category_analysis)
        }
    
    def _analyze_time_patterns(self, data: pd.DataFrame, value_column: str, time_column: str) -> Dict[str, Any]:
        """
        Analyze water consumption patterns over time.
        
        Args:
            data (pd.DataFrame): Water consumption data
            value_column (str): Column containing consumption values
            time_column (str): Column containing time information
            
        Returns:
            Dict[str, Any]: Time pattern analysis
        """
        # Ensure time column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
            try:
                data[time_column] = pd.to_datetime(data[time_column])
            except:
                logger.warning(f"Could not convert {time_column} to datetime format")
                return {"status": "error", "message": "Invalid time format"}
        
        # Create time-based features
        data['year'] = data[time_column].dt.year
        data['month'] = data[time_column].dt.month
        data['day_of_week'] = data[time_column].dt.dayofweek
        
        # Analyze by year
        yearly_analysis = self._analyze_by_category(data, value_column, 'year')
        
        # Analyze by month
        monthly_analysis = self._analyze_by_category(data, value_column, 'month')
        
        # Analyze by day of week
        dow_analysis = self._analyze_by_category(data, value_column, 'day_of_week')
        
        # Analyze trends
        trends = self._analyze_trends(data, value_column, time_column)
        
        return {
            "yearly_analysis": yearly_analysis,
            "monthly_analysis": monthly_analysis,
            "day_of_week_analysis": dow_analysis,
            "trends": trends
        }
    
    def _analyze_trends(self, data: pd.DataFrame, value_column: str, time_column: str) -> Dict[str, Any]:
        """
        Analyze consumption trends over time.
        
        Args:
            data (pd.DataFrame): Consumption data
            value_column (str): Column containing consumption values
            time_column (str): Column containing time information
            
        Returns:
            Dict[str, Any]: Trend analysis
        """
        # Sort data by time
        data_sorted = data.sort_values(by=time_column)
        
        # Resample to monthly data for trend analysis
        try:
            monthly_data = data_sorted.set_index(time_column).resample('M')[value_column].mean()
            
            # Check if we have enough data for trend analysis
            if len(monthly_data) < 3:
                return {"status": "warning", "message": "Insufficient data for trend analysis"}
            
            # Calculate simple linear regression for trend
            from scipy import stats
            
            x = np.arange(len(monthly_data))
            y = monthly_data.values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            trend_significance = "significant" if p_value < 0.05 else "not significant"
            
            # Calculate percent change from start to end
            if monthly_data.iloc[0] != 0:
                percent_change = (monthly_data.iloc[-1] - monthly_data.iloc[0]) / monthly_data.iloc[0] * 100
            else:
                percent_change = 0
            
            return {
                "slope": float(slope),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "direction": trend_direction,
                "significance": trend_significance,
                "percent_change": float(percent_change),
                "period_start": monthly_data.index[0].isoformat(),
                "period_end": monthly_data.index[-1].isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Error in trend analysis: {e}")
            return {"status": "error", "message": str(e)}
    
    def _calculate_sustainability_metrics(self, data: pd.DataFrame, consumption_column: str) -> Dict[str, Any]:
        """
        Calculate water sustainability metrics.
        
        Args:
            data (pd.DataFrame): Water consumption data
            consumption_column (str): Column containing consumption values
            
        Returns:
            Dict[str, Any]: Sustainability metrics
        """
        total_consumption = data[consumption_column].sum()
        avg_consumption = data[consumption_column].mean()
        
        # Calculate per capita metrics if population data available
        per_capita_available = False
        per_capita_metrics = {}
        
        if 'population' in data.columns:
            per_capita_available = True
            total_population = data['population'].sum()
            per_capita_consumption = total_consumption / total_population if total_population > 0 else 0
            
            per_capita_metrics = {
                "per_capita_consumption": float(per_capita_consumption),
                "population": int(total_population)
            }
        
        # Calculate efficiency metrics if data available
        efficiency_available = False
        efficiency_metrics = {}
        
        if 'area_served' in data.columns:
            efficiency_available = True
            total_area = data['area_served'].sum()
            consumption_per_area = total_consumption / total_area if total_area > 0 else 0
            
            efficiency_metrics = {
                "consumption_per_area": float(consumption_per_area),
                "total_area_served": float(total_area)
            }
        
        # Calculate basic sustainability score (example)
        # This is a simplified model; in reality, this would be more complex
        sustainability_score = 0
        
        # Assume industry benchmarks for comparison
        benchmark_consumption = 100  # Example benchmark
        
        if avg_consumption < benchmark_consumption * 0.8:
            sustainability_score = 5  # Excellent
        elif avg_consumption < benchmark_consumption:
            sustainability_score = 4  # Good
        elif avg_consumption < benchmark_consumption * 1.2:
            sustainability_score = 3  # Average
        elif avg_consumption < benchmark_consumption * 1.5:
            sustainability_score = 2  # Below average
        else:
            sustainability_score = 1  # Poor
        
        metrics = {
            "total_consumption": float(total_consumption),
            "average_consumption": float(avg_consumption),
            "sustainability_score": sustainability_score,
            "benchmark_comparison": float(avg_consumption / benchmark_consumption) if benchmark_consumption > 0 else 0
        }
        
        if per_capita_available:
            metrics["per_capita_metrics"] = per_capita_metrics
        
        if efficiency_available:
            metrics["efficiency_metrics"] = efficiency_metrics
        
        return metrics
    
    def analyze_water_availability(self, supply_data: pd.DataFrame, 
                                  demand_data: pd.DataFrame = None,
                                  supply_column: str = "supply",
                                  demand_column: str = "demand",
                                  location_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze water availability and supply-demand balance.
        
        Args:
            supply_data (pd.DataFrame): Water supply data
            demand_data (pd.DataFrame, optional): Water demand data (if separate from supply)
            supply_column (str): Column containing supply values
            demand_column (str): Column containing demand values
            location_column (str, optional): Column containing location information
            
        Returns:
            Dict[str, Any]: Water availability analysis results
        """
        if supply_data.empty:
            logger.warning("Empty supply dataset provided for water availability analysis")
            return {"status": "error", "message": "Empty supply dataset"}
        
        if supply_column not in supply_data.columns:
            logger.error(f"Supply column '{supply_column}' not found in data")
            return {"status": "error", "message": f"Column not found: {supply_column}"}
        
        logger.info("Analyzing water availability")
        
        try:
            # Initialize analysis result
            analysis = {
                "status": "success",
                "total_records": len(supply_data),
                "total_supply": float(supply_data[supply_column].sum()),
                "average_supply": float(supply_data[supply_column].mean()),
                "supply_stats": {
                    "min": float(supply_data[supply_column].min()),
                    "25%": float(supply_data[supply_column].quantile(0.25)),
                    "median": float(supply_data[supply_column].median()),
                    "75%": float(supply_data[supply_column].quantile(0.75)),
                    "max": float(supply_data[supply_column].max()),
                    "std_dev": float(supply_data[supply_column].std())
                }
            }
            
            # If demand data is provided separately
            if demand_data is not None and not demand_data.empty and demand_column in demand_data.columns:
                total_demand = demand_data[demand_column].sum()
                analysis["total_demand"] = float(total_demand)
                analysis["average_demand"] = float(demand_data[demand_column].mean())
                analysis["demand_stats"] = {
                    "min": float(demand_data[demand_column].min()),
                    "25%": float(demand_data[demand_column].quantile(0.25)),
                    "median": float(demand_data[demand_column].median()),
                    "75%": float(demand_data[demand_column].quantile(0.75)),
                    "max": float(demand_data[demand_column].max()),
                    "std_dev": float(demand_data[demand_column].std())
                }
                
                # Calculate supply-demand balance
                supply_demand_ratio = analysis["total_supply"] / total_demand if total_demand > 0 else float('inf')
                analysis["supply_demand_ratio"] = float(supply_demand_ratio)
                analysis["supply_surplus_deficit"] = float(analysis["total_supply"] - total_demand)
                
                # Assess water stress level
                if supply_demand_ratio > 1.5:
                    stress_level = "Low"
                elif supply_demand_ratio > 1.0:
                    stress_level = "Moderate"
                elif supply_demand_ratio > 0.8:
                    stress_level = "High"
                else:
                    stress_level = "Extremely High"
                
                analysis["water_stress_level"] = stress_level
                
            # If both supply and demand are in the same DataFrame
            elif demand_column in supply_data.columns:
                total_demand = supply_data[demand_column].sum()
                analysis["total_demand"] = float(total_demand)
                analysis["average_demand"] = float(supply_data[demand_column].mean())
                analysis["demand_stats"] = {
                    "min": float(supply_data[demand_column].min()),
                    "25%": float(supply_data[demand_column].quantile(0.25)),
                    "median": float(supply_data[demand_column].median()),
                    "75%": float(supply_data[demand_column].quantile(0.75)),
                    "max": float(supply_data[demand_column].max()),
                    "std_dev": float(supply_data[demand_column].std())
                }
                
                # Calculate supply-demand balance
                supply_demand_ratio = analysis["total_supply"] / total_demand if total_demand > 0 else float('inf')
                analysis["supply_demand_ratio"] = float(supply_demand_ratio)
                analysis["supply_surplus_deficit"] = float(analysis["total_supply"] - total_demand)
                
                # Assess water stress level
                if supply_demand_ratio > 1.5:
                    stress_level = "Low"
                elif supply_demand_ratio > 1.0:
                    stress_level = "Moderate"
                elif supply_demand_ratio > 0.8:
                    stress_level = "High"
                else:
                    stress_level = "Extremely High"
                
                analysis["water_stress_level"] = stress_level
            
            # Add location-based analysis if location column is provided
            if location_column and location_column in supply_data.columns:
                location_analysis = self._analyze_by_category(supply_data, supply_column, location_column)
                analysis["location_analysis"] = location_analysis
                
                # If we also have demand by location
                if demand_column in supply_data.columns:
                    # Calculate water stress by location
                    stress_by_location = {}
                    
                    for location in supply_data[location_column].unique():
                        location_data = supply_data[supply_data[location_column] == location]
                        location_supply = location_data[supply_column].sum()
                        location_demand = location_data[demand_column].sum()
                        
                        if location_demand > 0:
                            ratio = location_supply / location_demand
                            surplus_deficit = location_supply - location_demand
                            
                            if ratio > 1.5:
                                stress = "Low"
                            elif ratio > 1.0:
                                stress = "Moderate"
                            elif ratio > 0.8:
                                stress = "High"
                            else:
                                stress = "Extremely High"
                            
                            stress_by_location[str(location)] = {
                                "supply": float(location_supply),
                                "demand": float(location_demand),
                                "ratio": float(ratio),
                                "surplus_deficit": float(surplus_deficit),
                                "stress_level": stress
                            }
                    
                    analysis["water_stress_by_location"] = stress_by_location
            
            logger.info(f"Completed water availability analysis with {len(supply_data)} records")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error during water availability analysis: {e}")
            return {"status": "error", "message": str(e)}
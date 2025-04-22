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

class WasteManagementAnalyzer:
    """Evaluates waste handling and recycling patterns."""
    
    def __init__(self):
        """Initialize the WasteManagementAnalyzer."""
        logger.info("Initialized WasteManagementAnalyzer")
    
    def analyze_waste_generation(self, data: pd.DataFrame, 
                               waste_column: str,
                               category_column: Optional[str] = None,
                               location_column: Optional[str] = None,
                               time_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze waste generation patterns.
        
        Args:
            data (pd.DataFrame): Waste generation data
            waste_column (str): Column containing waste amount values
            category_column (str, optional): Column containing waste category information
            location_column (str, optional): Column containing location information
            time_column (str, optional): Column containing time information
            
        Returns:
            Dict[str, Any]: Waste generation analysis results
        """
        if data.empty:
            logger.warning("Empty dataset provided for waste generation analysis")
            return {"status": "error", "message": "Empty dataset"}
        
        if waste_column not in data.columns:
            logger.error(f"Waste column '{waste_column}' not found in data")
            return {"status": "error", "message": f"Column not found: {waste_column}"}
        
        logger.info(f"Analyzing waste generation from column: {waste_column}")
        
        try:
            analysis = {
                "status": "success",
                "total_records": len(data),
                "total_waste": float(data[waste_column].sum()),
                "average_waste": float(data[waste_column].mean()),
                "waste_stats": {
                    "min": float(data[waste_column].min()),
                    "25%": float(data[waste_column].quantile(0.25)),
                    "median": float(data[waste_column].median()),
                    "75%": float(data[waste_column].quantile(0.75)),
                    "max": float(data[waste_column].max()),
                    "std_dev": float(data[waste_column].std())
                }
            }
            
            # Add category-based analysis if category column is provided
            if category_column and category_column in data.columns:
                category_analysis = self._analyze_by_category(data, waste_column, category_column)
                analysis["category_analysis"] = category_analysis
            
            # Add location-based analysis if location column is provided
            if location_column and location_column in data.columns:
                location_analysis = self._analyze_by_category(data, waste_column, location_column)
                analysis["location_analysis"] = location_analysis
            
            # Add time-based analysis if time column is provided
            if time_column and time_column in data.columns:
                time_analysis = self._analyze_time_patterns(data, waste_column, time_column)
                analysis["time_analysis"] = time_analysis
            
            # Calculate waste generation metrics
            waste_metrics = self._calculate_waste_metrics(data, waste_column)
            analysis["waste_metrics"] = waste_metrics
            
            logger.info(f"Completed waste generation analysis with {len(data)} records")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error during waste generation analysis: {e}")
            return {"status": "error", "message": str(e)}
    
    def analyze_recycling(self, data: pd.DataFrame,
                         waste_column: str,
                         recycled_column: str,
                         category_column: Optional[str] = None,
                         location_column: Optional[str] = None,
                         time_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze recycling rates and patterns.
        
        Args:
            data (pd.DataFrame): Recycling data
            waste_column (str): Column containing total waste amount
            recycled_column (str): Column containing recycled waste amount
            category_column (str, optional): Column containing waste category information
            location_column (str, optional): Column containing location information
            time_column (str, optional): Column containing time information
            
        Returns:
            Dict[str, Any]: Recycling analysis results
        """
        if data.empty:
            logger.warning("Empty dataset provided for recycling analysis")
            return {"status": "error", "message": "Empty dataset"}
        
        if waste_column not in data.columns or recycled_column not in data.columns:
            missing = []
            if waste_column not in data.columns:
                missing.append(waste_column)
            if recycled_column not in data.columns:
                missing.append(recycled_column)
            
            logger.error(f"Column(s) not found: {', '.join(missing)}")
            return {"status": "error", "message": f"Column(s) not found: {', '.join(missing)}"}
        
        logger.info(f"Analyzing recycling using waste column: {waste_column} and recycled column: {recycled_column}")
        
        try:
            # Calculate recycling rate
            data['recycling_rate'] = np.where(
                data[waste_column] > 0,
                data[recycled_column] / data[waste_column] * 100,
                0
            )
            
            analysis = {
                "status": "success",
                "total_records": len(data),
                "total_waste": float(data[waste_column].sum()),
                "total_recycled": float(data[recycled_column].sum()),
                "overall_recycling_rate": float(data[recycled_column].sum() / data[waste_column].sum() * 100) if data[waste_column].sum() > 0 else 0,
                "average_recycling_rate": float(data['recycling_rate'].mean()),
                "recycling_rate_stats": {
                    "min": float(data['recycling_rate'].min()),
                    "25%": float(data['recycling_rate'].quantile(0.25)),
                    "median": float(data['recycling_rate'].median()),
                    "75%": float(data['recycling_rate'].quantile(0.75)),
                    "max": float(data['recycling_rate'].max()),
                    "std_dev": float(data['recycling_rate'].std())
                }
            }
            
            # Add category-based analysis if category column is provided
            if category_column and category_column in data.columns:
                category_analysis = self._analyze_recycling_by_category(data, waste_column, recycled_column, category_column)
                analysis["category_analysis"] = category_analysis
            
            # Add location-based analysis if location column is provided
            if location_column and location_column in data.columns:
                location_analysis = self._analyze_recycling_by_category(data, waste_column, recycled_column, location_column)
                analysis["location_analysis"] = location_analysis
            
            # Add time-based analysis if time column is provided
            if time_column and time_column in data.columns:
                time_analysis = self._analyze_recycling_time_patterns(data, waste_column, recycled_column, time_column)
                analysis["time_analysis"] = time_analysis
            
            # Calculate recycling efficiency metrics
            recycling_metrics = self._calculate_recycling_metrics(data, waste_column, recycled_column)
            analysis["recycling_metrics"] = recycling_metrics
            
            logger.info(f"Completed recycling analysis with {len(data)} records")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error during recycling analysis: {e}")
            return {"status": "error", "message": str(e)}
    
    def _analyze_by_category(self, data: pd.DataFrame, value_column: str, category_column: str) -> Dict[str, Any]:
        """
        Analyze waste generation by category.
        
        Args:
            data (pd.DataFrame): Waste data
            value_column (str): Column containing waste values
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
                "percentage_of_total": float(stats['sum'] / data[value_column].sum() * 100) if data[value_column].sum() > 0 else 0
            }
        
        # Get top and bottom categories by waste generation
        sorted_categories = sorted(category_analysis.items(), key=lambda x: x[1]['total'], reverse=True)
        
        top_categories = {k: v for k, v in sorted_categories[:5]}
        bottom_categories = {k: v for k, v in sorted_categories[-5:] if len(sorted_categories) >= 5}
        
        return {
            "by_category": category_analysis,
            "top_generators": top_categories,
            "bottom_generators": bottom_categories,
            "category_count": len(category_analysis)
        }
    
    def _analyze_recycling_by_category(self, data: pd.DataFrame, waste_column: str, 
                                     recycled_column: str, category_column: str) -> Dict[str, Any]:
        """
        Analyze recycling rates by category.
        
        Args:
            data (pd.DataFrame): Recycling data
            waste_column (str): Column containing total waste amount
            recycled_column (str): Column containing recycled waste amount
            category_column (str): Column containing category information
            
        Returns:
            Dict[str, Any]: Recycling analysis by category
        """
        # Group by category and calculate statistics
        grouped = data.groupby(category_column).agg({
            waste_column: ['sum', 'mean'],
            recycled_column: ['sum', 'mean'],
            'recycling_rate': ['mean', 'median', 'std', 'min', 'max']
        })
        
        # Flatten the multi-index columns
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        
        # Calculate overall recycling rate for each category
        grouped['overall_recycling_rate'] = grouped[f"{recycled_column}_sum"] / grouped[f"{waste_column}_sum"] * 100
        
        # Convert to dictionary for JSON serialization
        category_analysis = {}
        
        for category, stats in grouped.iterrows():
            category_analysis[str(category)] = {
                "total_waste": float(stats[f"{waste_column}_sum"]),
                "total_recycled": float(stats[f"{recycled_column}_sum"]),
                "overall_recycling_rate": float(stats['overall_recycling_rate']),
                "average_recycling_rate": float(stats['recycling_rate_mean']),
                "median_recycling_rate": float(stats['recycling_rate_median']),
                "min_recycling_rate": float(stats['recycling_rate_min']),
                "max_recycling_rate": float(stats['recycling_rate_max']),
                "std_dev_recycling_rate": float(stats['recycling_rate_std']) if not pd.isna(stats['recycling_rate_std']) else 0,
                "percentage_of_total_waste": float(stats[f"{waste_column}_sum"] / data[waste_column].sum() * 100) if data[waste_column].sum() > 0 else 0,
                "percentage_of_total_recycled": float(stats[f"{recycled_column}_sum"] / data[recycled_column].sum() * 100) if data[recycled_column].sum() > 0 else 0
            }
        
        # Get top and bottom categories by recycling rate
        sorted_by_rate = sorted(category_analysis.items(), key=lambda x: x[1]['overall_recycling_rate'], reverse=True)
        
        top_recyclers = {k: v for k, v in sorted_by_rate[:5]}
        bottom_recyclers = {k: v for k, v in sorted_by_rate[-5:] if len(sorted_by_rate) >= 5}
        
        # Get top waste generators
        sorted_by_waste = sorted(category_analysis.items(), key=lambda x: x[1]['total_waste'], reverse=True)
        top_waste_generators = {k: v for k, v in sorted_by_waste[:5]}
        
        return {
            "by_category": category_analysis,
            "top_recyclers": top_recyclers,
            "bottom_recyclers": bottom_recyclers,
            "top_waste_generators": top_waste_generators,
            "category_count": len(category_analysis)
        }
    
    def _analyze_time_patterns(self, data: pd.DataFrame, value_column: str, time_column: str) -> Dict[str, Any]:
        """
        Analyze waste generation patterns over time.
        
        Args:
            data (pd.DataFrame): Waste data
            value_column (str): Column containing waste values
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
    
    def _analyze_recycling_time_patterns(self, data: pd.DataFrame, waste_column: str, 
                                       recycled_column: str, time_column: str) -> Dict[str, Any]:
        """
        Analyze recycling patterns over time.
        
        Args:
            data (pd.DataFrame): Recycling data
            waste_column (str): Column containing total waste amount
            recycled_column (str): Column containing recycled waste amount
            time_column (str): Column containing time information
            
        Returns:
            Dict[str, Any]: Time pattern analysis for recycling
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
        
        # Analyze recycling by year
        yearly_grouped = data.groupby('year').agg({
            waste_column: 'sum',
            recycled_column: 'sum'
        })
        yearly_grouped['recycling_rate'] = yearly_grouped[recycled_column] / yearly_grouped[waste_column] * 100
        
        yearly_analysis = {
            "by_year": {
                str(year): {
                    "total_waste": float(stats[waste_column]),
                    "total_recycled": float(stats[recycled_column]),
                    "recycling_rate": float(stats['recycling_rate'])
                }
                for year, stats in yearly_grouped.iterrows()
            }
        }
        
        # Analyze recycling by month
        monthly_grouped = data.groupby('month').agg({
            waste_column: 'sum',
            recycled_column: 'sum'
        })
        monthly_grouped['recycling_rate'] = monthly_grouped[recycled_column] / monthly_grouped[waste_column] * 100
        
        monthly_analysis = {
            "by_month": {
                str(month): {
                    "total_waste": float(stats[waste_column]),
                    "total_recycled": float(stats[recycled_column]),
                    "recycling_rate": float(stats['recycling_rate'])
                }
                for month, stats in monthly_grouped.iterrows()
            }
        }
        
        # Analyze recycling by day of week
        dow_grouped = data.groupby('day_of_week').agg({
            waste_column: 'sum',
            recycled_column: 'sum'
        })
        dow_grouped['recycling_rate'] = dow_grouped[recycled_column] / dow_grouped[waste_column] * 100
        
        dow_analysis = {
            "by_day_of_week": {
                str(dow): {
                    "total_waste": float(stats[waste_column]),
                    "total_recycled": float(stats[recycled_column]),
                    "recycling_rate": float(stats['recycling_rate'])
                }
                for dow, stats in dow_grouped.iterrows()
            }
        }
        
        # Analyze trends in recycling rate
        trends = self._analyze_recycling_trends(data, waste_column, recycled_column, time_column)
        
        return {
            "yearly_analysis": yearly_analysis,
            "monthly_analysis": monthly_analysis,
            "day_of_week_analysis": dow_analysis,
            "trends": trends
        }
    
    def _analyze_trends(self, data: pd.DataFrame, value_column: str, time_column: str) -> Dict[str, Any]:
        """
        Analyze waste generation trends over time.
        
        Args:
            data (pd.DataFrame): Waste data
            value_column (str): Column containing waste values
            time_column (str): Column containing time information
            
        Returns:
            Dict[str, Any]: Trend analysis
        """
        # Sort data by time
        data_sorted = data.sort_values(by=time_column)
        
        # Resample to monthly data for trend analysis
        try:
            monthly_data = data_sorted.set_index(time_column).resample('M')[value_column].sum()
            
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
            
            # Calculate seasonality (simplified)
            has_seasonality = False
            seasonality_pattern = "none"
            
            if len(monthly_data) >= 24:  # At least 2 years of data
                # Check for yearly patterns
                yearly_avg = monthly_data.groupby(monthly_data.index.month).mean()
                yearly_std = monthly_data.groupby(monthly_data.index.month).std()
                
                # If standard deviation between months is high, might indicate seasonality
                if yearly_std.mean() / yearly_avg.mean() > 0.2:
                    has_seasonality = True
                    
                    # Check which months have highest values
                    high_month = yearly_avg.idxmax()
                    
                    if high_month in [12, 1, 2]:
                        seasonality_pattern = "winter peak"
                    elif high_month in [3, 4, 5]:
                        seasonality_pattern = "spring peak"
                    elif high_month in [6, 7, 8]:
                        seasonality_pattern = "summer peak"
                    elif high_month in [9, 10, 11]:
                        seasonality_pattern = "fall peak"
            
            return {
                "slope": float(slope),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "direction": trend_direction,
                "significance": trend_significance,
                "percent_change": float(percent_change),
                "has_seasonality": has_seasonality,
                "seasonality_pattern": seasonality_pattern,
                "period_start": monthly_data.index[0].isoformat(),
                "period_end": monthly_data.index[-1].isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Error in trend analysis: {e}")
            return {"status": "error", "message": str(e)}
    
    def _analyze_recycling_trends(self, data: pd.DataFrame, waste_column: str, 
                                recycled_column: str, time_column: str) -> Dict[str, Any]:
        """
        Analyze recycling trends over time.
        
        Args:
            data (pd.DataFrame): Recycling data
            waste_column (str): Column containing total waste amount
            recycled_column (str): Column containing recycled waste amount
            time_column (str): Column containing time information
            
        Returns:
            Dict[str, Any]: Recycling trend analysis
        """
        # Sort data by time
        data_sorted = data.sort_values(by=time_column)
        
        # Resample to monthly data for trend analysis
        try:
            monthly_data = data_sorted.set_index(time_column).resample('M').agg({
                waste_column: 'sum',
                recycled_column: 'sum'
            })
            
            # Calculate monthly recycling rate
            monthly_data['recycling_rate'] = monthly_data[recycled_column] / monthly_data[waste_column] * 100
            
            # Check if we have enough data for trend analysis
            if len(monthly_data) < 3:
                return {"status": "warning", "message": "Insufficient data for trend analysis"}
            
            # Calculate trend for recycling rate
            from scipy import stats
            
            x = np.arange(len(monthly_data))
            y = monthly_data['recycling_rate'].values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            trend_significance = "significant" if p_value < 0.05 else "not significant"
            
            # Calculate percent change from start to end
            if monthly_data['recycling_rate'].iloc[0] != 0:
                percent_change = (monthly_data['recycling_rate'].iloc[-1] - monthly_data['recycling_rate'].iloc[0]) / monthly_data['recycling_rate'].iloc[0] * 100
            else:
                percent_change = 0
            
            return {
                "recycling_rate_trend": {
                    "slope": float(slope),
                    "r_squared": float(r_value ** 2),
                    "p_value": float(p_value),
                    "direction": trend_direction,
                    "significance": trend_significance,
                    "percent_change": float(percent_change)
                },
                "period_start": monthly_data.index[0].isoformat(),
                "period_end": monthly_data.index[-1].isoformat(),
                "start_recycling_rate": float(monthly_data['recycling_rate'].iloc[0]),
                "end_recycling_rate": float(monthly_data['recycling_rate'].iloc[-1]),
                "peak_recycling_rate": {
                    "rate": float(monthly_data['recycling_rate'].max()),
                    "date": monthly_data['recycling_rate'].idxmax().isoformat()
                },
                "lowest_recycling_rate": {
                    "rate": float(monthly_data['recycling_rate'].min()),
                    "date": monthly_data['recycling_rate'].idxmin().isoformat()
                }
            }
            
        except Exception as e:
            logger.warning(f"Error in recycling trend analysis: {e}")
            return {"status": "error", "message": str(e)}
    
    def _calculate_waste_metrics(self, data: pd.DataFrame, waste_column: str) -> Dict[str, Any]:
        """
        Calculate waste generation metrics.
        
        Args:
            data (pd.DataFrame): Waste data
            waste_column (str): Column containing waste values
            
        Returns:
            Dict[str, Any]: Waste metrics
        """
        total_waste = data[waste_column].sum()
        avg_waste = data[waste_column].mean()
        
        # Calculate per capita metrics if population data available
        per_capita_available = False
        per_capita_metrics = {}
        
        if 'population' in data.columns:
            per_capita_available = True
            total_population = data['population'].sum()
            per_capita_waste = total_waste / total_population if total_population > 0 else 0
            
            per_capita_metrics = {
                "per_capita_waste": float(per_capita_waste),
                "population": int(total_population)
            }
        
        # Calculate waste generation intensity if area data available
        intensity_available = False
        intensity_metrics = {}
        
        if 'area' in data.columns:
            intensity_available = True
            total_area = data['area'].sum()
            waste_per_area = total_waste / total_area if total_area > 0 else 0
            
            intensity_metrics = {
                "waste_per_area": float(waste_per_area),
                "total_area": float(total_area)
            }
        
        # Calculate basic waste management score (example)
        # This is a simplified model; in reality, this would be more complex
        waste_score = 0
        
        # Assume industry benchmarks for comparison
        benchmark_waste = 100  # Example benchmark
        
        if avg_waste < benchmark_waste * 0.8:
            waste_score = 5  # Excellent
        elif avg_waste < benchmark_waste:
            waste_score = 4  # Good
        elif avg_waste < benchmark_waste * 1.2:
            waste_score = 3  # Average
        elif avg_waste < benchmark_waste * 1.5:
            waste_score = 2  # Below average
        else:
            waste_score = 1  # Poor
        
        metrics = {
            "total_waste": float(total_waste),
            "average_waste": float(avg_waste),
            "waste_management_score": waste_score,
            "benchmark_comparison": float(avg_waste / benchmark_waste) if benchmark_waste > 0 else 0
        }
        
        if per_capita_available:
            metrics["per_capita_metrics"] = per_capita_metrics
        
        if intensity_available:
            metrics["intensity_metrics"] = intensity_metrics
        
        return metrics
    
    def _calculate_recycling_metrics(self, data: pd.DataFrame, waste_column: str, recycled_column: str) -> Dict[str, Any]:
        """
        Calculate recycling efficiency metrics.
        
        Args:
            data (pd.DataFrame): Recycling data
            waste_column (str): Column containing total waste amount
            recycled_column (str): Column containing recycled waste amount
            
        Returns:
            Dict[str, Any]: Recycling metrics
        """
        total_waste = data[waste_column].sum()
        total_recycled = data[recycled_column].sum()
        overall_recycling_rate = total_recycled / total_waste * 100 if total_waste > 0 else 0
        
        # Calculate per capita metrics if population data available
        per_capita_available = False
        per_capita_metrics = {}
        
        if 'population' in data.columns:
            per_capita_available = True
            total_population = data['population'].sum()
            per_capita_recycled = total_recycled / total_population if total_population > 0 else 0
            
            per_capita_metrics = {
                "per_capita_recycled": float(per_capita_recycled),
                "population": int(total_population)
            }
        
        # Calculate recycling score
        # This is a simplified scoring model
        recycling_score = 0
        
        if overall_recycling_rate >= 50:
            recycling_score = 5  # Excellent
        elif overall_recycling_rate >= 35:
            recycling_score = 4  # Good
        elif overall_recycling_rate >= 20:
            recycling_score = 3  # Average
        elif overall_recycling_rate >= 10:
            recycling_score = 2  # Below average
        else:
            recycling_score = 1  # Poor
        
        # Calculate diversion rate (percentage of waste diverted from landfill)
        diversion_rate = overall_recycling_rate
        
        # If we have composting data, add to diversion rate
        if 'composted' in data.columns:
            total_composted = data['composted'].sum()
            composting_rate = total_composted / total_waste * 100 if total_waste > 0 else 0
            diversion_rate += composting_rate
        
        metrics = {
            "total_waste": float(total_waste),
            "total_recycled": float(total_recycled),
            "overall_recycling_rate": float(overall_recycling_rate),
            "diversion_rate": float(diversion_rate),
            "recycling_score": recycling_score,
            "landfill_percentage": float(100 - diversion_rate)
        }
        
        if per_capita_available:
            metrics["per_capita_metrics"] = per_capita_metrics
        
        # Benchmark comparison
        benchmark_recycling_rate = 35  # Example benchmark
        metrics["benchmark_comparison"] = float(overall_recycling_rate / benchmark_recycling_rate) if benchmark_recycling_rate > 0 else 0
        
        return metrics
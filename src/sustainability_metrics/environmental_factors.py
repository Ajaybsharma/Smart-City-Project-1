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

class EnvironmentalFactorsAnalyzer:
    """Considers various environmental parameters for sustainability analysis."""
    
    def __init__(self):
        """Initialize the EnvironmentalFactorsAnalyzer."""
        logger.info("Initialized EnvironmentalFactorsAnalyzer")
    
    def analyze_air_quality(self, data: pd.DataFrame,
                           pollutant_columns: List[str],
                           location_column: Optional[str] = None,
                           time_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze air quality data.
        
        Args:
            data (pd.DataFrame): Air quality data
            pollutant_columns (List[str]): Columns containing pollutant measurements
            location_column (str, optional): Column containing location information
            time_column (str, optional): Column containing time information
            
        Returns:
            Dict[str, Any]: Air quality analysis results
        """
        if data.empty:
            logger.warning("Empty dataset provided for air quality analysis")
            return {"status": "error", "message": "Empty dataset"}
        
        # Check if all pollutant columns exist
        missing_columns = [col for col in pollutant_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Columns not found: {', '.join(missing_columns)}")
            return {"status": "error", "message": f"Columns not found: {', '.join(missing_columns)}"}
        
        logger.info(f"Analyzing air quality using {len(pollutant_columns)} pollutant metrics")
        
        try:
            analysis = {
                "status": "success",
                "total_records": len(data),
                "pollutants_analyzed": pollutant_columns,
                "pollutant_stats": {}
            }
            
            # Calculate statistics for each pollutant
            for pollutant in pollutant_columns:
                analysis["pollutant_stats"][pollutant] = {
                    "mean": float(data[pollutant].mean()),
                    "median": float(data[pollutant].median()),
                    "min": float(data[pollutant].min()),
                    "max": float(data[pollutant].max()),
                    "std_dev": float(data[pollutant].std()),
                    "25%": float(data[pollutant].quantile(0.25)),
                    "75%": float(data[pollutant].quantile(0.75))
                }
            
            # Calculate Air Quality Index (simplified)
            aqi = self._calculate_air_quality_index(data, pollutant_columns)
            analysis["air_quality_index"] = aqi
            
            # Add location-based analysis if location column is provided
            if location_column and location_column in data.columns:
                location_analysis = self._analyze_by_location(data, pollutant_columns, location_column, aqi_enabled=True)
                analysis["location_analysis"] = location_analysis
            
            # Add time-based analysis if time column is provided
            if time_column and time_column in data.columns:
                time_analysis = self._analyze_time_patterns(data, pollutant_columns, time_column, analysis_type="air_quality")
                analysis["time_analysis"] = time_analysis
            
            # Calculate pollution hotspots
            if location_column and location_column in data.columns:
                hotspots = self._identify_pollution_hotspots(data, pollutant_columns, location_column)
                analysis["pollution_hotspots"] = hotspots
            
            logger.info(f"Completed air quality analysis with {len(data)} records")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error during air quality analysis: {e}")
            return {"status": "error", "message": str(e)}
    
    def analyze_greenhouse_emissions(self, data: pd.DataFrame,
                                   emission_columns: List[str],
                                   source_column: Optional[str] = None,
                                   location_column: Optional[str] = None,
                                   time_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze greenhouse gas emissions.
        
        Args:
            data (pd.DataFrame): Emissions data
            emission_columns (List[str]): Columns containing emission measurements
            source_column (str, optional): Column containing emission source information
            location_column (str, optional): Column containing location information
            time_column (str, optional): Column containing time information
            
        Returns:
            Dict[str, Any]: Emissions analysis results
        """
        if data.empty:
            logger.warning("Empty dataset provided for emissions analysis")
            return {"status": "error", "message": "Empty dataset"}
        
        # Check if all emission columns exist
        missing_columns = [col for col in emission_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Columns not found: {', '.join(missing_columns)}")
            return {"status": "error", "message": f"Columns not found: {', '.join(missing_columns)}"}
        
        logger.info(f"Analyzing greenhouse emissions using {len(emission_columns)} emission metrics")
        
        try:
            analysis = {
                "status": "success",
                "total_records": len(data),
                "emissions_analyzed": emission_columns,
                "emission_stats": {}
            }
            
            # Calculate total emissions and convert to CO2-equivalent if applicable
            total_emissions = 0
            
            for emission in emission_columns:
                emission_sum = data[emission].sum()
                analysis["emission_stats"][emission] = {
                    "total": float(emission_sum),
                    "mean": float(data[emission].mean()),
                    "median": float(data[emission].median()),
                    "min": float(data[emission].min()),
                    "max": float(data[emission].max()),
                    "std_dev": float(data[emission].std())
                }
                
                # Add to total emissions (simplified - in reality would use GWP conversion factors)
                # For example, methane has 25x the GWP of CO2
                if emission.lower() == "co2" or emission.lower() == "carbon_dioxide":
                    total_emissions += emission_sum
                elif emission.lower() == "ch4" or emission.lower() == "methane":
                    total_emissions += emission_sum * 25  # Methane GWP
                elif emission.lower() == "n2o" or emission.lower() == "nitrous_oxide":
                    total_emissions += emission_sum * 298  # N2O GWP
                else:
                    total_emissions += emission_sum  # Assume CO2-equivalent
            
            analysis["total_co2_equivalent"] = float(total_emissions)
            
            # Add source-based analysis if source column is provided
            if source_column and source_column in data.columns:
                source_analysis = self._analyze_emissions_by_source(data, emission_columns, source_column)
                analysis["source_analysis"] = source_analysis
            
            # Add location-based analysis if location column is provided
            if location_column and location_column in data.columns:
                location_analysis = self._analyze_by_location(data, emission_columns, location_column, analysis_type="emissions")
                analysis["location_analysis"] = location_analysis
            
            # Add time-based analysis if time column is provided
            if time_column and time_column in data.columns:
                time_analysis = self._analyze_time_patterns(data, emission_columns, time_column, analysis_type="emissions")
                analysis["time_analysis"] = time_analysis
            
            # Calculate carbon footprint and intensity
            carbon_metrics = self._calculate_carbon_metrics(data, emission_columns)
            analysis["carbon_metrics"] = carbon_metrics
            
            logger.info(f"Completed greenhouse emissions analysis with {len(data)} records")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error during emissions analysis: {e}")
            return {"status": "error", "message": str(e)}
    
    def analyze_biodiversity(self, data: pd.DataFrame,
                           species_column: str,
                           count_column: str,
                           habitat_column: Optional[str] = None,
                           location_column: Optional[str] = None,
                           time_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze biodiversity data.
        
        Args:
            data (pd.DataFrame): Biodiversity data
            species_column (str): Column containing species information
            count_column (str): Column containing species count
            habitat_column (str, optional): Column containing habitat information
            location_column (str, optional): Column containing location information
            time_column (str, optional): Column containing time information
            
        Returns:
            Dict[str, Any]: Biodiversity analysis results
        """
        if data.empty:
            logger.warning("Empty dataset provided for biodiversity analysis")
            return {"status": "error", "message": "Empty dataset"}
        
        # Check if required columns exist
        required_columns = [species_column, count_column]
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"Required columns not found: {', '.join(missing_columns)}")
            return {"status": "error", "message": f"Required columns not found: {', '.join(missing_columns)}"}
        
        logger.info("Analyzing biodiversity data")
        
        try:
            # Calculate diversity indices
            species_count = data[species_column].nunique()
            total_individuals = data[count_column].sum()
            
            # Calculate Shannon Diversity Index
            shannon_index = self._calculate_shannon_index(data, species_column, count_column)
            
            # Calculate Simpson Diversity Index
            simpson_index = self._calculate_simpson_index(data, species_column, count_column)
            
            analysis = {
                "status": "success",
                "total_records": len(data),
                "species_count": species_count,
                "total_individuals": float(total_individuals),
                "shannon_diversity_index": float(shannon_index),
                "simpson_diversity_index": float(simpson_index),
                "species_distribution": self._get_species_distribution(data, species_column, count_column)
            }
            
            # Add habitat-based analysis if habitat column is provided
            if habitat_column and habitat_column in data.columns:
                habitat_analysis = self._analyze_biodiversity_by_habitat(data, species_column, count_column, habitat_column)
                analysis["habitat_analysis"] = habitat_analysis
            
            # Add location-based analysis if location column is provided
            if location_column and location_column in data.columns:
                location_analysis = self._analyze_biodiversity_by_location(data, species_column, count_column, location_column)
                analysis["location_analysis"] = location_analysis
            
            # Add time-based analysis if time column is provided
            if time_column and time_column in data.columns:
                time_analysis = self._analyze_biodiversity_time_patterns(data, species_column, count_column, time_column)
                analysis["time_analysis"] = time_analysis
            
            logger.info(f"Completed biodiversity analysis with {len(data)} records")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error during biodiversity analysis: {e}")
            return {"status": "error", "message": str(e)}
    
    def _calculate_air_quality_index(self, data: pd.DataFrame, pollutant_columns: List[str]) -> Dict[str, Any]:
        """
        Calculate Air Quality Index based on pollutant data.
        
        Args:
            data (pd.DataFrame): Air quality data
            pollutant_columns (List[str]): Columns containing pollutant measurements
            
        Returns:
            Dict[str, Any]: Air Quality Index information
        """
        # This is a simplified AQI calculation
        # In reality, AQI calculations are more complex and vary by country/region
        
        # Define pollutant breakpoints (simplified example)
        # These would typically follow EPA or similar guidelines
        breakpoints = {
            "pm25": [0, 12, 35.4, 55.4, 150.4, 250.4, 350.4, 500.4],
            "pm10": [0, 54, 154, 254, 354, 424, 504, 604],
            "o3": [0, 54, 70, 85, 105, 200, 404, 504],
            "co": [0, 4.4, 9.4, 12.4, 15.4, 30.4, 40.4, 50.4],
            "so2": [0, 35, 75, 185, 304, 604, 804, 1004],
            "no2": [0, 53, 100, 360, 649, 1249, 1649, 2049]
        }
        
        # Define AQI levels
        aqi_levels = [0, 50, 100, 150, 200, 300, 400, 500]
        
        # Calculate AQI for each pollutant
        pollutant_aqis = {}
        max_aqi = 0
        max_pollutant = ""
        
        for pollutant in pollutant_columns:
            # Get the average concentration
            concentration = data[pollutant].mean()
            
            # Find the matching breakpoint key (simplistic approach)
            breakpoint_key = None
            for key in breakpoints:
                if key in pollutant.lower():
                    breakpoint_key = key
                    break
            
            # If we found a matching breakpoint, calculate AQI
            if breakpoint_key:
                # Find which breakpoint range the concentration falls into
                for i in range(len(breakpoints[breakpoint_key]) - 1):
                    if concentration >= breakpoints[breakpoint_key][i] and concentration < breakpoints[breakpoint_key][i + 1]:
                        # Calculate AQI using linear interpolation
                        bp_low = breakpoints[breakpoint_key][i]
                        bp_high = breakpoints[breakpoint_key][i + 1]
                        aqi_low = aqi_levels[i]
                        aqi_high = aqi_levels[i + 1]
                        
                        aqi = ((aqi_high - aqi_low) / (bp_high - bp_low)) * (concentration - bp_low) + aqi_low
                        
                        pollutant_aqis[pollutant] = float(aqi)
                        
                        # Keep track of the max AQI
                        if aqi > max_aqi:
                            max_aqi = aqi
                            max_pollutant = pollutant
                        
                        break
            else:
                logger.warning(f"No breakpoint definition found for pollutant: {pollutant}")
        
        # Determine AQI category
        if max_aqi <= 50:
            category = "Good"
            health_concern = "Air quality is satisfactory, and air pollution poses little or no risk."
        elif max_aqi <= 100:
            category = "Moderate"
            health_concern = "Air quality is acceptable. However, there may be a risk for some people."
        elif max_aqi <= 150:
            category = "Unhealthy for Sensitive Groups"
            health_concern = "Members of sensitive groups may experience health effects."
        elif max_aqi <= 200:
            category = "Unhealthy"
            health_concern = "Some members of the general public may experience health effects."
        elif max_aqi <= 300:
            category = "Very Unhealthy"
            health_concern = "Health alert: The risk of health effects is increased for everyone."
        else:
            category = "Hazardous"
            health_concern = "Health warning of emergency conditions: everyone is more likely to be affected."
        
        return {
            "aqi_value": float(max_aqi),
            "primary_pollutant": max_pollutant,
            "category": category,
            "health_concern": health_concern,
            "pollutant_aqis": pollutant_aqis
        }
    
    def _analyze_by_location(self, data: pd.DataFrame, value_columns: List[str], 
                           location_column: str, analysis_type: str = "general", 
                           aqi_enabled: bool = False) -> Dict[str, Any]:
        """
        Analyze data by location.
        
        Args:
            data (pd.DataFrame): Input data
            value_columns (List[str]): Columns containing values to analyze
            location_column (str): Column containing location information
            analysis_type (str): Type of analysis ('general', 'emissions', etc.)
            aqi_enabled (bool): Whether to calculate AQI for each location
            
        Returns:
            Dict[str, Any]: Analysis by location
        """
        locations = data[location_column].unique()
        location_analysis = {}
        
        for location in locations:
            location_data = data[data[location_column] == location]
            
            # Calculate statistics for each value column
            location_stats = {}
            
            for column in value_columns:
                location_stats[column] = {
                    "mean": float(location_data[column].mean()),
                    "median": float(location_data[column].median()),
                    "min": float(location_data[column].min()),
                    "max": float(location_data[column].max()),
                    "std_dev": float(location_data[column].std()),
                    "total": float(location_data[column].sum())
                }
            
            # Add AQI if enabled
            if aqi_enabled:
                aqi = self._calculate_air_quality_index(location_data, value_columns)
                location_stats["aqi"] = aqi
            
            # Add emissions-specific metrics if applicable
            if analysis_type == "emissions":
                total_emissions = sum(location_data[col].sum() for col in value_columns)
                location_stats["total_emissions"] = float(total_emissions)
                
                # Calculate per capita emissions if population data is available
                if "population" in location_data.columns:
                    population = location_data["population"].iloc[0]  # Assume same population for all records
                    per_capita = total_emissions / population if population > 0 else 0
                    location_stats["per_capita_emissions"] = float(per_capita)
            
            location_analysis[str(location)] = location_stats
        
        return location_analysis
    
    def _analyze_time_patterns(self, data: pd.DataFrame, value_columns: List[str], 
                             time_column: str, analysis_type: str = "general") -> Dict[str, Any]:
        """
        Analyze data patterns over time.
        
        Args:
            data (pd.DataFrame): Input data
            value_columns (List[str]): Columns containing values to analyze
            time_column (str): Column containing time information
            analysis_type (str): Type of analysis ('general', 'emissions', 'air_quality')
            
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
        
        # Analyze yearly patterns
        yearly_means = data.groupby('year')[value_columns].mean().reset_index()
        yearly_sums = data.groupby('year')[value_columns].sum().reset_index()
        
        yearly_trends = {}
        for column in value_columns:
            yearly_trends[column] = {
                "mean_by_year": {str(year): float(mean) for year, mean in zip(yearly_means['year'], yearly_means[column])},
                "total_by_year": {str(year): float(total) for year, total in zip(yearly_sums['year'], yearly_sums[column])}
            }
            
            # Calculate trend
            if len(yearly_means) >= 3:
                from scipy import stats
                
                x = np.array(yearly_means['year'])
                y = np.array(yearly_means[column])
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
                trend_significance = "significant" if p_value < 0.05 else "not significant"
                
                yearly_trends[column]["trend"] = {
                    "slope": float(slope),
                    "r_squared": float(r_value ** 2),
                    "p_value": float(p_value),
                    "direction": trend_direction,
                    "significance": trend_significance
                }
        
        # Analyze monthly patterns
        monthly_means = data.groupby('month')[value_columns].mean().reset_index()
        
        monthly_trends = {}
        for column in value_columns:
            monthly_trends[column] = {
                "mean_by_month": {str(month): float(mean) for month, mean in zip(monthly_means['month'], monthly_means[column])}
            }
            
            # Identify seasonal patterns
            max_month = monthly_means.loc[monthly_means[column].idxmax()]['month']
            min_month = monthly_means.loc[monthly_means[column].idxmin()]['month']
            
            season_mapping = {
                1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
                6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall",
                11: "Fall", 12: "Winter"
            }
            
            max_season = season_mapping[max_month]
            min_season = season_mapping[min_month]
            
            monthly_trends[column]["seasonal_pattern"] = {
                "max_month": int(max_month),
                "min_month": int(min_month),
                "max_season": max_season,
                "min_season": min_season
            }
        
        time_analysis = {
            "yearly_trends": yearly_trends,
            "monthly_trends": monthly_trends
        }
        
        # Add specific metrics based on analysis type
        if analysis_type == "emissions":
            # Calculate year-over-year change
            if len(yearly_sums) >= 2:
                years = sorted(yearly_sums['year'].unique())
                yoy_change = {}
                
                for i in range(1, len(years)):
                    prev_year = years[i-1]
                    curr_year = years[i]
                    
                    yoy_change[f"{prev_year}_{curr_year}"] = {}
                    
                    for column in value_columns:
                        prev_val = yearly_sums[yearly_sums['year'] == prev_year][column].iloc[0]
                        curr_val = yearly_sums[yearly_sums['year'] == curr_year][column].iloc[0]
                        
                        if prev_val != 0:
                            percent_change = (curr_val - prev_val) / prev_val * 100
                        else:
                            percent_change = float('inf') if curr_val > 0 else 0
                        
                        yoy_change[f"{prev_year}_{curr_year}"][column] = {
                            "absolute_change": float(curr_val - prev_val),
                            "percent_change": float(percent_change)
                        }
                
                time_analysis["year_over_year_change"] = yoy_change
        
        elif analysis_type == "air_quality":
            # Calculate days exceeding standards
            if 'day' not in data.columns and pd.api.types.is_datetime64_any_dtype(data[time_column]):
                data['day'] = data[time_column].dt.date
            
            # Simplified threshold values for demonstration
            thresholds = {
                "pm25": 35,  # μg/m³
                "pm10": 150,  # μg/m³
                "o3": 70,     # ppb
                "co": 9,      # ppm
                "so2": 75,    # ppb
                "no2": 100    # ppb
            }
            
            exceeding_days = {}
            
            for column in value_columns:
                threshold = None
                
                # Find matching threshold
                for key in thresholds:
                    if key in column.lower():
                        threshold = thresholds[key]
                        break
                
                if threshold is not None:
                    # Count days exceeding threshold
                    if 'day' in data.columns:
                        days_df = data.groupby('day')[column].max().reset_index()
                        exceeding_count = (days_df[column] > threshold).sum()
                        total_days = len(days_df)
                        
                        exceeding_days[column] = {
                            "threshold": float(threshold),
                            "days_exceeding": int(exceeding_count),
                            "total_days": int(total_days),
                            "percentage": float(exceeding_count / total_days * 100) if total_days > 0 else 0
                        }
            
            if exceeding_days:
                time_analysis["days_exceeding_standards"] = exceeding_days
        
        return time_analysis
    
    def _identify_pollution_hotspots(self, data: pd.DataFrame, pollutant_columns: List[str], 
                                   location_column: str) -> Dict[str, Any]:
        """
        Identify locations with consistently high pollution levels.
        
        Args:
            data (pd.DataFrame): Pollution data
            pollutant_columns (List[str]): Columns containing pollutant measurements
            location_column (str): Column containing location information
            
        Returns:
            Dict[str, Any]: Pollution hotspots analysis
        """
        # Calculate average pollution level for each location
        location_means = data.groupby(location_column)[pollutant_columns].mean().reset_index()
        
        hotspots = {}
        
        for pollutant in pollutant_columns:
            # Calculate mean and standard deviation across all locations
            overall_mean = location_means[pollutant].mean()
            overall_std = location_means[pollutant].std()
            
            # Identify hotspots (locations with pollution > mean + 1.5*std)
            threshold = overall_mean + 1.5 * overall_std
            hotspot_locations = location_means[location_means[pollutant] > threshold]
            
            if not hotspot_locations.empty:
                hotspots[pollutant] = {
                    "threshold": float(threshold),
                    "overall_mean": float(overall_mean),
                    "locations": {
                        str(row[location_column]): {
                            "value": float(row[pollutant]),
                            "percent_above_mean": float((row[pollutant] - overall_mean) / overall_mean * 100) if overall_mean > 0 else float('inf')
                        }
                        for _, row in hotspot_locations.iterrows()
                    }
                }
        
        # Calculate overall hotspot score for each location
        location_hotspot_scores = {}
        
        for location in data[location_column].unique():
            score = 0
            for pollutant in hotspots:
                if str(location) in hotspots[pollutant]["locations"]:
                    # Add to score based on how much above threshold
                    percent_above = hotspots[pollutant]["locations"][str(location)]["percent_above_mean"]
                    score += min(5, percent_above / 20)  # Cap at 5 points per pollutant
            
            if score > 0:
                location_hotspot_scores[str(location)] = float(score)
        
        # Rank hotspots
        ranked_hotspots = sorted(location_hotspot_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "pollutant_hotspots": hotspots,
            "overall_hotspot_ranking": {location: score for location, score in ranked_hotspots[:10]}
        }
    
    def _analyze_emissions_by_source(self, data: pd.DataFrame, emission_columns: List[str], 
                                  source_column: str) -> Dict[str, Any]:
        """
        Analyze emissions by source.
        
        Args:
            data (pd.DataFrame): Emissions data
            emission_columns (List[str]): Columns containing emission measurements
            source_column (str): Column containing emission source information
            
        Returns:
            Dict[str, Any]: Emissions analysis by source
        """
        # Group by source and calculate sum for each emission type
        source_emissions = data.groupby(source_column)[emission_columns].sum().reset_index()
        
        # Calculate total emissions across all sources
        total_emissions = {col: data[col].sum() for col in emission_columns}
        total_all_types = sum(total_emissions.values())
        
        source_analysis = {}
        
        for _, row in source_emissions.iterrows():
            source = row[source_column]
            source_total = sum(row[col] for col in emission_columns)
            
            source_analysis[str(source)] = {
                "emissions": {col: float(row[col]) for col in emission_columns},
                "total": float(source_total),
                "percentage_of_total": float(source_total / total_all_types * 100) if total_all_types > 0 else 0
            }
        
        # Rank sources by total emissions
        ranked_sources = sorted(source_analysis.items(), key=lambda x: x[1]["total"], reverse=True)
        top_sources = {k: v for k, v in ranked_sources[:5]}
        
        # Calculate percentage by emission type
        emission_by_type = {
            col: {
                "total": float(total_emissions[col]),
                "percentage": float(total_emissions[col] / total_all_types * 100) if total_all_types > 0 else 0
            }
            for col in emission_columns
        }
        
        return {
            "by_source": source_analysis,
            "top_sources": top_sources,
            "by_emission_type": emission_by_type,
            "total_emissions": float(total_all_types)
        }
    
    def _calculate_carbon_metrics(self, data: pd.DataFrame, emission_columns: List[str]) -> Dict[str, Any]:
        """
        Calculate carbon footprint and related metrics.
        
        Args:
            data (pd.DataFrame): Emissions data
            emission_columns (List[str]): Columns containing emission measurements
            
        Returns:
            Dict[str, Any]: Carbon metrics
        """
        # Calculate total CO2-equivalent emissions
        total_emissions = sum(data[col].sum() for col in emission_columns)
        
        # Calculate carbon intensity and footprint metrics
        carbon_metrics = {
            "total_emissions": float(total_emissions)
        }
        
        # Calculate per capita emissions if population data available
        if "population" in data.columns:
            total_population = data["population"].sum()
            per_capita = total_emissions / total_population if total_population > 0 else 0
            
            carbon_metrics["per_capita"] = {
                "value": float(per_capita),
                "population": float(total_population)
            }
        
        # Calculate emissions per GDP if economic data available
        if "gdp" in data.columns:
            total_gdp = data["gdp"].sum()
            emissions_per_gdp = total_emissions / total_gdp if total_gdp > 0 else 0
            
            carbon_metrics["per_gdp"] = {
                "value": float(emissions_per_gdp),
                "gdp": float(total_gdp)
            }
        
        # Calculate emissions per area if area data available
        if "area" in data.columns:
            total_area = data["area"].sum()
            emissions_per_area = total_emissions / total_area if total_area > 0 else 0
            
            carbon_metrics["per_area"] = {
                "value": float(emissions_per_area),
                "area": float(total_area)
            }
        
        # Calculate carbon score (simplified example)
        # In reality, this would use more complex benchmarking
        carbon_score = 0
        
        # Example benchmark
        benchmark_emissions = 1000000  # Example benchmark
        
        if total_emissions < benchmark_emissions * 0.5:
            carbon_score = 5  # Excellent
        elif total_emissions < benchmark_emissions * 0.8:
            carbon_score = 4  # Good
        elif total_emissions < benchmark_emissions:
            carbon_score = 3  # Average
        elif total_emissions < benchmark_emissions * 1.5:
            carbon_score = 2  # Below average
        else:
            carbon_score = 1  # Poor
        
        carbon_metrics["carbon_score"] = carbon_score
        carbon_metrics["benchmark_comparison"] = float(total_emissions / benchmark_emissions) if benchmark_emissions > 0 else 0
        
        return carbon_metrics
    
    def _calculate_shannon_index(self, data: pd.DataFrame, species_column: str, count_column: str) -> float:
        """
        Calculate Shannon Diversity Index.
        
        Args:
            data (pd.DataFrame): Biodiversity data
            species_column (str): Column containing species information
            count_column (str): Column containing species count
            
        Returns:
            float: Shannon Diversity Index
        """
        # Group by species and sum counts
        species_counts = data.groupby(species_column)[count_column].sum()
        
        # Calculate total individuals
        total_count = species_counts.sum()
        
        # Calculate proportions
        proportions = species_counts / total_count
        
        # Calculate Shannon index: -sum(p_i * ln(p_i))
        shannon_index = -np.sum(proportions * np.log(proportions))
        
        return shannon_index
    
    def _calculate_simpson_index(self, data: pd.DataFrame, species_column: str, count_column: str) -> float:
        """
        Calculate Simpson Diversity Index.
        
        Args:
            data (pd.DataFrame): Biodiversity data
            species_column (str): Column containing species information
            count_column (str): Column containing species count
            
        Returns:
            float: Simpson Diversity Index
        """
        # Group by species and sum counts
        species_counts = data.groupby(species_column)[count_column].sum()
        
        # Calculate total individuals
        total_count = species_counts.sum()
        
        # Calculate proportions
        proportions = species_counts / total_count
        
        # Calculate Simpson index: 1 - sum(p_i^2)
        simpson_index = 1 - np.sum(proportions ** 2)
        
        return simpson_index
    
    def _get_species_distribution(self, data: pd.DataFrame, species_column: str, count_column: str) -> Dict[str, Any]:
        """
        Get species distribution.
        
        Args:
            data (pd.DataFrame): Biodiversity data
            species_column (str): Column containing species information
            count_column (str): Column containing species count
            
        Returns:
            Dict[str, Any]: Species distribution
        """
        # Group by species and sum counts
        species_counts = data.groupby(species_column)[count_column].sum().reset_index()
        
        # Calculate total individuals
        total_count = species_counts[count_column].sum()
        
        # Calculate proportions
        species_counts['proportion'] = species_counts[count_column] / total_count
        
        # Sort by count descending
        species_counts = species_counts.sort_values(by=count_column, ascending=False)
        
        # Convert to dictionary
        distribution = {
            str(row[species_column]): {
                "count": float(row[count_column]),
                "proportion": float(row['proportion'])
            }
            for _, row in species_counts.iterrows()
        }
        
        # Get dominant species (top 5)
        dominant_species = {k: v for k, v in list(distribution.items())[:5]}
        
        return {
            "all_species": distribution,
            "dominant_species": dominant_species,
            "total_individuals": float(total_count)
        }
    
    def _analyze_biodiversity_by_habitat(self, data: pd.DataFrame, species_column: str, 
                                       count_column: str, habitat_column: str) -> Dict[str, Any]:
        """
        Analyze biodiversity by habitat.
        
        Args:
            data (pd.DataFrame): Biodiversity data
            species_column (str): Column containing species information
            count_column (str): Column containing species count
            habitat_column (str): Column containing habitat information
            
        Returns:
            Dict[str, Any]: Biodiversity analysis by habitat
        """
        habitat_analysis = {}
        
        for habitat in data[habitat_column].unique():
            habitat_data = data[data[habitat_column] == habitat]
            
            # Calculate diversity indices for this habitat
            shannon_index = self._calculate_shannon_index(habitat_data, species_column, count_column)
            simpson_index = self._calculate_simpson_index(habitat_data, species_column, count_column)
            
            # Count species in this habitat
            species_in_habitat = habitat_data[species_column].nunique()
            total_individuals = habitat_data[count_column].sum()
            
            habitat_analysis[str(habitat)] = {
                "species_count": species_in_habitat,
                "total_individuals": float(total_individuals),
                "shannon_diversity_index": float(shannon_index),
                "simpson_diversity_index": float(simpson_index),
                "species_distribution": self._get_species_distribution(habitat_data, species_column, count_column)
            }
        
        # Calculate habitat with highest diversity
        max_shannon_habitat = max(habitat_analysis.items(), key=lambda x: x[1]["shannon_diversity_index"])
        max_simpson_habitat = max(habitat_analysis.items(), key=lambda x: x[1]["simpson_diversity_index"])
        max_species_habitat = max(habitat_analysis.items(), key=lambda x: x[1]["species_count"])
        
        return {
            "by_habitat": habitat_analysis,
            "highest_shannon_diversity": {
                "habitat": max_shannon_habitat[0],
                "value": float(max_shannon_habitat[1]["shannon_diversity_index"])
            },
            "highest_simpson_diversity": {
                "habitat": max_simpson_habitat[0],
                "value": float(max_simpson_habitat[1]["simpson_diversity_index"])
            },
            "highest_species_count": {
                "habitat": max_species_habitat[0],
                "value": max_species_habitat[1]["species_count"]
            }
        }
    
    def _analyze_biodiversity_by_location(self, data: pd.DataFrame, species_column: str, 
                                        count_column: str, location_column: str) -> Dict[str, Any]:
        """
        Analyze biodiversity by location.
        
        Args:
            data (pd.DataFrame): Biodiversity data
            species_column (str): Column containing species information
            count_column (str): Column containing species count
            location_column (str): Column containing location information
            
        Returns:
            Dict[str, Any]: Biodiversity analysis by location
        """
        location_analysis = {}
        
        for location in data[location_column].unique():
            location_data = data[data[location_column] == location]
            
            # Calculate diversity indices for this location
            shannon_index = self._calculate_shannon_index(location_data, species_column, count_column)
            simpson_index = self._calculate_simpson_index(location_data, species_column, count_column)
            
            # Count species in this location
            species_in_location = location_data[species_column].nunique()
            total_individuals = location_data[count_column].sum()
            
            location_analysis[str(location)] = {
                "species_count": species_in_location,
                "total_individuals": float(total_individuals),
                "shannon_diversity_index": float(shannon_index),
                "simpson_diversity_index": float(simpson_index),
                "dominant_species": self._get_dominant_species(location_data, species_column, count_column)
            }
        
        # Calculate location with highest diversity
        max_shannon_location = max(location_analysis.items(), key=lambda x: x[1]["shannon_diversity_index"])
        max_simpson_location = max(location_analysis.items(), key=lambda x: x[1]["simpson_diversity_index"])
        max_species_location = max(location_analysis.items(), key=lambda x: x[1]["species_count"])
        
        return {
            "by_location": location_analysis,
            "highest_shannon_diversity": {
                "location": max_shannon_location[0],
                "value": float(max_shannon_location[1]["shannon_diversity_index"])
            },
            "highest_simpson_diversity": {
                "location": max_simpson_location[0],
                "value": float(max_simpson_location[1]["simpson_diversity_index"])
            },
            "highest_species_count": {
                "location": max_species_location[0],
                "value": max_species_location[1]["species_count"]
            }
        }
    
    def _get_dominant_species(self, data: pd.DataFrame, species_column: str, count_column: str) -> Dict[str, float]:
        """
        Get dominant species (top 3) for a subset of data.
        
        Args:
            data (pd.DataFrame): Biodiversity data
            species_column (str): Column containing species information
            count_column (str): Column containing species count
            
        Returns:
            Dict[str, float]: Dominant species with their counts
        """
        # Group by species and sum counts
        species_counts = data.groupby(species_column)[count_column].sum().reset_index()
        
        # Sort by count descending
        species_counts = species_counts.sort_values(by=count_column, ascending=False)
        
        # Take top 3
        dominant = {
            str(row[species_column]): float(row[count_column])
            for _, row in species_counts.head(3).iterrows()
        }
        
        return dominant
    
    def _analyze_biodiversity_time_patterns(self, data: pd.DataFrame, species_column: str, 
                                          count_column: str, time_column: str) -> Dict[str, Any]:
        """
        Analyze biodiversity patterns over time.
        
        Args:
            data (pd.DataFrame): Biodiversity data
            species_column (str): Column containing species information
            count_column (str): Column containing species count
            time_column (str): Column containing time information
            
        Returns:
            Dict[str, Any]: Biodiversity time pattern analysis
        """
        # Ensure time column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
            try:
                data[time_column] = pd.to_datetime(data[time_column])
            except:
                logger.warning(f"Could not convert {time_column} to datetime format")
                return {"status": "error", "message": "Invalid time format"}
        
        # Create year and month columns
        data['year'] = data[time_column].dt.year
        data['month'] = data[time_column].dt.month
        
        # Analyze yearly biodiversity
        yearly_diversity = {}
        
        for year in data['year'].unique():
            year_data = data[data['year'] == year]
            
            shannon_index = self._calculate_shannon_index(year_data, species_column, count_column)
            simpson_index = self._calculate_simpson_index(year_data, species_column, count_column)
            species_count = year_data[species_column].nunique()
            
            yearly_diversity[str(year)] = {
                "species_count": species_count,
                "shannon_diversity_index": float(shannon_index),
                "simpson_diversity_index": float(simpson_index),
                "total_individuals": float(year_data[count_column].sum())
            }
        
        # Analyze seasonal biodiversity
        monthly_diversity = {}
        
        for month in range(1, 13):
            if month in data['month'].unique():
                month_data = data[data['month'] == month]
                
                shannon_index = self._calculate_shannon_index(month_data, species_column, count_column)
                simpson_index = self._calculate_simpson_index(month_data, species_column, count_column)
                species_count = month_data[species_column].nunique()
                
                monthly_diversity[str(month)] = {
                    "species_count": species_count,
                    "shannon_diversity_index": float(shannon_index),
                    "simpson_diversity_index": float(simpson_index),
                    "total_individuals": float(month_data[count_column].sum())
                }
        
        # Calculate biodiversity trends
        trends = {}
        
        if len(yearly_diversity) >= 3:
            # Analyze trends in species count
            years = sorted([int(y) for y in yearly_diversity.keys()])
            species_counts = [yearly_diversity[str(y)]["species_count"] for y in years]
            
            from scipy import stats
            
            # Calculate trend for species count
            slope, intercept, r_value, p_value, std_err = stats.linregress(years, species_counts)
            
            trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            trend_significance = "significant" if p_value < 0.05 else "not significant"
            
            trends["species_count"] = {
                "slope": float(slope),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "direction": trend_direction,
                "significance": trend_significance
            }
            
            # Calculate trend for Shannon diversity
            shannon_values = [yearly_diversity[str(y)]["shannon_diversity_index"] for y in years]
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(years, shannon_values)
            
            trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            trend_significance = "significant" if p_value < 0.05 else "not significant"
            
            trends["shannon_diversity"] = {
                "slope": float(slope),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "direction": trend_direction,
                "significance": trend_significance
            }
        
        return {
            "yearly_diversity": yearly_diversity,
            "monthly_diversity": monthly_diversity,
            "trends": trends
        }
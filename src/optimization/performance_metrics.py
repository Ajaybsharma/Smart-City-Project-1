import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional
import logging
import datetime
import json
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceMetricsCalculator:
    """Measures optimization effectiveness using various performance metrics."""
    
    def __init__(self, metrics_dir: str = "performance_metrics"):
        """
        Initialize the PerformanceMetricsCalculator.
        
        Args:
            metrics_dir (str): Directory to save performance metrics
        """
        self.metrics_dir = metrics_dir
        
        # Create metrics directory if it doesn't exist
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        logger.info(f"Initialized PerformanceMetricsCalculator with metrics directory: {self.metrics_dir}")
    
    def calculate_sustainability_metrics(self, data: pd.DataFrame, metric_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate sustainability performance metrics based on configuration.
        
        Args:
            data (pd.DataFrame): Input data
            metric_config (Dict[str, Any]): Configuration specifying which metrics to calculate
            
        Returns:
            Dict[str, Any]: Calculated sustainability metrics
        """
        if data.empty:
            logger.warning("Empty dataset provided for sustainability metrics calculation")
            return {"status": "error", "message": "Empty dataset"}
        
        logger.info("Calculating sustainability performance metrics")
        
        try:
            metrics_result = {
                "status": "success",
                "calculation_time": datetime.datetime.now().isoformat(),
                "metrics": {}
            }
            
            # Process water metrics if configured
            if metric_config.get("water_metrics", False):
                water_config = metric_config.get("water_config", {})
                water_metrics = self._calculate_water_metrics(data, water_config)
                metrics_result["metrics"]["water"] = water_metrics
            
            # Process waste metrics if configured
            if metric_config.get("waste_metrics", False):
                waste_config = metric_config.get("waste_config", {})
                waste_metrics = self._calculate_waste_metrics(data, waste_config)
                metrics_result["metrics"]["waste"] = waste_metrics
            
            # Process energy metrics if configured
            if metric_config.get("energy_metrics", False):
                energy_config = metric_config.get("energy_config", {})
                energy_metrics = self._calculate_energy_metrics(data, energy_config)
                metrics_result["metrics"]["energy"] = energy_metrics
            
            # Process emissions metrics if configured
            if metric_config.get("emissions_metrics", False):
                emissions_config = metric_config.get("emissions_config", {})
                emissions_metrics = self._calculate_emissions_metrics(data, emissions_config)
                metrics_result["metrics"]["emissions"] = emissions_metrics
            
            # Process biodiversity metrics if configured
            if metric_config.get("biodiversity_metrics", False):
                biodiversity_config = metric_config.get("biodiversity_config", {})
                biodiversity_metrics = self._calculate_biodiversity_metrics(data, biodiversity_config)
                metrics_result["metrics"]["biodiversity"] = biodiversity_metrics
            
            # Calculate overall sustainability score
            sustainability_score = self._calculate_overall_score(metrics_result["metrics"])
            metrics_result["overall_sustainability_score"] = sustainability_score
            
            # Save metrics to file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = os.path.join(self.metrics_dir, f"sustainability_metrics_{timestamp}.json")
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_result, f, indent=2)
            
            logger.info(f"Sustainability metrics calculated and saved to {metrics_file}")
            
            return metrics_result
            
        except Exception as e:
            logger.error(f"Error calculating sustainability metrics: {e}")
            return {"status": "error", "message": str(e)}
    
    def calculate_optimization_impact(self, baseline_data: pd.DataFrame, optimized_data: pd.DataFrame, 
                                    impact_metrics: List[str]) -> Dict[str, Any]:
        """
        Calculate the impact of optimization by comparing baseline and optimized data.
        
        Args:
            baseline_data (pd.DataFrame): Baseline data before optimization
            optimized_data (pd.DataFrame): Data after applying optimization
            impact_metrics (List[str]): Metrics to measure optimization impact
            
        Returns:
            Dict[str, Any]: Optimization impact results
        """
        if baseline_data.empty or optimized_data.empty:
            logger.warning("Empty dataset provided for optimization impact calculation")
            return {"status": "error", "message": "Empty dataset"}
        
        logger.info("Calculating optimization impact")
        
        try:
            impact_result = {
                "status": "success",
                "calculation_time": datetime.datetime.now().isoformat(),
                "impact": {}
            }
            
            # Calculate impact for each specified metric
            for metric in impact_metrics:
                if metric not in baseline_data.columns or metric not in optimized_data.columns:
                    logger.warning(f"Metric '{metric}' not found in data")
                    continue
                
                baseline_value = baseline_data[metric].mean()
                optimized_value = optimized_data[metric].mean()
                
                absolute_change = optimized_value - baseline_value
                percent_change = (absolute_change / baseline_value * 100) if baseline_value != 0 else 0
                
                impact_result["impact"][metric] = {
                    "baseline_value": float(baseline_value),
                    "optimized_value": float(optimized_value),
                    "absolute_change": float(absolute_change),
                    "percent_change": float(percent_change)
                }
            
            # Calculate overall impact score
            overall_score = self._calculate_impact_score(impact_result["impact"])
            impact_result["overall_impact_score"] = overall_score
            
            # Save impact results to file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            impact_file = os.path.join(self.metrics_dir, f"optimization_impact_{timestamp}.json")
            
            with open(impact_file, 'w') as f:
                json.dump(impact_result, f, indent=2)
            
            logger.info(f"Optimization impact calculated and saved to {impact_file}")
            
            return impact_result
            
        except Exception as e:
            logger.error(f"Error calculating optimization impact: {e}")
            return {"status": "error", "message": str(e)}
    
    def _calculate_water_metrics(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate water-related sustainability metrics.
        
        Args:
            data (pd.DataFrame): Input data
            config (Dict[str, Any]): Configuration for water metrics
            
        Returns:
            Dict[str, Any]: Water sustainability metrics
        """
        metrics = {}
        
        # Water consumption metrics
        if "consumption_column" in config:
            consumption_col = config["consumption_column"]
            if consumption_col in data.columns:
                total_consumption = data[consumption_col].sum()
                avg_consumption = data[consumption_col].mean()
                
                metrics["consumption"] = {
                    "total": float(total_consumption),
                    "average": float(avg_consumption)
                }
                
                # Calculate per capita metrics if population data available
                if "population_column" in config and config["population_column"] in data.columns:
                    pop_col = config["population_column"]
                    total_population = data[pop_col].sum()
                    per_capita = total_consumption / total_population if total_population > 0 else 0
                    
                    metrics["consumption"]["per_capita"] = float(per_capita)
                    metrics["consumption"]["population"] = float(total_population)
                
                # Calculate efficiency
                if "area_column" in config and config["area_column"] in data.columns:
                    area_col = config["area_column"]
                    total_area = data[area_col].sum()
                    consumption_per_area = total_consumption / total_area if total_area > 0 else 0
                    
                    metrics["consumption"]["per_area"] = float(consumption_per_area)
                    metrics["consumption"]["total_area"] = float(total_area)
        
        # Water quality metrics
        if "quality_columns" in config:
            quality_metrics = {}
            
            for col in config["quality_columns"]:
                if col in data.columns:
                    quality_metrics[col] = {
                        "average": float(data[col].mean()),
                        "min": float(data[col].min()),
                        "max": float(data[col].max())
                    }
            
            if quality_metrics:
                metrics["quality"] = quality_metrics
        
        # Water availability metrics
        if "supply_column" in config and "demand_column" in config:
            supply_col = config["supply_column"]
            demand_col = config["demand_column"]
            
            if supply_col in data.columns and demand_col in data.columns:
                total_supply = data[supply_col].sum()
                total_demand = data[demand_col].sum()
                
                supply_demand_ratio = total_supply / total_demand if total_demand > 0 else float('inf')
                stress_level = "Low" if supply_demand_ratio > 1.5 else "Moderate" if supply_demand_ratio > 1.0 else "High" if supply_demand_ratio > 0.8 else "Extreme"
                
                metrics["availability"] = {
                    "total_supply": float(total_supply),
                    "total_demand": float(total_demand),
                    "supply_demand_ratio": float(supply_demand_ratio),
                    "water_stress_level": stress_level
                }
        
        # Calculate water sustainability score (simplified model)
        if metrics:
            water_score = 0
            num_factors = 0
            
            if "consumption" in metrics:
                # Lower consumption is better
                consumption_score = 0
                
                if "benchmark" in config:
                    benchmark = config["benchmark"]
                    avg_consumption = metrics["consumption"]["average"]
                    
                    if avg_consumption < benchmark * 0.7:
                        consumption_score = 5  # Excellent
                    elif avg_consumption < benchmark * 0.9:
                        consumption_score = 4  # Good
                    elif avg_consumption < benchmark * 1.1:
                        consumption_score = 3  # Average
                    elif avg_consumption < benchmark * 1.3:
                        consumption_score = 2  # Below average
                    else:
                        consumption_score = 1  # Poor
                else:
                    consumption_score = 3  # Default without benchmark
                
                water_score += consumption_score
                num_factors += 1
            
            if "availability" in metrics:
                # Higher supply/demand ratio is better
                availability_score = 0
                ratio = metrics["availability"]["supply_demand_ratio"]
                
                if ratio > 1.5:
                    availability_score = 5  # Excellent
                elif ratio > 1.2:
                    availability_score = 4  # Good
                elif ratio > 1.0:
                    availability_score = 3  # Average
                elif ratio > 0.8:
                    availability_score = 2  # Below average
                else:
                    availability_score = 1  # Poor
                
                water_score += availability_score
                num_factors += 1
            
            if "quality" in metrics and config.get("quality_benchmark"):
                quality_score = 0
                quality_benchmark = config["quality_benchmark"]
                
                # Check each quality metric against benchmark
                num_quality_metrics = 0
                total_quality_score = 0
                
                for metric, values in metrics["quality"].items():
                    if metric in quality_benchmark:
                        benchmark = quality_benchmark[metric]
                        avg_value = values["average"]
                        
                        # Assume lower is better for contaminants
                        if avg_value < benchmark * 0.5:
                            metric_score = 5
                        elif avg_value < benchmark * 0.8:
                            metric_score = 4
                        elif avg_value < benchmark:
                            metric_score = 3
                        elif avg_value < benchmark * 1.5:
                            metric_score = 2
                        else:
                            metric_score = 1
                        
                        total_quality_score += metric_score
                        num_quality_metrics += 1
                
                if num_quality_metrics > 0:
                    quality_score = total_quality_score / num_quality_metrics
                    water_score += quality_score
                    num_factors += 1
            
            # Calculate overall water score
            overall_water_score = water_score / num_factors if num_factors > 0 else 0
            metrics["sustainability_score"] = float(overall_water_score)
        
        return metrics
    
    def _calculate_waste_metrics(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate waste-related sustainability metrics.
        
        Args:
            data (pd.DataFrame): Input data
            config (Dict[str, Any]): Configuration for waste metrics
            
        Returns:
            Dict[str, Any]: Waste sustainability metrics
        """
        metrics = {}
        
        # Waste generation metrics
        if "waste_column" in config:
            waste_col = config["waste_column"]
            if waste_col in data.columns:
                total_waste = data[waste_col].sum()
                avg_waste = data[waste_col].mean()
                
                metrics["generation"] = {
                    "total": float(total_waste),
                    "average": float(avg_waste)
                }
                
                # Calculate per capita metrics if population data available
                if "population_column" in config and config["population_column"] in data.columns:
                    pop_col = config["population_column"]
                    total_population = data[pop_col].sum()
                    per_capita = total_waste / total_population if total_population > 0 else 0
                    
                    metrics["generation"]["per_capita"] = float(per_capita)
                    metrics["generation"]["population"] = float(total_population)
        
        # Recycling metrics
        if "recycled_column" in config and "waste_column" in config:
            recycled_col = config["recycled_column"]
            waste_col = config["waste_column"]
            
            if recycled_col in data.columns and waste_col in data.columns:
                total_recycled = data[recycled_col].sum()
                total_waste = data[waste_col].sum()
                
                recycling_rate = total_recycled / total_waste * 100 if total_waste > 0 else 0
                
                metrics["recycling"] = {
                    "total_recycled": float(total_recycled),
                    "recycling_rate": float(recycling_rate)
                }
        
        # Waste composition metrics
        if "composition_columns" in config:
            composition = {}
            
            for col in config["composition_columns"]:
                if col in data.columns:
                    composition[col] = float(data[col].sum())
            
            if composition:
                metrics["composition"] = composition
        
        # Calculate waste sustainability score
        if metrics:
            waste_score = 0
            num_factors = 0
            
            if "generation" in metrics:
                # Lower waste generation is better
                generation_score = 0
                
                if "benchmark" in config:
                    benchmark = config["benchmark"]
                    avg_waste = metrics["generation"]["average"]
                    
                    if avg_waste < benchmark * 0.7:
                        generation_score = 5
                    elif avg_waste < benchmark * 0.9:
                        generation_score = 4
                    elif avg_waste < benchmark * 1.1:
                        generation_score = 3
                    elif avg_waste < benchmark * 1.3:
                        generation_score = 2
                    else:
                        generation_score = 1
                else:
                    generation_score = 3  # Default without benchmark
                
                waste_score += generation_score
                num_factors += 1
            
            if "recycling" in metrics:
                # Higher recycling rate is better
                recycling_score = 0
                rate = metrics["recycling"]["recycling_rate"]
                
                if rate >= 70:
                    recycling_score = 5
                elif rate >= 50:
                    recycling_score = 4
                elif rate >= 30:
                    recycling_score = 3
                elif rate >= 10:
                    recycling_score = 2
                else:
                    recycling_score = 1
                
                waste_score += recycling_score
                num_factors += 1
            
            # Calculate overall waste score
            overall_waste_score = waste_score / num_factors if num_factors > 0 else 0
            metrics["sustainability_score"] = float(overall_waste_score)
        
        return metrics
    
    def _calculate_energy_metrics(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate energy-related sustainability metrics.
        
        Args:
            data (pd.DataFrame): Input data
            config (Dict[str, Any]): Configuration for energy metrics
            
        Returns:
            Dict[str, Any]: Energy sustainability metrics
        """
        metrics = {}
        
        # Energy consumption metrics
        if "consumption_column" in config:
            consumption_col = config["consumption_column"]
            if consumption_col in data.columns:
                total_consumption = data[consumption_col].sum()
                avg_consumption = data[consumption_col].mean()
                
                metrics["consumption"] = {
                    "total": float(total_consumption),
                    "average": float(avg_consumption)
                }
                
                # Calculate per capita metrics if population data available
                if "population_column" in config and config["population_column"] in data.columns:
                    pop_col = config["population_column"]
                    total_population = data[pop_col].sum()
                    per_capita = total_consumption / total_population if total_population > 0 else 0
                    
                    metrics["consumption"]["per_capita"] = float(per_capita)
                    metrics["consumption"]["population"] = float(total_population)
                
                # Calculate per area metrics if area data available
                if "area_column" in config and config["area_column"] in data.columns:
                    area_col = config["area_column"]
                    total_area = data[area_col].sum()
                    per_area = total_consumption / total_area if total_area > 0 else 0
                    
                    metrics["consumption"]["per_area"] = float(per_area)
                    metrics["consumption"]["total_area"] = float(total_area)
        
        # Renewable energy metrics
        if "renewable_column" in config and "consumption_column" in config:
            renewable_col = config["renewable_column"]
            consumption_col = config["consumption_column"]
            
            if renewable_col in data.columns and consumption_col in data.columns:
                total_renewable = data[renewable_col].sum()
                total_consumption = data[consumption_col].sum()
                
                renewable_percentage = total_renewable / total_consumption * 100 if total_consumption > 0 else 0
                
                metrics["renewable"] = {
                    "total": float(total_renewable),
                    "percentage": float(renewable_percentage)
                }
        
        # Energy source breakdown
        if "source_columns" in config:
            sources = {}
            total_energy = 0
            
            for col in config["source_columns"]:
                if col in data.columns:
                    source_total = data[col].sum()
                    sources[col] = float(source_total)
                    total_energy += source_total
            
            if sources and total_energy > 0:
                # Calculate percentages
                for source in sources:
                    percentage = sources[source] / total_energy * 100
                    sources[f"{source}_percentage"] = float(percentage)
                
                metrics["sources"] = sources
                metrics["sources"]["total"] = float(total_energy)
        
        # Calculate energy sustainability score
        if metrics:
            energy_score = 0
            num_factors = 0
            
            if "consumption" in metrics:
                # Lower consumption is better
                consumption_score = 0
                
                if "benchmark" in config:
                    benchmark = config["benchmark"]
                    avg_consumption = metrics["consumption"]["average"]
                    
                    if avg_consumption < benchmark * 0.7:
                        consumption_score = 5
                    elif avg_consumption < benchmark * 0.9:
                        consumption_score = 4
                    elif avg_consumption < benchmark * 1.1:
                        consumption_score = 3
                    elif avg_consumption < benchmark * 1.3:
                        consumption_score = 2
                    else:
                        consumption_score = 1
                else:
                    consumption_score = 3  # Default without benchmark
                
                energy_score += consumption_score
                num_factors += 1
            
            if "renewable" in metrics:
                # Higher renewable percentage is better
                renewable_score = 0
                percentage = metrics["renewable"]["percentage"]
                
                if percentage >= 80:
                    renewable_score = 5
                elif percentage >= 60:
                    renewable_score = 4
                elif percentage >= 40:
                    renewable_score = 3
                elif percentage >= 20:
                    renewable_score = 2
                else:
                    renewable_score = 1
                
                energy_score += renewable_score
                num_factors += 1
            
            # Calculate overall energy score
            overall_energy_score = energy_score / num_factors if num_factors > 0 else 0
            metrics["sustainability_score"] = float(overall_energy_score)
        
        return metrics
    
    def _calculate_emissions_metrics(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate emissions-related sustainability metrics.
        
        Args:
            data (pd.DataFrame): Input data
            config (Dict[str, Any]): Configuration for emissions metrics
            
        Returns:
            Dict[str, Any]: Emissions sustainability metrics
        """
        metrics = {}
        
        # Total emissions metrics
        if "emissions_columns" in config:
            emissions_data = {}
            total_emissions = 0
            
            for col in config["emissions_columns"]:
                if col in data.columns:
                    emissions_total = data[col].sum()
                    emissions_data[col] = float(emissions_total)
                    total_emissions += emissions_total
            
            if emissions_data:
                metrics["emissions"] = emissions_data
                metrics["emissions"]["total"] = float(total_emissions)
                
                # Calculate per capita metrics if population data available
                if "population_column" in config and config["population_column"] in data.columns:
                    pop_col = config["population_column"]
                    total_population = data[pop_col].sum()
                    per_capita = total_emissions / total_population if total_population > 0 else 0
                    
                    metrics["emissions"]["per_capita"] = float(per_capita)
                    metrics["emissions"]["population"] = float(total_population)
                
                # Calculate per GDP metrics if GDP data available
                if "gdp_column" in config and config["gdp_column"] in data.columns:
                    gdp_col = config["gdp_column"]
                    total_gdp = data[gdp_col].sum()
                    per_gdp = total_emissions / total_gdp if total_gdp > 0 else 0
                    
                    metrics["emissions"]["per_gdp"] = float(per_gdp)
                    metrics["emissions"]["total_gdp"] = float(total_gdp)
        
        # Emission reduction metrics
        if "baseline_emissions" in config and "emissions_columns" in config:
            baseline = config["baseline_emissions"]
            current_emissions = 0
            
            for col in config["emissions_columns"]:
                if col in data.columns:
                    current_emissions += data[col].sum()
            
            if current_emissions > 0 and baseline > 0:
                reduction_percentage = (baseline - current_emissions) / baseline * 100
                
                metrics["reduction"] = {
                    "baseline": float(baseline),
                    "current": float(current_emissions),
                    "absolute_reduction": float(baseline - current_emissions),
                    "percentage_reduction": float(reduction_percentage)
                }
        
        # Calculate emissions sustainability score
        if metrics:
            emissions_score = 0
            num_factors = 0
            
            if "emissions" in metrics and "benchmark" in config:
                # Lower emissions is better
                emissions_total = metrics["emissions"]["total"]
                benchmark = config["benchmark"]
                
                if emissions_total < benchmark * 0.6:
                    emissions_score = 5
                elif emissions_total < benchmark * 0.8:
                    emissions_score = 4
                elif emissions_total < benchmark:
                    emissions_score = 3
                elif emissions_total < benchmark * 1.2:
                    emissions_score = 2
                else:
                    emissions_score = 1
                
                num_factors += 1
            
            if "reduction" in metrics:
                # Higher reduction percentage is better
                reduction_score = 0
                percentage = metrics["reduction"]["percentage_reduction"]
                
                if percentage >= 40:
                    reduction_score = 5
                elif percentage >= 30:
                    reduction_score = 4
                elif percentage >= 20:
                    reduction_score = 3
                elif percentage >= 10:
                    reduction_score = 2
                else:
                    reduction_score = 1
                
                emissions_score += reduction_score
                num_factors += 1
            
            # Calculate overall emissions score
            overall_emissions_score = emissions_score / num_factors if num_factors > 0 else 0
            metrics["sustainability_score"] = float(overall_emissions_score)
        
        return metrics
    
    def _calculate_biodiversity_metrics(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate biodiversity-related sustainability metrics.
        
        Args:
            data (pd.DataFrame): Input data
            config (Dict[str, Any]): Configuration for biodiversity metrics
            
        Returns:
            Dict[str, Any]: Biodiversity sustainability metrics
        """
        metrics = {}
        
        # Species diversity metrics
        if "species_column" in config and "count_column" in config:
            species_col = config["species_column"]
            count_col = config["count_column"]
            
            if species_col in data.columns and count_col in data.columns:
                # Calculate number of species
                species_count = data[species_col].nunique()
                
                # Calculate Shannon diversity index
                species_counts = data.groupby(species_col)[count_col].sum()
                total_count = species_counts.sum()
                proportions = species_counts / total_count
                shannon_index = -np.sum(proportions * np.log(proportions))
                
                # Calculate Simpson diversity index
                simpson_index = 1 - np.sum(proportions ** 2)
                
                metrics["diversity"] = {
                    "species_count": int(species_count),
                    "shannon_index": float(shannon_index),
                    "simpson_index": float(simpson_index),
                    "total_individuals": float(total_count)
                }
        
        # Habitat metrics
        if "habitat_column" in config and "area_column" in config:
            habitat_col = config["habitat_column"]
            area_col = config["area_column"]
            
            if habitat_col in data.columns and area_col in data.columns:
                habitat_counts = data.groupby(habitat_col)[area_col].sum()
                total_area = habitat_counts.sum()
                
                # Calculate habitat diversity
                habitat_proportions = habitat_counts / total_area
                habitat_shannon = -np.sum(habitat_proportions * np.log(habitat_proportions))
                
                metrics["habitat"] = {
                    "habitat_count": int(len(habitat_counts)),
                    "habitat_diversity": float(habitat_shannon),
                    "total_area": float(total_area),
                    "habitat_areas": {
                        str(habitat): float(area)
                        for habitat, area in habitat_counts.items()
                    }
                }
        
        # Calculate biodiversity sustainability score
        if metrics:
            biodiversity_score = 0
            num_factors = 0
            
            if "diversity" in metrics:
                diversity_score = 0
                shannon_index = metrics["diversity"]["shannon_index"]
                
                # Shannon index typically ranges from 0 to ~4.5 for very diverse communities
                if shannon_index >= 3.5:
                    diversity_score = 5
                elif shannon_index >= 2.5:
                    diversity_score = 4
                elif shannon_index >= 1.5:
                    diversity_score = 3
                elif shannon_index >= 0.5:
                    diversity_score = 2
                else:
                    diversity_score = 1
                
                biodiversity_score += diversity_score
                num_factors += 1
            
            if "habitat" in metrics:
                habitat_score = 0
                habitat_count = metrics["habitat"]["habitat_count"]
                
                # More habitats generally mean higher biodiversity
                if habitat_count >= 8:
                    habitat_score = 5
                elif habitat_count >= 6:
                    habitat_score = 4
                elif habitat_count >= 4:
                    habitat_score = 3
                elif habitat_count >= 2:
                    habitat_score = 2
                else:
                    habitat_score = 1
                
                biodiversity_score += habitat_score
                num_factors += 1
            
            # Calculate overall biodiversity score
            overall_biodiversity_score = biodiversity_score / num_factors if num_factors > 0 else 0
            metrics["sustainability_score"] = float(overall_biodiversity_score)
        
        return metrics
    
    def _calculate_overall_score(self, metrics_categories: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall sustainability score from individual metrics.
        
        Args:
            metrics_categories (Dict[str, Any]): Dictionary of metrics by category
            
        Returns:
            Dict[str, Any]: Overall sustainability score and breakdown
        """
        category_scores = {}
        overall_score = 0
        num_categories = 0
        
        # Extract scores from each category
        for category, metrics in metrics_categories.items():
            if "sustainability_score" in metrics:
                score = metrics["sustainability_score"]
                category_scores[category] = score
                overall_score += score
                num_categories += 1
        
        # Calculate overall average
        average_score = overall_score / num_categories if num_categories > 0 else 0
        
        # Determine rating
        if average_score >= 4.5:
            rating = "Excellent"
        elif average_score >= 3.5:
            rating = "Good"
        elif average_score >= 2.5:
            rating = "Average"
        elif average_score >= 1.5:
            rating = "Below Average"
        else:
            rating = "Poor"
        
        return {
            "overall_score": float(average_score),
            "rating": rating,
            "category_scores": category_scores,
            "categories_evaluated": num_categories
        }
    
    def _calculate_impact_score(self, impact_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall impact score from impact metrics.
        
        Args:
            impact_metrics (Dict[str, Any]): Dictionary of impact metrics
            
        Returns:
            Dict[str, Any]: Overall impact score and assessment
        """
        # Count positive and negative impacts
        positive_impacts = 0
        negative_impacts = 0
        neutral_impacts = 0
        
        # Sum of percentage changes (signed)
        total_percent_change = 0
        num_metrics = len(impact_metrics)
        
        for metric, values in impact_metrics.items():
            percent_change = values["percent_change"]
            
            if percent_change > 5:
                positive_impacts += 1
            elif percent_change < -5:
                negative_impacts += 1
            else:
                neutral_impacts += 1
            
            total_percent_change += percent_change
        
        # Calculate average percentage change
        avg_percent_change = total_percent_change / num_metrics if num_metrics > 0 else 0
        
        # Determine impact level
        if avg_percent_change >= 20:
            impact_level = "Transformative"
        elif avg_percent_change >= 10:
            impact_level = "Significant"
        elif avg_percent_change >= 5:
            impact_level = "Moderate"
        elif avg_percent_change >= 1:
            impact_level = "Slight"
        elif avg_percent_change > -1:
            impact_level = "Neutral"
        elif avg_percent_change > -5:
            impact_level = "Slightly Negative"
        else:
            impact_level = "Negative"
        
        return {
            "average_percent_change": float(avg_percent_change),
            "impact_level": impact_level,
            "positive_impacts": positive_impacts,
            "negative_impacts": negative_impacts,
            "neutral_impacts": neutral_impacts,
            "metrics_evaluated": num_metrics
        }
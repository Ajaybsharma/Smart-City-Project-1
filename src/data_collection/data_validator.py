import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataValidator:
    """Ensures data quality and consistency."""
    
    def __init__(self):
        """Initialize the DataValidator."""
        logger.info("Initialized DataValidator")
    
    def validate_dataframe(self, df: pd.DataFrame, validation_rules: Dict[str, Any] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a pandas DataFrame based on specified rules.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            validation_rules (Dict[str, Any], optional): Dictionary of validation rules
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (validation success, validation results)
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for validation")
            return False, {"error": "Empty DataFrame"}
        
        results = {
            "row_count": len(df),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "validation_checks": {}
        }
        
        if validation_rules is None:
            # Default validation - basic quality checks
            results["validation_checks"]["has_data"] = len(df) > 0
            results["validation_checks"]["no_missing_values"] = df.isnull().sum().sum() == 0
            results["validation_checks"]["no_duplicates"] = df.duplicated().sum() == 0
            
            # Check if validation passed
            validation_passed = all(results["validation_checks"].values())
        else:
            # Apply custom validation rules
            try:
                validation_passed = self._apply_validation_rules(df, validation_rules, results)
            except Exception as e:
                logger.error(f"Error applying validation rules: {e}")
                return False, {"error": str(e)}
        
        logger.info(f"Validation {'passed' if validation_passed else 'failed'}")
        return validation_passed, results
    
    def _apply_validation_rules(self, df: pd.DataFrame, rules: Dict[str, Any], results: Dict[str, Any]) -> bool:
        """
        Apply custom validation rules to the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            rules (Dict[str, Any]): Validation rules
            results (Dict[str, Any]): Results dictionary to update
            
        Returns:
            bool: True if validation passed, False otherwise
        """
        all_passed = True
        
        # Column presence check
        if "required_columns" in rules:
            required_cols = set(rules["required_columns"])
            existing_cols = set(df.columns)
            missing_cols = required_cols - existing_cols
            
            results["validation_checks"]["required_columns"] = len(missing_cols) == 0
            if missing_cols:
                results["validation_checks"]["missing_columns"] = list(missing_cols)
                all_passed = False
        
        # Data type check
        if "column_types" in rules:
            type_checks = {}
            for col, expected_type in rules["column_types"].items():
                if col in df.columns:
                    # Convert pandas dtype to string representation
                    actual_type = str(df[col].dtype)
                    type_checks[col] = {
                        "expected": expected_type,
                        "actual": actual_type,
                        "matches": self._check_compatible_types(actual_type, expected_type)
                    }
                    if not type_checks[col]["matches"]:
                        all_passed = False
            
            results["validation_checks"]["column_types"] = type_checks
        
        # Value range check
        if "value_ranges" in rules:
            range_checks = {}
            for col, range_vals in rules["value_ranges"].items():
                if col in df.columns:
                    min_val, max_val = range_vals
                    is_within_range = (df[col] >= min_val).all() and (df[col] <= max_val).all()
                    range_checks[col] = {
                        "min": min_val,
                        "max": max_val,
                        "within_range": is_within_range
                    }
                    if not is_within_range:
                        all_passed = False
            
            results["validation_checks"]["value_ranges"] = range_checks
        
        # Custom validation functions
        if "custom_validations" in rules:
            custom_checks = {}
            for validation_name, validation_func in rules["custom_validations"].items():
                try:
                    is_valid = validation_func(df)
                    custom_checks[validation_name] = is_valid
                    if not is_valid:
                        all_passed = False
                except Exception as e:
                    logger.error(f"Error in custom validation '{validation_name}': {e}")
                    custom_checks[validation_name] = False
                    all_passed = False
            
            results["validation_checks"]["custom_validations"] = custom_checks
        
        return all_passed
    
    def _check_compatible_types(self, actual_type: str, expected_type: str) -> bool:
        """
        Check if actual data type is compatible with expected type.
        
        Args:
            actual_type (str): Actual data type from DataFrame
            expected_type (str): Expected data type from validation rules
            
        Returns:
            bool: True if types are compatible, False otherwise
        """
        # Convert common pandas type strings to more general categories
        type_mapping = {
            'int': ['int', 'int8', 'int16', 'int32', 'int64'],
            'float': ['float', 'float16', 'float32', 'float64'],
            'str': ['object', 'string'],
            'bool': ['bool'],
            'datetime': ['datetime64', 'datetime64[ns]']
        }
        
        # Check if actual type belongs to the expected type category
        for general_type, specific_types in type_mapping.items():
            if expected_type.lower() == general_type:
                return any(t in actual_type.lower() for t in specific_types)
        
        # Direct string comparison as fallback
        return expected_type.lower() in actual_type.lower()
    
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable validation report.
        
        Args:
            validation_results (Dict[str, Any]): Validation results from validate_dataframe
            
        Returns:
            str: Formatted validation report
        """
        if "error" in validation_results:
            return f"Validation Error: {validation_results['error']}"
        
        report = []
        report.append("Data Validation Report")
        report.append("=====================")
        report.append(f"Row count: {validation_results['row_count']}")
        report.append(f"Duplicate rows: {validation_results['duplicate_rows']}")
        
        report.append("\nMissing Values:")
        for col, count in validation_results["missing_values"].items():
            if count > 0:
                pct = (count / validation_results['row_count']) * 100
                report.append(f"  - {col}: {count} ({pct:.2f}%)")
        
        report.append("\nValidation Checks:")
        for check_type, checks in validation_results.get("validation_checks", {}).items():
            report.append(f"\n{check_type.replace('_', ' ').title()}:")
            if isinstance(checks, dict):
                for item, result in checks.items():
                    if isinstance(result, dict):
                        report.append(f"  - {item}: {'Pass' if result.get('matches', False) else 'Fail'}")
                        for k, v in result.items():
                            if k != 'matches':
                                report.append(f"    {k}: {v}")
                    else:
                        report.append(f"  - {item}: {'Pass' if result else 'Fail'}")
            else:
                report.append(f"  - {check_type}: {'Pass' if checks else 'Fail'}")
        
        return "\n".join(report)
    
    def clean_and_validate(self, df: pd.DataFrame, rules: Dict[str, Any] = None) -> Tuple[pd.DataFrame, bool, Dict[str, Any]]:
        """
        Clean and validate a DataFrame in one step.
        
        Args:
            df (pd.DataFrame): DataFrame to clean and validate
            rules (Dict[str, Any], optional): Validation rules
            
        Returns:
            Tuple[pd.DataFrame, bool, Dict[str, Any]]: (cleaned DataFrame, validation success, validation results)
        """
        # Basic cleaning
        cleaned_df = df.copy()
        
        # Remove completely empty rows
        cleaned_df = cleaned_df.dropna(how='all')
        
        # Remove duplicate rows
        cleaned_df = cleaned_df.drop_duplicates()
        
        # Validate the cleaned DataFrame
        validation_passed, validation_results = self.validate_dataframe(cleaned_df, rules)
        
        return cleaned_df, validation_passed, validation_results
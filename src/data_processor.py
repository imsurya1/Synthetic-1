import pandas as pd
import numpy as np
from datetime import datetime
import openpyxl
from typing import Dict, Tuple, Any
import io

class DataProcessor:
    """
    Handles loading and automatic analysis of Excel files for synthetic data generation.
    Automatically detects column types, distributions, and relationships.
    """
    
    def __init__(self):
        self.supported_types = ['numerical', 'categorical', 'datetime', 'boolean']
    
    def load_and_analyze(self, file) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load Excel file and perform automatic schema detection and analysis.
        
        Args:
            file: Uploaded file object from Streamlit
            
        Returns:
            Tuple of (dataframe, column_info_dict)
        """
        try:
            # Load Excel file
            df = pd.read_excel(file)
            
            # Basic validation
            if df.empty:
                raise ValueError("Excel file is empty")
            
            if len(df) < 3:
                raise ValueError("Need at least 3 rows for meaningful analysis")
            
            # Perform automatic column analysis
            column_info = self._analyze_columns(df)
            
            # Clean and prepare data
            df_cleaned = self._clean_data(df, column_info)
            
            return df_cleaned, column_info
            
        except Exception as e:
            raise Exception(f"Error processing Excel file: {str(e)}")
    
    def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Automatically detect column types and characteristics.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with column information
        """
        column_info = {}
        
        for col in df.columns:
            column_info[col] = self._analyze_single_column(df[col])
        
        return column_info
    
    def _analyze_single_column(self, series: pd.Series) -> Dict[str, Any]:
        """
        Analyze a single column to determine its type and characteristics.
        
        Args:
            series: Pandas series to analyze
            
        Returns:
            Dictionary with column analysis results
        """
        info = {
            'name': series.name,
            'non_null_count': series.count(),
            'null_count': series.isnull().sum(),
            'unique_count': series.nunique(),
            'type': None,
            'subtype': None,
            'distribution_info': {},
            'constraints': []
        }
        
        # Remove null values for analysis
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            info['type'] = 'categorical'
            return info
        
        # Detect column type
        info['type'] = self._detect_column_type(clean_series)
        
        # Get type-specific analysis
        if info['type'] == 'numerical':
            info['distribution_info'] = self._analyze_numerical_column(clean_series)
        elif info['type'] == 'categorical':
            info['distribution_info'] = self._analyze_categorical_column(clean_series)
        elif info['type'] == 'datetime':
            info['distribution_info'] = self._analyze_datetime_column(clean_series)
        elif info['type'] == 'boolean':
            info['distribution_info'] = self._analyze_boolean_column(clean_series)
        
        # Detect potential constraints
        info['constraints'] = self._detect_constraints(clean_series, info['type'])
        
        return info
    
    def _detect_column_type(self, series: pd.Series) -> str:
        """
        Detect the primary type of a column.
        
        Args:
            series: Pandas series to analyze
            
        Returns:
            Detected column type
        """
        # Check for boolean
        unique_vals = set(series.dropna().astype(str).str.lower())
        if unique_vals.issubset({'true', 'false', '1', '0', 'yes', 'no', 'y', 'n'}):
            return 'boolean'
        
        # Check for datetime
        if self._is_datetime_column(series):
            return 'datetime'
        
        # Check for numerical
        if self._is_numerical_column(series):
            return 'numerical'
        
        # Default to categorical
        return 'categorical'
    
    def _is_datetime_column(self, series: pd.Series) -> bool:
        """Check if column contains datetime values."""
        if series.dtype.name.startswith('datetime'):
            return True
        
        # Try to parse as datetime
        try:
            sample_size = min(len(series), 10)
            sample = series.head(sample_size)
            parsed = pd.to_datetime(sample, errors='coerce')
            return parsed.notna().sum() / len(sample) > 0.8
        except:
            return False
    
    def _is_numerical_column(self, series: pd.Series) -> bool:
        """Check if column contains numerical values."""
        if pd.api.types.is_numeric_dtype(series):
            return True
        
        # Try to convert to numeric
        try:
            converted = pd.to_numeric(series, errors='coerce')
            return converted.notna().sum() / len(series) > 0.8
        except:
            return False
    
    def _analyze_numerical_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze numerical column characteristics."""
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        
        return {
            'min': float(numeric_series.min()),
            'max': float(numeric_series.max()),
            'mean': float(numeric_series.mean()),
            'median': float(numeric_series.median()),
            'std': float(numeric_series.std()),
            'skewness': float(numeric_series.skew()),
            'kurtosis': float(numeric_series.kurtosis()),
            'is_integer': all(x.is_integer() for x in numeric_series if pd.notna(x)),
            'percentiles': {
                '25': float(numeric_series.quantile(0.25)),
                '50': float(numeric_series.quantile(0.50)),
                '75': float(numeric_series.quantile(0.75)),
                '90': float(numeric_series.quantile(0.90)),
                '95': float(numeric_series.quantile(0.95))
            }
        }
    
    def _analyze_categorical_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze categorical column characteristics."""
        value_counts = series.value_counts()
        
        return {
            'unique_values': list(value_counts.index[:50]),  # Limit to top 50
            'value_counts': value_counts.to_dict(),
            'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
            'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
            'frequency_distribution': {
                'high_frequency': len(value_counts[value_counts > value_counts.mean()]),
                'low_frequency': len(value_counts[value_counts <= value_counts.mean()])
            }
        }
    
    def _analyze_datetime_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze datetime column characteristics."""
        dt_series = pd.to_datetime(series, errors='coerce').dropna()
        
        if len(dt_series) == 0:
            return {}
        
        return {
            'min_date': dt_series.min().isoformat(),
            'max_date': dt_series.max().isoformat(),
            'date_range_days': (dt_series.max() - dt_series.min()).days,
            'has_time_component': any(dt.time() != datetime.min.time() for dt in dt_series),
            'frequency_analysis': self._analyze_datetime_frequency(dt_series)
        }
    
    def _analyze_datetime_frequency(self, dt_series: pd.Series) -> Dict[str, Any]:
        """Analyze frequency patterns in datetime data."""
        if len(dt_series) < 2:
            return {}
        
        # Calculate time differences
        sorted_dates = dt_series.sort_values()
        time_diffs = sorted_dates.diff().dropna()
        
        # Most common time difference
        most_common_diff = time_diffs.mode()
        
        return {
            'most_common_interval_days': most_common_diff.iloc[0].days if len(most_common_diff) > 0 else None,
            'avg_interval_days': time_diffs.mean().days,
            'is_sequential': all(diff.days >= 0 for diff in time_diffs)
        }
    
    def _analyze_boolean_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze boolean column characteristics."""
        # Normalize boolean values
        normalized = series.astype(str).str.lower()
        true_values = {'true', '1', 'yes', 'y'}
        
        true_count = sum(1 for val in normalized if val in true_values)
        total_count = len(normalized)
        
        return {
            'true_ratio': true_count / total_count if total_count > 0 else 0,
            'false_ratio': (total_count - true_count) / total_count if total_count > 0 else 0,
            'true_count': true_count,
            'false_count': total_count - true_count
        }
    
    def _detect_constraints(self, series: pd.Series, col_type: str) -> list:
        """Detect potential constraints for a column."""
        constraints = []
        
        # Check for uniqueness
        if series.nunique() == len(series):
            constraints.append('unique')
        
        # Check for non-null constraint
        if series.isnull().sum() == 0:
            constraints.append('not_null')
        
        # Type-specific constraints
        if col_type == 'numerical':
            numeric_series = pd.to_numeric(series, errors='coerce').dropna()
            if all(x >= 0 for x in numeric_series):
                constraints.append('positive')
            if all(x.is_integer() for x in numeric_series if pd.notna(x)):
                constraints.append('integer')
        
        elif col_type == 'categorical':
            # Check for limited value set
            if series.nunique() <= 10:
                constraints.append('limited_values')
        
        return constraints
    
    def _clean_data(self, df: pd.DataFrame, column_info: Dict[str, Any]) -> pd.DataFrame:
        """
        Clean and prepare data based on detected column types.
        
        Args:
            df: Input dataframe
            column_info: Column analysis results
            
        Returns:
            Cleaned dataframe
        """
        df_cleaned = df.copy()
        
        for col, info in column_info.items():
            if info['type'] == 'numerical':
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
            
            elif info['type'] == 'datetime':
                df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
            
            elif info['type'] == 'boolean':
                # Standardize boolean values
                df_cleaned[col] = self._standardize_boolean(df_cleaned[col])
            
            # Note: Categorical columns are left as-is for SDV to handle
        
        return df_cleaned
    
    def _standardize_boolean(self, series: pd.Series) -> pd.Series:
        """Standardize boolean values to True/False."""
        normalized = series.astype(str).str.lower()
        true_values = {'true', '1', 'yes', 'y'}
        
        return normalized.isin(true_values)
    
    def get_column_relationships(self, df: pd.DataFrame, column_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze relationships between columns.
        
        Args:
            df: Input dataframe
            column_info: Column analysis results
            
        Returns:
            Dictionary with relationship analysis
        """
        relationships = {
            'correlations': {},
            'dependencies': {},
            'hierarchies': []
        }
        
        # Numerical correlations
        numerical_cols = [col for col, info in column_info.items() if info['type'] == 'numerical']
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            relationships['correlations'] = corr_matrix.to_dict()
        
        # Categorical dependencies (simplified)
        categorical_cols = [col for col, info in column_info.items() if info['type'] == 'categorical']
        for col1 in categorical_cols:
            for col2 in categorical_cols:
                if col1 != col2:
                    # Check if col2 values depend on col1
                    dependency_strength = self._calculate_dependency(df[col1], df[col2])
                    if dependency_strength > 0.7:
                        relationships['dependencies'][f"{col1}->{col2}"] = dependency_strength
        
        return relationships
    
    def _calculate_dependency(self, series1: pd.Series, series2: pd.Series) -> float:
        """Calculate dependency strength between two categorical series."""
        try:
            # Use conditional entropy approach
            joint_counts = pd.crosstab(series1, series2, normalize=True)
            marginal_counts = series1.value_counts(normalize=True)
            
            conditional_entropy = 0
            for val1 in series1.unique():
                if pd.isna(val1):
                    continue
                
                prob_val1 = marginal_counts.get(val1, 0)
                if prob_val1 == 0:
                    continue
                
                for val2 in series2.unique():
                    if pd.isna(val2):
                        continue
                    
                    joint_prob = joint_counts.loc[val1, val2] if (val1 in joint_counts.index and val2 in joint_counts.columns) else 0
                    if joint_prob > 0:
                        conditional_prob = joint_prob / prob_val1
                        conditional_entropy -= joint_prob * np.log2(conditional_prob)
            
            # Convert to dependency strength (0-1)
            max_entropy = -np.log2(1/len(series2.unique()))
            dependency = 1 - (conditional_entropy / max_entropy) if max_entropy > 0 else 0
            
            return max(0, min(1, dependency))
            
        except:
            return 0

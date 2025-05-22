import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, CopulaGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    from sdv.constraints import create_custom_constraint
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False

# Import our advanced synthesizer
from .advanced_synthesizer import AdvancedSynthesizer

class DataSynthesizer:
    """
    Handles synthetic data generation using various methods while preserving
    statistical distributions and relationships.
    """
    
    def __init__(self, method: str = "GaussianCopula", privacy_level: str = "Standard", random_seed: Optional[int] = None):
        """
        Initialize the synthesizer.
        
        Args:
            method: Synthesis method ('GaussianCopula', 'CTGAN', 'CopulaGAN')
            privacy_level: Privacy level ('Standard', 'High', 'Maximum')
            random_seed: Random seed for reproducibility
        """
        self.method = method
        self.privacy_level = privacy_level
        self.random_seed = random_seed
        self.synthesizer = None
        self.metadata = None
        self.constraints = []
        self.original_data = None
        self.column_info = None
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def fit(self, data: pd.DataFrame, column_info: Dict[str, Any], 
            unique_columns: List[str] = None, preserve_correlations: bool = True):
        """
        Fit the synthesizer to the original data.
        
        Args:
            data: Original dataset
            column_info: Column analysis information
            unique_columns: Columns that must have unique values
            preserve_correlations: Whether to preserve inter-column correlations
        """
        self.original_data = data.copy()
        self.column_info = column_info
        
        try:
            # Create metadata
            self.metadata = self._create_metadata(data, column_info)
            
            # Add constraints
            if unique_columns:
                self._add_unique_constraints(unique_columns)
            
            # Initialize synthesizer based on method
            self.synthesizer = self._create_synthesizer()
            
            # Fit the model
            self.synthesizer.fit(data)
            
        except Exception as e:
            raise Exception(f"Error fitting synthesizer: {str(e)}")
    
    def generate(self, num_rows: int, constraint_handling: str = "Reject Sampling") -> pd.DataFrame:
        """
        Generate synthetic data with advanced correlation preservation.
        
        Args:
            num_rows: Number of rows to generate
            constraint_handling: How to handle constraints
            
        Returns:
            Synthetic dataset
        """
        if self.synthesizer is None:
            raise ValueError("Synthesizer not fitted. Call fit() first.")
        
        try:
            # Use advanced synthesizer with enhanced correlation preservation
            advanced_synth = AdvancedSynthesizer(
                random_seed=self.random_seed, 
                correlation_threshold=0.8,
                enforce_constraints=True
            )
            
            # Get non-negative columns from column info
            non_negative_cols = [
                col for col, info in self.column_info.items()
                if info['type'] == 'numerical' and 
                   info.get('constraints', []).count('positive')
            ]
            
            # Fit with additional constraints
            advanced_synth.fit(
                self.original_data, 
                self.column_info,
                non_negative_columns=non_negative_cols
            )
            
            # Generate data with constraints
            synthetic_data = advanced_synth.generate(
                num_rows=num_rows,
                unique_columns=self.unique_columns if hasattr(self, 'unique_columns') else None,
                enforce_min_max=True
            )
            
            # Apply privacy adjustments while preserving constraints
            synthetic_data = self._apply_privacy_adjustments(
                synthetic_data,
                preserve_columns=non_negative_cols
            )
            
            return synthetic_data
            
        except Exception as e:
            print(f"Advanced generation failed: {e}")
            # Fallback to improved statistical generation
            return self._fallback_generation(num_rows)
    
    def _create_metadata(self, data: pd.DataFrame, column_info: Dict[str, Any]) -> 'SingleTableMetadata':
        """Create SDV metadata object."""
        try:
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data)
            
            # Update metadata based on our analysis
            for col, info in column_info.items():
                if info['type'] == 'categorical':
                    metadata.update_column(col, sdtype='categorical')
                elif info['type'] == 'numerical':
                    if info['distribution_info'].get('is_integer', False):
                        metadata.update_column(col, sdtype='numerical', computer_representation='Int64')
                    else:
                        metadata.update_column(col, sdtype='numerical', computer_representation='Float')
                elif info['type'] == 'datetime':
                    metadata.update_column(col, sdtype='datetime')
                elif info['type'] == 'boolean':
                    metadata.update_column(col, sdtype='boolean')
            
            return metadata
            
        except:
            # Fallback metadata creation
            return self._create_fallback_metadata(data, column_info)
    
    def _create_fallback_metadata(self, data: pd.DataFrame, column_info: Dict[str, Any]) -> Dict[str, str]:
        """Create fallback metadata when SDV is not available."""
        metadata = {}
        for col, info in column_info.items():
            metadata[col] = info['type']
        return metadata
    
    def _create_synthesizer(self):
        """Create the appropriate synthesizer based on method."""
        try:
            if self.method == "GaussianCopula":
                return GaussianCopulaSynthesizer(
                    metadata=self.metadata,
                    enforce_rounding=True,
                    default_distribution='gaussian'
                )
            elif self.method == "CTGAN":
                return CTGANSynthesizer(
                    metadata=self.metadata,
                    epochs=50 if self.privacy_level == "Standard" else 30,
                    verbose=False
                )
            elif self.method == "CopulaGAN":
                return CopulaGANSynthesizer(
                    metadata=self.metadata,
                    epochs=50 if self.privacy_level == "Standard" else 30,
                    verbose=False
                )
            else:
                raise ValueError(f"Unknown synthesis method: {self.method}")
                
        except:
            # Return fallback synthesizer
            return FallbackSynthesizer(self.original_data, self.column_info, self.random_seed)
    
    def _add_unique_constraints(self, unique_columns: List[str]):
        """Add uniqueness constraints for specified columns."""
        for col in unique_columns:
            if col in self.original_data.columns:
                try:
                    # SDV constraint
                    constraint = create_custom_constraint(
                        constraint=lambda data, col=col: data[col].is_unique,
                        transform=lambda data, col=col: self._ensure_uniqueness(data, col),
                        reverse_transform=lambda data: data
                    )
                    self.constraints.append(constraint)
                except:
                    # Store for manual handling
                    self.constraints.append({'type': 'unique', 'column': col})
    
    def _ensure_uniqueness(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Ensure uniqueness in a column by modifying duplicate values."""
        data = data.copy()
        duplicates = data.duplicated(subset=[column], keep='first')
        
        if duplicates.any():
            # For numerical columns, add small random increments
            if self.column_info[column]['type'] == 'numerical':
                random_increments = np.random.uniform(0.0001, 0.001, duplicates.sum())
                data.loc[duplicates, column] += random_increments
            
            # For categorical columns, append suffixes
            elif self.column_info[column]['type'] == 'categorical':
                for i, idx in enumerate(data.index[duplicates]):
                    data.loc[idx, column] = f"{data.loc[idx, column]}_{i+1}"
            
            # For datetime columns, add small time increments
            elif self.column_info[column]['type'] == 'datetime':
                time_increments = pd.to_timedelta(np.random.randint(1, 3600, duplicates.sum()), unit='s')
                data.loc[duplicates, column] += time_increments
        
        return data
    
    def _post_process(self, synthetic_data: pd.DataFrame) -> pd.DataFrame:
        """Apply post-processing to ensure data quality and constraints."""
        processed_data = synthetic_data.copy()
        
        # Apply manual constraints if any
        for constraint in self.constraints:
            if isinstance(constraint, dict) and constraint['type'] == 'unique':
                processed_data = self._ensure_uniqueness(processed_data, constraint['column'])
        
        # Ensure data types match original
        for col, info in self.column_info.items():
            if col in processed_data.columns:
                try:
                    if info['type'] == 'numerical':
                        processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
                        
                        # Apply range constraints
                        if 'distribution_info' in info:
                            min_val = info['distribution_info'].get('min')
                            max_val = info['distribution_info'].get('max')
                            if min_val is not None and max_val is not None:
                                processed_data[col] = np.clip(processed_data[col], min_val, max_val)
                        
                        # Ensure integer types remain integers
                        if info['distribution_info'].get('is_integer', False):
                            processed_data[col] = processed_data[col].round().astype('Int64')
                    
                    elif info['type'] == 'datetime':
                        processed_data[col] = pd.to_datetime(processed_data[col], errors='coerce')
                    
                    elif info['type'] == 'boolean':
                        # Ensure boolean values
                        processed_data[col] = processed_data[col].astype('boolean')
                    
                except Exception as e:
                    print(f"Warning: Could not post-process column {col}: {e}")
        
        return processed_data
    
    def _apply_privacy_adjustments(self, synthetic_data: pd.DataFrame) -> pd.DataFrame:
        """Apply privacy-preserving modifications based on privacy level."""
        if self.privacy_level == "Standard":
            return synthetic_data
        
        adjusted_data = synthetic_data.copy()
        
        for col, info in self.column_info.items():
            if col not in adjusted_data.columns:
                continue
            
            try:
                if info['type'] == 'numerical':
                    # Add noise to numerical columns
                    noise_scale = 0.01 if self.privacy_level == "High" else 0.05
                    std_dev = adjusted_data[col].std()
                    noise = np.random.normal(0, std_dev * noise_scale, len(adjusted_data))
                    adjusted_data[col] += noise
                
                elif info['type'] == 'categorical' and self.privacy_level == "Maximum":
                    # Occasionally replace values with similar ones
                    unique_vals = list(info['distribution_info'].get('unique_values', []))
                    if len(unique_vals) > 1:
                        replace_mask = np.random.random(len(adjusted_data)) < 0.05
                        replacement_vals = np.random.choice(unique_vals, replace_mask.sum())
                        adjusted_data.loc[replace_mask, col] = replacement_vals
                
            except Exception as e:
                print(f"Warning: Could not apply privacy adjustment to column {col}: {e}")
        
        return adjusted_data
    
    def _fallback_generation(self, num_rows: int) -> pd.DataFrame:
        """Fallback generation method when SDV is not available."""
        synthetic_data = pd.DataFrame()
        
        for col, info in self.column_info.items():
            try:
                if info['type'] == 'numerical':
                    # Generate from normal distribution with proper variation
                    mean = info['distribution_info'].get('mean', 0)
                    std = max(info['distribution_info'].get('std', 1), abs(mean) * 0.1)  # Ensure minimum std
                    values = np.random.normal(mean, std, num_rows)
                    
                    # Apply constraints
                    min_val = info['distribution_info'].get('min')
                    max_val = info['distribution_info'].get('max')
                    if min_val is not None and max_val is not None:
                        # Add some buffer to avoid all values being at bounds
                        range_buffer = (max_val - min_val) * 0.1
                        values = np.clip(values, min_val - range_buffer, max_val + range_buffer)
                    
                    if info['distribution_info'].get('is_integer', False):
                        values = values.round().astype(int)
                    
                    synthetic_data[col] = values
                
                elif info['type'] == 'categorical':
                    # Sample from original distribution with proper randomness
                    unique_vals = info['distribution_info'].get('unique_values', [f'Category_{i}' for i in range(1, 6)])
                    value_counts = info['distribution_info'].get('value_counts', {})
                    
                    if value_counts and len(value_counts) > 0:
                        # Use original distribution but add some randomness
                        probabilities = np.array(list(value_counts.values()), dtype=float)
                        probabilities = probabilities / probabilities.sum()
                        # Add small random noise to avoid identical patterns
                        probabilities += np.random.uniform(0, 0.05, len(probabilities))
                        probabilities = probabilities / probabilities.sum()
                        values = np.random.choice(list(value_counts.keys()), num_rows, p=probabilities)
                    else:
                        values = np.random.choice(unique_vals, num_rows)
                    
                    synthetic_data[col] = values
                
                elif info['type'] == 'datetime':
                    # Generate dates within original range with proper variation
                    min_date = pd.to_datetime(info['distribution_info'].get('min_date', '2020-01-01'))
                    max_date = pd.to_datetime(info['distribution_info'].get('max_date', '2024-12-31'))
                    
                    date_range = max((max_date - min_date).days, 365)  # Ensure minimum 1 year range
                    random_days = np.random.randint(0, date_range + 1, num_rows)
                    values = min_date + pd.to_timedelta(random_days, unit='D')
                    
                    synthetic_data[col] = values
                
                elif info['type'] == 'boolean':
                    # Generate based on original true ratio with variation
                    true_ratio = info['distribution_info'].get('true_ratio', 0.5)
                    # Add slight random variation to avoid patterns
                    varied_ratio = max(0.05, min(0.95, true_ratio + np.random.uniform(-0.1, 0.1)))
                    values = np.random.random(num_rows) < varied_ratio
                    synthetic_data[col] = values
                
                else:
                    # Default: treat as categorical
                    original_col = self.original_data[col].dropna()
                    if len(original_col) > 0:
                        values = np.random.choice(original_col, num_rows, replace=True)
                    else:
                        values = [f'Value_{i}' for i in range(num_rows)]
                    synthetic_data[col] = values
                    
            except Exception as e:
                print(f"Error generating column {col}: {e}")
                # Emergency fallback for this column
                original_col = self.original_data[col].dropna()
                if len(original_col) > 0:
                    synthetic_data[col] = np.random.choice(original_col, num_rows, replace=True)
                else:
                    synthetic_data[col] = [None] * num_rows
        
        return synthetic_data
    
    def _emergency_fallback(self, num_rows: int) -> pd.DataFrame:
        """Emergency fallback when all else fails."""
        synthetic_data = pd.DataFrame()
        
        for col in self.original_data.columns:
            original_col = self.original_data[col].dropna()
            if len(original_col) > 0:
                # Handle different data types properly
                if pd.api.types.is_numeric_dtype(original_col):
                    # For numeric data, add some variation
                    mean_val = original_col.mean()
                    std_val = original_col.std() if original_col.std() > 0 else mean_val * 0.1
                    synthetic_data[col] = np.random.normal(mean_val, std_val, num_rows)
                    # Ensure we stay within reasonable bounds
                    min_val, max_val = original_col.min(), original_col.max()
                    synthetic_data[col] = np.clip(synthetic_data[col], min_val * 0.8, max_val * 1.2)
                elif pd.api.types.is_datetime64_any_dtype(original_col):
                    # For datetime data, sample within the range
                    min_date = original_col.min()
                    max_date = original_col.max()
                    date_range = (max_date - min_date).days
                    if date_range > 0:
                        random_days = np.random.randint(0, max(1, date_range), num_rows)
                        synthetic_data[col] = min_date + pd.to_timedelta(random_days, unit='D')
                    else:
                        synthetic_data[col] = [min_date] * num_rows
                else:
                    # For categorical data, sample from original values
                    synthetic_data[col] = np.random.choice(original_col, num_rows, replace=True)
            else:
                synthetic_data[col] = [None] * num_rows
        
        return synthetic_data


class FallbackSynthesizer:
    """Fallback synthesizer when SDV is not available."""
    
    def __init__(self, data: pd.DataFrame, column_info: Dict[str, Any], random_seed: Optional[int] = None):
        self.data = data
        self.column_info = column_info
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def fit(self, data: pd.DataFrame):
        """Fit method for compatibility."""
        self.data = data
    
    def sample(self, num_rows: int) -> pd.DataFrame:
        """Generate synthetic data using simple statistical sampling."""
        synthetic_data = pd.DataFrame()
        
        for col, info in self.column_info.items():
            original_col = self.data[col].dropna()
            
            if info['type'] == 'numerical':
                # Use bootstrap sampling with added noise
                sampled = np.random.choice(original_col, num_rows, replace=True)
                noise = np.random.normal(0, original_col.std() * 0.1, num_rows)
                synthetic_data[col] = sampled + noise
            
            elif info['type'] == 'categorical':
                # Bootstrap sampling
                synthetic_data[col] = np.random.choice(original_col, num_rows, replace=True)
            
            elif info['type'] == 'datetime':
                # Bootstrap with time jitter
                sampled = np.random.choice(original_col, num_rows, replace=True)
                jitter = pd.to_timedelta(np.random.randint(-30, 31, num_rows), unit='D')
                synthetic_data[col] = sampled + jitter
            
            elif info['type'] == 'boolean':
                # Bootstrap sampling
                synthetic_data[col] = np.random.choice(original_col, num_rows, replace=True)
        
        return synthetic_data

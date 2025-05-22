import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class AdvancedSynthesizer:
    """
    Advanced synthetic data generator inspired by Gretel.ai approach.
    Uses statistical modeling and machine learning for high-quality synthesis.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialize the advanced synthesizer."""
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.fitted = False
        self.original_data = None
        self.column_info = None
        self.encoders = {}
        self.scalers = {}
        self.models = {}
        self.correlation_matrix = None
        
    def fit(self, data: pd.DataFrame, column_info: Dict[str, Any]):
        """
        Fit the synthesizer to the original data using advanced statistical modeling.
        
        Args:
            data: Original dataset
            column_info: Column analysis information
        """
        self.original_data = data.copy()
        self.column_info = column_info
        
        # Prepare data for modeling
        processed_data = self._preprocess_data(data)
        
        # Build statistical models for different column types
        self._build_models(processed_data)
        
        # Capture correlation structure
        self._capture_correlations(processed_data)
        
        self.fitted = True
    
    def generate(self, num_rows: int, unique_columns: List[str] = None) -> pd.DataFrame:
        """
        Generate high-quality synthetic data.
        
        Args:
            num_rows: Number of rows to generate
            unique_columns: Columns that must have unique values
            
        Returns:
            Synthetic dataset
        """
        if not self.fitted:
            raise ValueError("Synthesizer must be fitted before generating data")
        
        # Generate synthetic data using advanced methods
        synthetic_data = self._generate_correlated_data(num_rows)
        
        # Apply uniqueness constraints
        if unique_columns:
            synthetic_data = self._enforce_uniqueness(synthetic_data, unique_columns)
        
        # Post-process to ensure data quality
        synthetic_data = self._post_process(synthetic_data)
        
        return synthetic_data
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for modeling."""
        processed_data = pd.DataFrame()
        
        for col, info in self.column_info.items():
            if col not in data.columns:
                continue
                
            if info['type'] == 'numerical':
                # Standardize numerical data
                scaler = StandardScaler()
                values = data[col].fillna(data[col].median()).values.reshape(-1, 1)
                processed_values = scaler.fit_transform(values).flatten()
                processed_data[col] = processed_values
                self.scalers[col] = scaler
                
            elif info['type'] == 'categorical':
                # Encode categorical data
                encoder = LabelEncoder()
                values = data[col].fillna('Unknown').astype(str)
                encoded_values = encoder.fit_transform(values)
                processed_data[col] = encoded_values
                self.encoders[col] = encoder
                
            elif info['type'] == 'datetime':
                # Convert datetime to numerical (days since min date)
                dt_values = pd.to_datetime(data[col])
                min_date = dt_values.min()
                days_since_min = (dt_values - min_date).dt.days.fillna(0)
                
                scaler = StandardScaler()
                processed_values = scaler.fit_transform(days_since_min.values.reshape(-1, 1)).flatten()
                processed_data[col] = processed_values
                self.scalers[col] = scaler
                self.encoders[col] = min_date  # Store min date for reconstruction
                
            elif info['type'] == 'boolean':
                # Convert boolean to 0/1
                bool_values = data[col].fillna(False).astype(bool).astype(int)
                processed_data[col] = bool_values
        
        return processed_data
    
    def _build_models(self, processed_data: pd.DataFrame):
        """Build statistical models for data generation."""
        # Use Gaussian Mixture Models for better distribution modeling
        for col in processed_data.columns:
            try:
                # Fit Gaussian Mixture Model for each column
                gmm = GaussianMixture(
                    n_components=min(3, len(processed_data[col].unique())),
                    random_state=self.random_seed,
                    covariance_type='full'
                )
                
                col_data = processed_data[col].values.reshape(-1, 1)
                gmm.fit(col_data)
                self.models[col] = gmm
                
            except Exception as e:
                print(f"Warning: Could not fit model for column {col}: {e}")
                # Fallback to simple statistics
                self.models[col] = {
                    'mean': processed_data[col].mean(),
                    'std': processed_data[col].std(),
                    'unique_values': processed_data[col].unique()
                }
    
    def _capture_correlations(self, processed_data: pd.DataFrame):
        """Capture correlation structure using PCA and correlation matrix."""
        try:
            # Calculate correlation matrix
            self.correlation_matrix = processed_data.corr().fillna(0)
            
            # Use PCA to capture main patterns
            if len(processed_data.columns) > 1:
                pca = PCA(n_components=min(len(processed_data.columns), len(processed_data)))
                pca.fit(processed_data.fillna(0))
                self.pca_model = pca
            else:
                self.pca_model = None
                
        except Exception as e:
            print(f"Warning: Could not capture correlations: {e}")
            self.correlation_matrix = pd.DataFrame()
            self.pca_model = None
    
    def _generate_correlated_data(self, num_rows: int) -> pd.DataFrame:
        """Generate synthetic data preserving correlations."""
        synthetic_data = pd.DataFrame()
        
        # Generate base synthetic data
        for col, model in self.models.items():
            try:
                if hasattr(model, 'sample'):
                    # Use Gaussian Mixture Model
                    generated_values = model.sample(num_rows)[0].flatten()
                else:
                    # Use fallback statistics
                    if 'unique_values' in model and len(model['unique_values']) < 20:
                        # Categorical-like data
                        generated_values = np.random.choice(model['unique_values'], num_rows)
                    else:
                        # Continuous data
                        generated_values = np.random.normal(
                            model['mean'], 
                            max(model['std'], 0.1), 
                            num_rows
                        )
                
                synthetic_data[col] = generated_values
                
            except Exception as e:
                print(f"Warning: Error generating column {col}: {e}")
                # Emergency fallback
                original_col = self.original_data[col].dropna()
                if len(original_col) > 0:
                    synthetic_data[col] = np.random.choice(original_col, num_rows, replace=True)
                else:
                    synthetic_data[col] = [0] * num_rows
        
        # Apply correlation adjustments if possible
        if self.pca_model is not None and len(synthetic_data.columns) > 1:
            try:
                # Transform to PCA space and back to preserve relationships
                pca_transformed = self.pca_model.transform(synthetic_data.fillna(0))
                
                # Add some controlled noise in PCA space
                noise_scale = 0.1
                pca_transformed += np.random.normal(0, noise_scale, pca_transformed.shape)
                
                # Transform back
                reconstructed = self.pca_model.inverse_transform(pca_transformed)
                synthetic_data = pd.DataFrame(reconstructed, columns=synthetic_data.columns)
                
            except Exception as e:
                print(f"Warning: Could not apply correlation adjustment: {e}")
        
        return synthetic_data
    
    def _post_process(self, synthetic_data: pd.DataFrame) -> pd.DataFrame:
        """Post-process synthetic data to restore original formats."""
        final_data = pd.DataFrame()
        
        for col, info in self.column_info.items():
            if col not in synthetic_data.columns:
                continue
            
            try:
                if info['type'] == 'numerical':
                    # Reverse standardization
                    if col in self.scalers:
                        values = self.scalers[col].inverse_transform(
                            synthetic_data[col].values.reshape(-1, 1)
                        ).flatten()
                    else:
                        values = synthetic_data[col].values
                    
                    # Apply original constraints
                    dist_info = info.get('distribution_info', {})
                    if 'min' in dist_info and 'max' in dist_info:
                        values = np.clip(values, dist_info['min'], dist_info['max'])
                    
                    if dist_info.get('is_integer', False):
                        values = values.round().astype(int)
                    
                    final_data[col] = values
                
                elif info['type'] == 'categorical':
                    # Reverse encoding
                    if col in self.encoders:
                        # Ensure values are within encoder range
                        encoded_values = synthetic_data[col].round().astype(int)
                        max_label = len(self.encoders[col].classes_) - 1
                        encoded_values = np.clip(encoded_values, 0, max_label)
                        
                        decoded_values = self.encoders[col].inverse_transform(encoded_values)
                        final_data[col] = decoded_values
                    else:
                        final_data[col] = synthetic_data[col]
                
                elif info['type'] == 'datetime':
                    # Reverse datetime conversion
                    if col in self.scalers and col in self.encoders:
                        # Reverse standardization
                        days_values = self.scalers[col].inverse_transform(
                            synthetic_data[col].values.reshape(-1, 1)
                        ).flatten()
                        
                        # Convert back to dates
                        min_date = self.encoders[col]  # Stored min date
                        final_dates = min_date + pd.to_timedelta(days_values.round(), unit='D')
                        final_data[col] = final_dates
                    else:
                        # Fallback
                        original_dates = self.original_data[col].dropna()
                        if len(original_dates) > 0:
                            min_date, max_date = original_dates.min(), original_dates.max()
                            date_range = (max_date - min_date).days
                            random_days = np.random.randint(0, max(1, date_range), len(synthetic_data))
                            final_data[col] = min_date + pd.to_timedelta(random_days, unit='D')
                        else:
                            final_data[col] = pd.Timestamp('2024-01-01')
                
                elif info['type'] == 'boolean':
                    # Convert back to boolean
                    bool_values = synthetic_data[col] > 0.5
                    final_data[col] = bool_values
                
                else:
                    final_data[col] = synthetic_data[col]
                    
            except Exception as e:
                print(f"Warning: Error post-processing column {col}: {e}")
                # Emergency fallback - sample from original
                original_col = self.original_data[col].dropna()
                if len(original_col) > 0:
                    final_data[col] = np.random.choice(original_col, len(synthetic_data), replace=True)
                else:
                    final_data[col] = [None] * len(synthetic_data)
        
        return final_data
    
    def _enforce_uniqueness(self, data: pd.DataFrame, unique_columns: List[str]) -> pd.DataFrame:
        """Enforce uniqueness constraints on specified columns."""
        result_data = data.copy()
        
        for col in unique_columns:
            if col in result_data.columns:
                # Check for duplicates
                duplicates = result_data.duplicated(subset=[col], keep='first')
                
                if duplicates.any():
                    dup_indices = result_data.index[duplicates]
                    
                    if self.column_info[col]['type'] == 'numerical':
                        # Add small increments to make unique
                        base_values = result_data.loc[dup_indices, col].values
                        increments = np.arange(1, len(dup_indices) + 1) * 0.001
                        result_data.loc[dup_indices, col] = base_values + increments
                        
                    elif self.column_info[col]['type'] == 'categorical':
                        # Append unique suffixes
                        for i, idx in enumerate(dup_indices):
                            original_val = str(result_data.loc[idx, col])
                            result_data.loc[idx, col] = f"{original_val}_{i+1:04d}"
                    
                    elif self.column_info[col]['type'] == 'datetime':
                        # Add random seconds to make unique
                        for idx in dup_indices:
                            random_seconds = np.random.randint(1, 3600)
                            result_data.loc[idx, col] += pd.Timedelta(seconds=random_seconds)
        
        return result_data
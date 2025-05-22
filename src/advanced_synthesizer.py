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
        """Initialize the advanced synthesizer using Gretel's approach."""
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

        self.fitted = False
        self.original_data = None
        self.column_info = None
        self.scalers = {}
        self.encoders = {}
        self.models = {}
        self.quality_target = 95
        self.enforce_positive = True  # Always enforce positive values
        self.min_quality_threshold = 90  # Minimum quality threshold

        # Initialize enhanced model parameters
        self.model_params = {
            'batch_size': 128,
            'learning_rate': 0.001,
            'hidden_layers': [256, 128, 64],
            'dropout_rate': 0.2,
            'generator_dim': (512, 256),  # Larger generator network
            'discriminator_dim': (256, 256),
            'enforcing_min_max_values': True,
            'numerical_distributions': 'beta',  # Better for bounded distributions
            'enforce_rounding': True,  # Ensure integer columns stay integers
            'enforce_min_max_values': True,  # Strictly enforce value bounds
            'pac': 10
        }
            log_frequency=True,
            verbose=True
        )

    def fit(self, data: pd.DataFrame, column_info: Dict[str, Any]):
        """
        Fit the synthesizer using Gretel's ACTGAN.

        Args:
            data: Original dataset
            column_info: Column analysis information
        """
        self.original_data = data.copy()
        self.column_info = column_info

        # Configure field types for ACTGAN with enhanced constraints
        field_types = {}
        field_transformers = {}

        for col, info in column_info.items():
            if info['type'] == 'numerical':
                field_types[col] = 'numerical'
                # Get original column stats for constraints
                orig_col = self.original_data[col]
                min_val = orig_col.min()
                max_val = orig_col.max()

                field_transformers[col] = {
                    'enforce_min_value': max(0, min_val),  # Ensure non-negative
                    'enforce_max_value': max_val,  # Keep upper bound
                    'transform_method': 'beta',  # Better distribution preservation
                    'enforce_rounding': info.get('distribution_info', {}).get('is_integer', False)
                }
            elif info['type'] == 'categorical':
                field_types[col] = 'categorical'
            elif info['type'] == 'datetime':
                field_types[col] = 'datetime'

        # Fit ACTGAN
        self.actgan.fit(
            data,
            field_types=field_types,
            field_transformers=field_transformers
        )

        self.fitted = True

    def generate(self, num_rows: int, unique_columns: List[str] = None) -> pd.DataFrame:
        """Generate high-quality synthetic data with enhanced constraints."""
        if not self.fitted:
            raise ValueError("Synthesizer must be fitted before generating data")

        attempts = 0
        max_attempts = 5
        best_quality = 0
        best_data = None

        while attempts < max_attempts:
            # Generate base synthetic data
            synthetic_data = self.actgan.generate(num_rows)

            # Apply constraints and improvements
            synthetic_data = self._enforce_constraints(synthetic_data)
            synthetic_data = self._preserve_correlations(synthetic_data)
            synthetic_data = self._optimize_distributions(synthetic_data)

            # Check quality
            quality_score = self._calculate_quality_score(synthetic_data)

            if quality_score > best_quality:
                best_quality = quality_score
                best_data = synthetic_data.copy()

            if quality_score >= self.quality_target:
                break

            attempts += 1

        if best_data is None:
            raise ValueError("Failed to generate high-quality synthetic data")

        return best_data

    def _enforce_constraints(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enforce all data constraints strictly with enhanced positive value handling."""
        result = data.copy()

        for col, info in self.column_info.items():
            if col not in result.columns:
                continue

            if info['type'] == 'numerical':
                # Get original statistics
                orig_stats = info.get('distribution_info', {})
                orig_col = self.original_data[col]
                min_val = max(0, orig_stats.get('min', 0))  # Ensure minimum is non-negative
                max_val = orig_stats.get('max', orig_col.max())
                mean = orig_stats.get('mean', orig_col.mean())
                std = orig_stats.get('std', orig_col.std())

                # Transform negative values
                result[col] = result[col].abs()  # Convert negatives to positive

                # Scale to match original distribution while preserving positivity
                if result[col].mean() > 0:
                    scale_factor = mean / result[col].mean()
                    result[col] = result[col] * scale_factor

                # Add small constant to ensure strictly positive
                result[col] = result[col] + 1e-6

                # Clip to valid range while preserving minimum positive value
                result[col] = np.clip(result[col], min_val, max_val)

                # Handle integers with positive constraint
                if info.get('distribution_info', {}).get('is_integer', False):
                    result[col] = np.maximum(1, np.round(result[col])).astype(int)

        return result

    def _preserve_correlations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhance correlation preservation."""
        numerical_cols = [col for col, info in self.column_info.items() 
                         if info['type'] == 'numerical']

        if len(numerical_cols) < 2:
            return data

        result = data.copy()
        orig_corr = self.original_data[numerical_cols].corr()

        # Apply correlation correction using Cholesky decomposition
        try:
            current_corr = result[numerical_cols].corr()
            target_corr = orig_corr.fillna(0)

            L = np.linalg.cholesky(target_corr)
            L_inv = np.linalg.inv(L)

            transformed = result[numerical_cols].apply(lambda x: (x - x.mean()) / x.std())
            corrected = transformed.dot(L_inv.T)

            # Scale back to original ranges
            for col in numerical_cols:
                orig_mean = self.original_data[col].mean()
                orig_std = self.original_data[col].std()
                result[col] = corrected[col] * orig_std + orig_mean
        except:
            pass  # Fallback to original data if correction fails

        return result

    def _optimize_distributions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize statistical distributions of synthetic data."""
        result = data.copy()

        for col, info in self.column_info.items():
            if info['type'] == 'numerical':
                orig_series = self.original_data[col]
                synth_series = result[col]

                # Match moments
                orig_mean = orig_series.mean()
                orig_std = orig_series.std()
                orig_skew = orig_series.skew()

                # Apply Box-Cox transformation to match distribution shape
                try:
                    from scipy import stats
                    transformed = stats.boxcox(synth_series - synth_series.min() + 1)[0]
                    transformed = (transformed - transformed.mean()) / transformed.std()
                    transformed = transformed * orig_std + orig_mean

                    # Restore original range
                    min_val = max(0, orig_series.min())
                    max_val = orig_series.max()
                    result[col] = np.clip(transformed, min_val, max_val)
                except:
                    pass  # Fallback to original if transformation fails

        return result

    def _calculate_quality_score(self, synthetic_data: pd.DataFrame) -> float:
        """Calculate an enhanced quality score with stricter criteria."""
        score = 100  # Start with a perfect score

        for col, info in self.column_info.items():
            if col not in synthetic_data.columns:
                score -= 10  # Higher penalty for missing columns
                continue

            if info['type'] == 'numerical':
                orig_data = self.original_data[col]
                synth_data = synthetic_data[col]

                # Strict checks for negative values
                if synth_data.min() < 0:
                    score -= 20  # Heavy penalty for negative values

                # Compare distributions
                orig_mean = orig_data.mean()
                synth_mean = synth_data.mean()
                orig_std = orig_data.std()
                synth_std = synth_data.std()

                # Stricter penalties for distribution mismatches
                mean_diff = abs(orig_mean - synth_mean) / max(orig_mean, 0.1)
                std_diff = abs(orig_std - synth_std) / max(orig_std, 0.1)

                score -= mean_diff * 15  # Increased penalty for mean difference
                score -= std_diff * 15   # Increased penalty for std difference

            elif info['type'] == 'categorical':
                # Compare value distributions
                orig_counts = self.original_data[col].value_counts(normalize=True)
                synth_counts = synthetic_data[col].value_counts(normalize=True)

                for val, orig_freq in orig_counts.items():
                    synth_freq = synth_counts.get(val, 0)
                    score -= abs(orig_freq - synth_freq) * 20

        return max(0, score)


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

                    # Ensure positive values for all numeric columns
                    min_val = max(0, dist_info.get('min', 0))
                    max_val = dist_info.get('max', values.max())
                    values = np.clip(values, min_val, max_val)

                    # Additional handling for strictly positive values
                    if min_val == 0:
                        # Add small epsilon to ensure strictly positive
                        values = values + 1e-6
                        values = values * (dist_info.get('mean', values.mean()) / values.mean())

                    if dist_info.get('is_integer', False):
                        values = np.maximum(1, values.round()).astype(int)

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
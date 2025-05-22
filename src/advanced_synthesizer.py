import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from scipy import stats # Added import
import warnings
warnings.filterwarnings('ignore')

# Placeholder for ACTGAN - in a real scenario, this would be imported
class ACTGANPlaceholder:
    def __init__(self, **kwargs):
        print(f"ACTGANPlaceholder initialized with params: {kwargs}")
        self.model_params = kwargs
        self._field_types = None
        self._field_transformers = None
        self._data_columns = None

    def fit(self, data: pd.DataFrame, field_types: Dict[str, str], field_transformers: Dict[str, Any]):
        print("ACTGANPlaceholder: Fit called.")
        self._field_types = field_types
        self._field_transformers = field_transformers
        self._data_columns = data.columns
        # In a real ACTGAN, model training would happen here.

    def generate(self, num_rows: int) -> pd.DataFrame:
        print(f"ACTGANPlaceholder: Generate called for {num_rows} rows.")
        # This is a very naive generation for placeholder purposes.
        # A real ACTGAN would generate data based on its learned model.
        if self._data_columns is None:
            raise ValueError("ACTGANPlaceholder must be fitted before generating data.")
        
        data_dict = {}
        for col_name in self._data_columns:
            col_type = self._field_types.get(col_name, 'numerical')
            if col_type == 'numerical':
                # Try to respect min/max from transformers if provided
                min_val = self._field_transformers.get(col_name, {}).get('enforce_min_value', 0)
                max_val = self._field_transformers.get(col_name, {}).get('enforce_max_value', 100)
                if self._field_transformers.get(col_name, {}).get('enforce_rounding', False):
                    data_dict[col_name] = np.random.randint(max(1, int(min_val)), int(max_val) + 1, num_rows)
                else:
                    data_dict[col_name] = np.random.uniform(min_val, max_val, num_rows)
            elif col_type == 'categorical':
                # Placeholder: generate random categories (e.g., A, B, C)
                categories = [f"Cat{i}" for i in range(np.random.randint(2,5))]
                data_dict[col_name] = np.random.choice(categories, num_rows)
            elif col_type == 'datetime':
                start_date = pd.Timestamp('2020-01-01')
                data_dict[col_name] = start_date + pd.to_timedelta(np.random.randint(0, 365*3, num_rows), unit='D')
            else:
                data_dict[col_name] = [None] * num_rows
        return pd.DataFrame(data_dict)

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
        self.models = {} # For the alternative GMM-based synthesis path
        self.quality_target = 95
        self.enforce_positive = True  # Always enforce positive values
        self.min_quality_threshold = 95  # Increased minimum threshold

        # Initialize enhanced model parameters with stricter quality settings
        self.model_params = {
            'batch_size': 64,
            'learning_rate': 0.0005,
            'hidden_layers': [512, 256, 128],
            'dropout_rate': 0.1,
            'generator_dim': (1024, 512),
            'discriminator_dim': (512, 256),
            'enforcing_min_max_values': True,
            'numerical_distributions': 'beta', # Changed from numerical_distribution
            'enforce_rounding': True,
            # 'enforce_min_max_values': True, # Duplicates 'enforcing_min_max_values'
            'enforce_positivity': True,
            'pac': 5,
            'log_frequency': True,
            'verbose': True
        }
        # Initialize the ACTGAN model (or placeholder)
        # In a real application, you would import ACTGAN from gretel_synthetics or similar
        self.actgan = ACTGANPlaceholder(**self.model_params)


    def fit(self, data: pd.DataFrame, column_info: Dict[str, Any]):
        """
        Fit the synthesizer using Gretel's ACTGAN.

        Args:
            data: Original dataset
            column_info: Column analysis information
        """
        self.original_data = data.copy()
        self.column_info = column_info

        field_types = {}
        field_transformers = {}

        for col, info in column_info.items():
            if info['type'] == 'numerical':
                field_types[col] = 'numerical'
                orig_col = self.original_data[col]
                min_val = orig_col.min()
                max_val = orig_col.max()

                field_transformers[col] = {
                    'enforce_min_value': max(0, min_val),
                    'enforce_max_value': max_val,
                    'transform_method': self.model_params.get('numerical_distributions', 'beta'), # Use from model_params
                    'enforce_rounding': info.get('distribution_info', {}).get('is_integer', False)
                }
            elif info['type'] == 'categorical':
                field_types[col] = 'categorical'
            elif info['type'] == 'datetime':
                field_types[col] = 'datetime'
            # Add boolean if it's a distinct type handled by ACTGAN
            elif info['type'] == 'boolean':
                field_types[col] = 'boolean' # Or categorical, depending on ACTGAN

        self.actgan.fit(
            data,
            field_types=field_types,
            field_transformers=field_transformers
        )

        self.fitted = True
        
        # Fit the GMM-based models as well if this path is intended to be used
        # processed_data = self._preprocess_data(self.original_data)
        # self._build_models(processed_data)
        # self._capture_correlations(processed_data)


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

            # Apply constraints and improvements in a revised order
            synthetic_data = self._preserve_correlations(synthetic_data.copy())
            synthetic_data = self._optimize_distributions(synthetic_data.copy())
            synthetic_data = self._enforce_constraints(synthetic_data.copy()) # Enforce last

            if unique_columns:
                 synthetic_data = self._enforce_uniqueness(synthetic_data.copy(), unique_columns)

            quality_score = self._calculate_quality_score(synthetic_data)

            if quality_score > best_quality:
                best_quality = quality_score
                best_data = synthetic_data.copy()

            if quality_score >= self.quality_target:
                print(f"Achieved target quality score of {quality_score:.2f}% in attempt {attempts + 1}.")
                break
            
            print(f"Attempt {attempts + 1}: Quality score {quality_score:.2f}% (Target: {self.quality_target}%)")
            attempts += 1

        if best_data is None:
            raise ValueError("Failed to generate data. No successful attempts.")
        if best_quality < self.min_quality_threshold :
             print(f"Warning: Best quality score ({best_quality:.2f}%) is below minimum threshold ({self.min_quality_threshold}%).")
        
        return best_data

    def _enforce_constraints(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enforce all data constraints strictly with enhanced positive value handling."""
        result = data.copy()
        small_epsilon = 1e-6  # Small constant for strict positivity for floats

        for col, info in self.column_info.items():
            if col not in result.columns:
                continue

            if info['type'] == 'numerical':
                current_col_synth = result[col].copy()
                orig_col_series = self.original_data[col]

                # Get original statistics, preferring info if available, else from original_data
                dist_info = info.get('distribution_info', {})
                orig_min_val = dist_info.get('min', orig_col_series.min())
                orig_max_val = dist_info.get('max', orig_col_series.max())
                orig_mean_val = dist_info.get('mean', orig_col_series.mean())
                is_integer = dist_info.get('is_integer', False)

                min_constraint = max(0, orig_min_val) # Floor original min at 0 for constraint baseline

                if not is_integer:
                    # --- Floating point numbers ---
                    # 1. Initial floor at 0
                    current_col_synth = np.maximum(current_col_synth, 0)

                    # 2. Enforce strict positivity if original was strictly positive,
                    #    or if global enforce_positive is true and original min was 0.
                    should_be_strictly_positive = (orig_min_val > 0) or \
                                                  (min_constraint == 0 and self.enforce_positive)

                    if should_be_strictly_positive:
                        current_col_synth = np.maximum(current_col_synth, small_epsilon)
                        
                        # 3. Rescale to target mean, if current mean is positive and target is positive
                        current_mean_synth = current_col_synth.mean()
                        if current_mean_synth > (small_epsilon / 2) and orig_mean_val > 0: # Avoid division by zero/small
                            scale_factor = orig_mean_val / current_mean_synth
                            current_col_synth = current_col_synth * scale_factor
                            # Re-floor at epsilon after scaling, as values could dip below
                            current_col_synth = np.maximum(current_col_synth, small_epsilon)
                    
                    # 4. Clip to the final range
                    #    The min_clip_bound should be `small_epsilon` if we made it strictly positive and original min was 0.
                    #    Otherwise, it's `min_constraint`.
                    min_clip_bound = min_constraint
                    if should_be_strictly_positive and min_constraint == 0:
                        min_clip_bound = small_epsilon
                    
                    result[col] = np.clip(current_col_synth, min_clip_bound, orig_max_val)

                else:
                    # --- Integer numbers ---
                    current_col_synth = np.round(current_col_synth)
                    
                    # Determine the minimum allowed integer value
                    # Default to 1 if enforce_positive is on and original min was <=0 or not specified higher
                    min_integer_value = 1
                    if min_constraint > 0 : # If original min (floored at 0) was > 0, use that as base
                        min_integer_value = int(np.ceil(min_constraint))
                        if min_integer_value == 0 and self.enforce_positive: # if original was 0.x, ceil is 1
                             min_integer_value = 1
                    elif not self.enforce_positive and min_constraint == 0 : # if not enforcing positivity and orig can be 0
                        min_integer_value = 0


                    current_col_synth = np.maximum(current_col_synth, min_integer_value)
                    result[col] = np.clip(current_col_synth, min_integer_value, int(np.round(orig_max_val))).astype(int)
        return result

    def _preserve_correlations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhance correlation preservation."""
        numerical_cols = [col for col, info in self.column_info.items()
                          if info['type'] == 'numerical' and col in data.columns]

        if len(numerical_cols) < 2:
            return data

        result = data.copy()
        
        # Ensure original_data numerical_cols exist and are numeric
        valid_original_numerical_cols = [col for col in numerical_cols if col in self.original_data.columns and pd.api.types.is_numeric_dtype(self.original_data[col])]
        if len(valid_original_numerical_cols) < 2:
            return data
            
        orig_corr = self.original_data[valid_original_numerical_cols].corr().fillna(0)
        
        # Ensure result numerical_cols exist and are numeric
        valid_result_numerical_cols = [col for col in numerical_cols if col in result.columns and pd.api.types.is_numeric_dtype(result[col])]
        if len(valid_result_numerical_cols) < 2: # Should be same as valid_original_numerical_cols ideally
             return data
        
        # Align columns for Cholesky decomposition
        common_cols = list(set(valid_original_numerical_cols) & set(valid_result_numerical_cols))
        if len(common_cols) < 2:
            return data

        orig_corr_common = self.original_data[common_cols].corr().fillna(0)
        current_data_common = result[common_cols].copy()


        try:
            # Normalize current data (handling potential zero std dev)
            normalized_data = pd.DataFrame(index=current_data_common.index)
            means = {}
            stds = {}
            for col in common_cols:
                mean = current_data_common[col].mean()
                std = current_data_common[col].std()
                means[col] = mean
                stds[col] = std if std > 1e-6 else 1.0 # Avoid division by zero
                normalized_data[col] = (current_data_common[col] - mean) / stds[col]

            # Add small identity matrix to ensure positive definiteness for Cholesky
            identity_matrix = np.eye(len(common_cols)) * 1e-6
            L_target = np.linalg.cholesky(orig_corr_common + identity_matrix)
            
            # Transform data
            # Note: This step assumes the data is already somewhat decorrelated or that
            # applying L_target directly imposes the target correlation. More advanced
            # methods might involve transforming current data to be uncorrelated first.
            # For simplicity, we directly apply the target Cholesky factor.
            # This is a simplification of some correlation imposition techniques.
            corrected_normalized = normalized_data.dot(L_target.T) # This might not be the standard way

            # Denormalize using original data's stats for target scale
            for col in common_cols:
                orig_mean = self.original_data[col].mean()
                orig_std = self.original_data[col].std()
                if orig_std > 1e-6 : # Avoid division by zero for original std
                     result[col] = corrected_normalized[col] * orig_std + orig_mean
                else: # if original std is zero, set to original mean (constant column)
                     result[col] = orig_mean


        except np.linalg.LinAlgError as e:
            # print(f"Warning: Cholesky decomposition failed in _preserve_correlations: {e}. Skipping correlation adjustment.")
            pass # Fallback to data without this specific correlation adjustment attempt

        return result

    def _optimize_distributions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize statistical distributions of synthetic data."""
        result = data.copy()

        for col, info in self.column_info.items():
            if col not in result.columns or col not in self.original_data.columns:
                continue

            if info['type'] == 'numerical' and not info.get('distribution_info', {}).get('is_integer', False):
                orig_series = self.original_data[col]
                synth_series = result[col].copy()

                orig_mean = orig_series.mean()
                orig_std = orig_series.std()
                
                # Ensure synth_series is positive for Box-Cox by shifting with a small constant
                # (Box-Cox requires positive data)
                min_synth = synth_series.min()
                shift = 0
                if min_synth <= 0:
                    shift = abs(min_synth) + 1e-6 # Add small epsilon
                
                synth_series_shifted = synth_series + shift

                try:
                    # Apply Box-Cox transformation
                    transformed_synth, _ = stats.boxcox(synth_series_shifted)
                    
                    # Standardize the transformed data
                    ts_mean = transformed_synth.mean()
                    ts_std = transformed_synth.std()

                    if ts_std > 1e-6: # Avoid division by zero
                        standardized_transformed_synth = (transformed_synth - ts_mean) / ts_std
                        # Rescale to original mean and std
                        rescaled_synth = standardized_transformed_synth * orig_std + orig_mean
                    else: # If std is zero, implies constant data after transform
                        rescaled_synth = np.full_like(transformed_synth, orig_mean)

                    # Inverse shift (if any shift was applied)
                    # This step is tricky because Box-Cox inverse is non-trivial and
                    # we did not store lambda. A common approach is to use quantile matching
                    # or to ensure the result is then passed to _enforce_constraints.
                    # For now, we'll rely on _enforce_constraints to fix ranges.
                    # The primary goal here is to adjust the shape of the distribution.
                    
                    result[col] = rescaled_synth # The range will be fixed by _enforce_constraints

                except (ValueError, TypeError, FloatingPointError) as e:
                    # print(f"Warning: Box-Cox transformation failed for column {col}: {e}. Skipping distribution optimization.")
                    pass # Fallback if transformation fails

        return result


    def _calculate_quality_score(self, synthetic_data: pd.DataFrame) -> float:
        """Calculate an enhanced quality score with stricter criteria."""
        score = 100.0  # Start with a perfect score

        if self.original_data is None or self.column_info is None:
            return 0.0 # Cannot calculate score without original data context

        for col, info in self.column_info.items():
            if col not in synthetic_data.columns or col not in self.original_data.columns:
                score -= 10  # Higher penalty for missing columns
                continue

            orig_data_col = self.original_data[col]
            synth_data_col = synthetic_data[col]

            if info['type'] == 'numerical':
                dist_info = info.get('distribution_info', {})
                is_integer = dist_info.get('is_integer', False)
                
                # Check for negative values if positivity is expected
                expected_min = 0.0
                if is_integer:
                    expected_min = 1.0 # Assuming integers are made >= 1 by constraints
                    # More precise expected min for integers if original min was higher:
                    orig_min_for_int = dist_info.get('min', orig_data_col.min())
                    if orig_min_for_int > 0:
                        expected_min = np.ceil(orig_min_for_int)
                    if dist_info.get('min',0) == 0 and not self.enforce_positive: # if original could be 0 and not forcing positive
                        expected_min = 0

                # Allow tiny tolerance for float comparisons to expected_min (e.g. 1e-6)
                # synth_data_col.min() could be problematic if col is empty or all NaN
                if synth_data_col.notna().sum() == 0: # if all NaNs
                    score -= 15 # penalty for all NaN column
                    continue

                actual_min_synth = synth_data_col.dropna().min()

                if actual_min_synth < (expected_min - (1e-7 if not is_integer else 0)) : # stricter check
                    score -= 20  # Heavy penalty for not meeting positivity/integer minimum

                # Compare distributions (mean and std)
                orig_mean = orig_data_col.mean()
                synth_mean = synth_data_col.mean()
                orig_std = orig_data_col.std()
                synth_std = synth_data_col.std()

                # Handle cases where original mean/std might be zero or very small
                mean_denominator = max(abs(orig_mean), 0.1)
                std_denominator = max(orig_std, 0.1)

                mean_diff_penalty = abs(orig_mean - synth_mean) / mean_denominator
                std_diff_penalty = abs(orig_std - synth_std) / std_denominator
                
                # Cap penalties to avoid excessive score reduction from one metric
                score -= min(mean_diff_penalty * 15, 15) # Max 15 points penalty for mean
                score -= min(std_diff_penalty * 15, 15) # Max 15 points penalty for std

            elif info['type'] == 'categorical':
                orig_counts = orig_data_col.value_counts(normalize=True)
                synth_counts = synth_data_col.value_counts(normalize=True)
                
                # Calculate Total Variation Distance (TVD) or similar for categorical distribution
                tvd = 0.0
                all_categories = set(orig_counts.index) | set(synth_counts.index)
                for val in all_categories:
                    orig_freq = orig_counts.get(val, 0)
                    synth_freq = synth_counts.get(val, 0)
                    tvd += abs(orig_freq - synth_freq)
                tvd /= 2 # Normalize TVD to be between 0 and 1

                score -= tvd * 30 # Penalty based on TVD (max 30 points)
                
        return max(0, min(100, score)) # Ensure score is between 0 and 100


    # Methods _preprocess_data, _build_models, _capture_correlations, 
    # _generate_correlated_data, _post_process define an alternative GMM-based synthesis.
    # They are kept for completeness but the primary flow uses ACTGAN via fit/generate.

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for modeling (GMM path)."""
        processed_data = pd.DataFrame(index=data.index)

        for col, info in self.column_info.items():
            if col not in data.columns:
                continue

            if info['type'] == 'numerical':
                scaler = StandardScaler()
                values = data[col].fillna(data[col].median()).values.reshape(-1, 1)
                processed_data[col] = scaler.fit_transform(values).flatten()
                self.scalers[col] = scaler
            elif info['type'] == 'categorical':
                encoder = LabelEncoder()
                values = data[col].fillna('Unknown').astype(str)
                processed_data[col] = encoder.fit_transform(values)
                self.encoders[col] = encoder
            elif info['type'] == 'datetime':
                dt_values = pd.to_datetime(data[col])
                min_date = dt_values.min()
                self.encoders[col] = min_date # Store min date for reconstruction
                days_since_min = (dt_values - min_date).dt.days.fillna(0)
                scaler = StandardScaler()
                processed_data[f'{col}_days'] = scaler.fit_transform(days_since_min.values.reshape(-1, 1)).flatten()
                self.scalers[f'{col}_days'] = scaler # Store scaler for the numeric part
            elif info['type'] == 'boolean':
                processed_data[col] = data[col].fillna(False).astype(bool).astype(int)
        return processed_data

    def _build_models(self, processed_data: pd.DataFrame):
        """Build statistical models for data generation (GMM path)."""
        for col in processed_data.columns: # Columns might include '{col}_days' for datetimes
            if col not in self.column_info and not col.endswith('_days'): # Skip auxiliary columns if not directly in column_info
                 # unless it's a known transformation like datetime to days
                is_datetime_derived = any(dt_col for dt_col in self.column_info if f"{dt_col}_days" == col and self.column_info[dt_col]['type'] == 'datetime')
                if not is_datetime_derived:
                    continue

            try:
                unique_vals_count = len(processed_data[col].unique())
                n_components = min(max(1,unique_vals_count // 2 if unique_vals_count > 1 else 1), 5) # Adjusted n_components logic

                gmm = GaussianMixture(
                    n_components=n_components,
                    random_state=self.random_seed,
                    covariance_type='full'
                )
                col_data = processed_data[col].values.reshape(-1, 1)
                gmm.fit(col_data)
                self.models[col] = gmm
            except Exception as e:
                # print(f"Warning: Could not fit GMM for column {col}: {e}. Using fallback statistics.")
                self.models[col] = {
                    'mean': processed_data[col].mean(),
                    'std': processed_data[col].std(),
                    'min': processed_data[col].min(), # Store min/max for fallback
                    'max': processed_data[col].max(),
                    'unique_values': processed_data[col].unique() if unique_vals_count < 20 else None
                }

    def _capture_correlations(self, processed_data: pd.DataFrame):
        """Capture correlation structure using PCA and correlation matrix (GMM path)."""
        try:
            # Select only numeric columns for correlation matrix and PCA
            numeric_cols_for_pca = [col for col in processed_data.columns if pd.api.types.is_numeric_dtype(processed_data[col])]
            if not numeric_cols_for_pca:
                self.correlation_matrix = pd.DataFrame()
                self.pca_model = None
                return

            self.correlation_matrix = processed_data[numeric_cols_for_pca].corr().fillna(0)
            if len(numeric_cols_for_pca) > 1:
                # Ensure n_components is valid
                n_comp = min(len(numeric_cols_for_pca), len(processed_data))
                if n_comp > 0:
                    pca = PCA(n_components=n_comp, random_state=self.random_seed)
                    pca.fit(processed_data[numeric_cols_for_pca].fillna(0))
                    self.pca_model = pca
                else:
                    self.pca_model = None
            else:
                self.pca_model = None
        except Exception as e:
            # print(f"Warning: Could not capture correlations: {e}")
            self.correlation_matrix = pd.DataFrame()
            self.pca_model = None
            
    def _generate_correlated_data(self, num_rows: int) -> pd.DataFrame:
        """Generate synthetic data preserving correlations (GMM path)."""
        synthetic_data = pd.DataFrame()

        # Generate base synthetic data for each column
        for col, model_or_stats in self.models.items():
            try:
                if hasattr(model_or_stats, 'sample'): # GMM model
                    generated_values, _ = model_or_stats.sample(num_rows)
                    synthetic_data[col] = generated_values.flatten()
                else: # Fallback statistics
                    stats = model_or_stats
                    if stats.get('unique_values') is not None and len(stats['unique_values']) > 0 :
                        synthetic_data[col] = np.random.choice(stats['unique_values'], num_rows)
                    else: # Continuous data from normal distribution
                        # Ensure std is non-negative and reasonably small if mean is near zero
                        std_val = max(stats.get('std', 0.1), 1e-6)
                        synthetic_data[col] = np.random.normal(stats.get('mean',0), std_val, num_rows)
            except Exception as e:
                # print(f"Warning: Error generating column {col} with GMM/fallback: {e}")
                # Emergency fallback: sample from original preprocessed data for this column if possible
                # This part is complex as original_data is not preprocessed here.
                # A simple fallback: use a default value (e.g., 0 or mean from stats)
                synthetic_data[col] = np.full(num_rows, model_or_stats.get('mean', 0) if isinstance(model_or_stats, dict) else 0)


        # Apply correlation adjustments if PCA model exists
        numeric_cols_in_synthetic = [c for c in synthetic_data.columns if pd.api.types.is_numeric_dtype(synthetic_data[c])]
        
        if self.pca_model is not None and self.correlation_matrix.shape[0] > 1 and \
           set(self.correlation_matrix.columns) == set(numeric_cols_in_synthetic): # Check if PCA was fit on these cols
            
            # Ensure columns are in the same order as when PCA was fitted
            ordered_synthetic_numeric_data = synthetic_data[self.correlation_matrix.columns].fillna(0)

            try:
                pca_transformed = self.pca_model.transform(ordered_synthetic_numeric_data)
                noise_scale = 0.05 # Reduced noise
                pca_transformed += np.random.normal(0, noise_scale, pca_transformed.shape) * self.pca_model.explained_variance_ratio_
                reconstructed_numeric = self.pca_model.inverse_transform(pca_transformed)
                
                reconstructed_df = pd.DataFrame(reconstructed_numeric, columns=self.correlation_matrix.columns, index=synthetic_data.index)
                for r_col in reconstructed_df.columns:
                    synthetic_data[r_col] = reconstructed_df[r_col]

            except Exception as e:
                # print(f"Warning: Could not apply PCA correlation adjustment: {e}")
                pass
        return synthetic_data

    def _post_process(self, synthetic_data_processed: pd.DataFrame) -> pd.DataFrame:
        """Post-process synthetic data to restore original formats (GMM path)."""
        final_data = pd.DataFrame(index=synthetic_data_processed.index)

        for original_col_name, info in self.column_info.items():
            try:
                if info['type'] == 'numerical':
                    if original_col_name in self.scalers and original_col_name in synthetic_data_processed:
                        values_scaled = synthetic_data_processed[original_col_name].values.reshape(-1, 1)
                        values = self.scalers[original_col_name].inverse_transform(values_scaled).flatten()
                    elif original_col_name in synthetic_data_processed:
                        values = synthetic_data_processed[original_col_name].values
                    else:
                        continue # Column not found in processed data

                    # Apply original constraints (min, max, integer, positivity)
                    dist_info = info.get('distribution_info', {})
                    orig_min = dist_info.get('min', self.original_data[original_col_name].min())
                    orig_max = dist_info.get('max', self.original_data[original_col_name].max())
                    
                    # Enforce positivity based on self.enforce_positive and original min
                    effective_min = max(0, orig_min) # Baseline min is 0 or original min if >0
                    if self.enforce_positive:
                        if effective_min == 0 : # If original could be 0, make it epsilon for floats
                             values = np.maximum(values, 1e-6 if not dist_info.get('is_integer', False) else 1)
                        else: # if original min was >0, enforce that
                             values = np.maximum(values, effective_min)
                    else: # if not enforcing strict positivity, just ensure >= original min (which could be negative if not handled)
                        values = np.maximum(values, orig_min)


                    values = np.clip(values, effective_min if self.enforce_positive else orig_min , orig_max)

                    if dist_info.get('is_integer', False):
                        min_int_val = 1 if self.enforce_positive and effective_min ==0 else int(np.round(effective_min))
                        values = np.maximum(min_int_val, values.round()).astype(int)
                    
                    final_data[original_col_name] = values

                elif info['type'] == 'categorical':
                    if original_col_name in self.encoders and original_col_name in synthetic_data_processed:
                        # Ensure encoded values are valid integers for the encoder
                        encoded_values = synthetic_data_processed[original_col_name].round().astype(int)
                        max_label = len(self.encoders[original_col_name].classes_) - 1
                        encoded_values = np.clip(encoded_values, 0, max_label)
                        final_data[original_col_name] = self.encoders[original_col_name].inverse_transform(encoded_values)
                    elif original_col_name in synthetic_data_processed:
                         final_data[original_col_name] = synthetic_data_processed[original_col_name] # No encoder found

                elif info['type'] == 'datetime':
                    datetime_numeric_col = f'{original_col_name}_days'
                    if datetime_numeric_col in self.scalers and datetime_numeric_col in synthetic_data_processed:
                        days_values_scaled = synthetic_data_processed[datetime_numeric_col].values.reshape(-1,1)
                        days_values = self.scalers[datetime_numeric_col].inverse_transform(days_values_scaled).flatten()
                        min_date = self.encoders[original_col_name] # Stored min_date
                        final_data[original_col_name] = min_date + pd.to_timedelta(np.round(days_values), unit='D')
                    else: # Fallback for datetime
                        final_data[original_col_name] = pd.Timestamp('1970-01-01')


                elif info['type'] == 'boolean':
                    if original_col_name in synthetic_data_processed:
                        final_data[original_col_name] = synthetic_data_processed[original_col_name] > 0.5
                    
            except Exception as e:
                # print(f"Warning: Error post-processing column {original_col_name}: {e}. Using fallback sampling.")
                if original_col_name in self.original_data: # Check if original column exists
                    original_sample_col = self.original_data[original_col_name].dropna()
                    if not original_sample_col.empty:
                        final_data[original_col_name] = np.random.choice(original_sample_col, len(synthetic_data_processed), replace=True)
                    else: # If original column is all NaN or empty
                        final_data[original_col_name] = [None] * len(synthetic_data_processed)


        return final_data


    def _enforce_uniqueness(self, data: pd.DataFrame, unique_columns: List[str]) -> pd.DataFrame:
        """Enforce uniqueness constraints on specified columns."""
        result_data = data.copy()

        for col in unique_columns:
            if col in result_data.columns and col in self.column_info:
                # Ensure no NaNs if uniqueness is critical; fill or handle as per policy
                # For simplicity, we proceed assuming NaNs are not the values needing unique replacement strategy
                if result_data[col].hasnans:
                    # print(f"Warning: Column {col} for uniqueness check has NaNs. Uniqueness might not be fully guaranteed for NaNs.")
                    pass # Decide on a NaN strategy if needed, e.g., fill them uniquely or ignore

                duplicates = result_data.duplicated(subset=[col], keep='first')
                if duplicates.any():
                    dup_indices = result_data.index[duplicates]
                    col_type = self.column_info[col]['type']

                    if col_type == 'numerical':
                        is_integer = self.column_info[col].get('distribution_info',{}).get('is_integer',False)
                        for idx in dup_indices:
                            # Find a new unique value. This can be complex.
                            # Simple strategy: add small increment or find next available integer.
                            current_val = result_data.loc[idx, col]
                            increment = 0.001 if not is_integer else 1
                            new_val = current_val + increment
                            # Ensure new_val is not already taken and within bounds
                            # This loop can be inefficient for many duplicates.
                            while new_val in result_data[col].values or \
                                  (self.column_info[col].get('distribution_info',{}).get('max') is not None and \
                                   new_val > self.column_info[col]['distribution_info']['max']):
                                new_val += increment
                                if self.column_info[col].get('distribution_info',{}).get('max') is not None and \
                                   new_val > self.column_info[col]['distribution_info']['max']:
                                    # If max is hit, try decrementing or another strategy (could lead to issues)
                                    # print(f"Could not make {col} unique for value {current_val} due to max bound.")
                                    break 
                            result_data.loc[idx, col] = new_val

                    elif col_type == 'categorical' or col_type == 'text': # Assuming text might be treated as categorical for this
                        for i, idx in enumerate(dup_indices):
                            original_val = str(result_data.loc[idx, col])
                            suffix_counter = 1
                            new_val = f"{original_val}_uniq{suffix_counter}"
                            while new_val in result_data[col].values:
                                suffix_counter += 1
                                new_val = f"{original_val}_uniq{suffix_counter}"
                            result_data.loc[idx, col] = new_val
                    
                    elif col_type == 'datetime':
                        for idx in dup_indices:
                            current_val = pd.to_datetime(result_data.loc[idx, col])
                            timedelta_increment = pd.Timedelta(seconds=1)
                            new_val = current_val + timedelta_increment
                            while new_val in result_data[col].values: # This check can be slow for many values
                                new_val += timedelta_increment
                            result_data.loc[idx, col] = new_val
            # else:
                # print(f"Warning: Column {col} for uniqueness not found in data or column_info.")
        return result_data

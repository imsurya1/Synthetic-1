
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, PowerTransformer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedSynthesizer:
    def __init__(self, random_seed: Optional[int] = None):
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

        # Enhanced model components
        self.scalers = {}
        self.models = {}
        self.correlation_matrix = None
        self.feature_correlations = None
        self.pca_model = None

        # Quality parameters
        self.quality_threshold = 98  # Increased threshold
        self.max_optimization_attempts = 100  # More attempts
        self.convergence_threshold = 0.0001  # Tighter convergence

        # Distribution parameters
        self.min_value = 1e-8  # Smaller minimum value
        self.distribution_similarity_threshold = 0.98  # Higher similarity requirement
        self.moment_matching_threshold = 0.005  # Stricter moment matching

    def fit(self, data: pd.DataFrame, column_info: Dict[str, Any]):
        """Fit the synthesizer to the training data"""
        self.original_data = data.copy()
        self.column_info = column_info
        self.feature_correlations = data.corr()

        for col, info in column_info.items():
            if info['type'] == 'numerical':
                col_data = data[col].values.reshape(-1, 1)
                
                # Handle negative values with improved shifting
                if data[col].min() < 0:
                    shift = abs(data[col].min()) + self.min_value
                    col_data = col_data + shift
                    self.column_info[col]['shift'] = shift

                # Enhanced transformation pipeline
                scaler = PowerTransformer(method='yeo-johnson', standardize=True)
                scaled_data = scaler.fit_transform(col_data)
                self.scalers[col] = scaler

                # Optimized GMM with more components
                n_components = min(15, max(5, len(data[col].unique()) // 4))
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type='full',
                    random_state=self.random_seed,
                    max_iter=2000,
                    tol=1e-6,
                    reg_covar=1e-8,
                    n_init=5
                )
                gmm.fit(scaled_data)
                self.models[col] = gmm

    def generate(self, num_rows: int) -> pd.DataFrame:
        """Generate synthetic data with enhanced quality"""
        best_quality = 0
        best_result = None

        for attempt in range(self.max_optimization_attempts):
            try:
                current_data = self._generate_base_data(num_rows)
                current_data = self._enforce_correlations(current_data)
                current_data = self._post_process(current_data)
                quality_score = self._calculate_quality(current_data)

                if quality_score > best_quality:
                    best_quality = quality_score
                    best_result = current_data.copy()

                    if quality_score >= self.quality_threshold:
                        print(f"Achieved target quality score: {quality_score:.2f}")
                        return best_result

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                continue

        if best_result is not None:
            print(f"Best achieved quality score: {best_quality:.2f}")
            return best_result

        raise ValueError("Failed to generate synthetic data with required quality")

    def _generate_base_data(self, num_rows: int) -> pd.DataFrame:
        """Generate initial synthetic data with improved sampling"""
        synthetic_data = pd.DataFrame()

        for col, info in self.column_info.items():
            if info['type'] == 'numerical':
                gmm = self.models[col]
                scaled_samples = gmm.sample(num_rows)[0]
                
                # Apply inverse transform with noise reduction
                samples = self.scalers[col].inverse_transform(scaled_samples)
                
                # Remove shift if applied during fitting
                if 'shift' in info:
                    samples = samples - info['shift']
                
                # Ensure non-negative values where required
                if self.original_data[col].min() >= 0:
                    samples = np.maximum(samples, self.min_value)

                synthetic_data[col] = samples.flatten()

        return synthetic_data

    def _enforce_correlations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enforce correlation structure with improved precision"""
        if len(data.columns) < 2:
            return data

        target_corr = self.feature_correlations.fillna(0)
        L = np.linalg.cholesky(np.clip(target_corr, -1 + 1e-8, 1 - 1e-8))

        # Improved correlation enforcement
        scaled_data = data.apply(lambda x: (x - x.mean()) / (x.std() + 1e-8))
        transformed_data = pd.DataFrame(
            np.dot(scaled_data, L), 
            columns=data.columns
        )

        # Restore scales with precision
        for col in data.columns:
            transformed_data[col] = (
                transformed_data[col] * data[col].std() + data[col].mean()
            )

        return transformed_data

    def _post_process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced post-processing for better quality"""
        result = data.copy()

        for col, info in self.column_info.items():
            if info['type'] == 'numerical':
                # Ensure positive values where required
                if self.original_data[col].min() >= 0:
                    result[col] = np.maximum(result[col], self.min_value)

                # Enhanced moment matching
                orig_mean = self.original_data[col].mean()
                orig_std = self.original_data[col].std()
                orig_skew = self.original_data[col].skew()

                # Multi-moment matching
                if result[col].std() > 0:
                    result[col] = (
                        (result[col] - result[col].mean()) / result[col].std() * orig_std + orig_mean
                    )

                    # Match skewness direction
                    if np.sign(result[col].skew()) != np.sign(orig_skew):
                        result[col] = -result[col] + 2 * orig_mean

        return result

    def _calculate_quality(self, synthetic_data: pd.DataFrame) -> float:
        """Calculate quality score with enhanced metrics"""
        quality_score = 100.0

        for col in synthetic_data.columns:
            orig_data = self.original_data[col]
            synth_data = synthetic_data[col]

            # Enhanced statistical tests
            ks_stat, _ = stats.ks_2samp(orig_data, synth_data)
            quality_score -= ks_stat * 15

            # Improved moment matching penalties
            mean_diff = abs(orig_data.mean() - synth_data.mean()) / (abs(orig_data.mean()) + 1e-8)
            std_diff = abs(orig_data.std() - synth_data.std()) / (orig_data.std() + 1e-8)
            skew_diff = abs(orig_data.skew() - synth_data.skew()) / (abs(orig_data.skew()) + 1e-8)

            quality_score -= mean_diff * 8
            quality_score -= std_diff * 8
            quality_score -= skew_diff * 4

            # Strict penalty for negative values where not allowed
            if self.original_data[col].min() >= 0 and synthetic_data[col].min() < 0:
                quality_score -= 25

        # Enhanced correlation quality check
        if len(synthetic_data.columns) > 1:
            corr_diff = np.abs(
                self.feature_correlations - synthetic_data.corr()
            ).mean().mean()
            quality_score -= corr_diff * 10

        return max(0, min(100, quality_score))

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedSynthesizer:
    def __init__(self, random_seed: Optional[int] = None):
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

        self.fitted = False
        self.original_data = None
        self.column_info = None
        self.scalers = {}
        self.encoders = {}
        self.models = {}
        self.correlation_matrix = None
        self.pca_model = None
        self.quality_target = 98
        self.min_quality_threshold = 96

        # Strict quality parameters
        self.min_positive_value = 1e-10
        self.distribution_similarity_threshold = 0.98
        self.correlation_tolerance = 0.02
        self.max_attempts = 20
        self.quality_boost_factor = 1.2
        self.precision_factor = 1e-8

    def fit(self, data: pd.DataFrame, column_info: Dict[str, Any]):
        self.original_data = data.copy()
        self.column_info = column_info

        # Preprocess with enhanced precision
        for col, info in column_info.items():
            if info['type'] == 'numerical' and col in data.columns:
                if data[col].min() < 0:
                    shift_value = abs(data[col].min()) + self.precision_factor
                    self.column_info[col]['shift_value'] = shift_value
                    self.original_data[col] = data[col] + shift_value

        processed_data = self._preprocess_data(self.original_data)
        self._build_precise_models(processed_data)
        self._capture_correlations(processed_data)
        self.fitted = True

    def generate(self, num_rows: int, unique_columns: List[str] = None) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Must fit synthesizer before generating")

        best_quality = 0
        best_data = None

        for attempt in range(self.max_attempts):
            try:
                synthetic_data = self._generate_base_data(num_rows)
                synthetic_data = self._enforce_positivity(synthetic_data)
                synthetic_data = self._preserve_correlations(synthetic_data)
                synthetic_data = self._optimize_distributions(synthetic_data)
                synthetic_data = self._post_process(synthetic_data)

                if unique_columns:
                    synthetic_data = self._enforce_uniqueness(synthetic_data, unique_columns)

                quality_score = self._calculate_quality_score(synthetic_data)

                if quality_score > best_quality:
                    best_quality = quality_score
                    best_data = synthetic_data.copy()

                if quality_score >= self.quality_target:
                    print(f"Achieved quality score: {quality_score:.2f}%")
                    return best_data

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                continue

        if best_data is None:
            raise ValueError("Failed to generate synthetic data")

        return best_data

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        processed_data = pd.DataFrame()

        for col, info in self.column_info.items():
            if col not in data.columns:
                continue

            if info['type'] == 'numerical':
                values = data[col].fillna(data[col].median())
                values = np.maximum(values, self.min_positive_value)

                transformer = PowerTransformer(method='yeo-johnson', standardize=True)
                processed_values = transformer.fit_transform(values.values.reshape(-1, 1)).flatten()
                processed_data[col] = processed_values
                self.scalers[col] = transformer

            elif info['type'] == 'categorical':
                encoder = LabelEncoder()
                values = data[col].fillna('Unknown').astype(str)
                processed_data[col] = encoder.fit_transform(values)
                self.encoders[col] = encoder

            elif info['type'] == 'datetime':
                dt_values = pd.to_datetime(data[col])
                min_date = dt_values.min()
                days_since_min = (dt_values - min_date).dt.days.fillna(0)
                
                # Ensure positive days
                days_since_min = np.maximum(days_since_min, 0)
                
                scaler = StandardScaler()
                processed_values = scaler.fit_transform(days_since_min.values.reshape(-1, 1)).flatten()
                processed_data[col] = processed_values
                self.scalers[col] = scaler
                self.encoders[col] = min_date

            elif info['type'] == 'boolean':
                bool_values = data[col].fillna(False).astype(bool).astype(int)
                processed_data[col] = bool_values

        return processed_data

    def _build_precise_models(self, data: pd.DataFrame):
        self.models = {}

        for col in data.columns:
            col_data = data[col].dropna()

            if len(col_data.unique()) < 10:
                value_counts = col_data.value_counts(normalize=True)
                self.models[col] = {
                    'type': 'categorical',
                    'values': value_counts.index.tolist(),
                    'probabilities': value_counts.values.tolist()
                }
            else:
                models = {}

                # Enhanced GMM
                n_components = min(10, max(3, len(col_data.unique()) // 5))
                gmm = GaussianMixture(
                    n_components=n_components,
                    random_state=self.random_seed,
                    covariance_type='full',
                    max_iter=1000,
                    tol=1e-8,
                    reg_covar=1e-8,
                    n_init=10
                )
                gmm.fit(col_data.values.reshape(-1, 1))
                models['gmm'] = gmm

                # Capture statistics
                self.models[col] = {
                    'type': 'continuous',
                    'models': models,
                    'stats': {
                        'mean': col_data.mean(),
                        'std': col_data.std(),
                        'min': col_data.min(),
                        'max': col_data.max(),
                        'skew': col_data.skew(),
                        'kurt': col_data.kurtosis()
                    }
                }

    def _capture_correlations(self, data: pd.DataFrame):
        try:
            self.correlation_matrix = data.corr(method='spearman').fillna(0)

            if len(data.columns) > 1:
                self.pca_model = PCA(n_components=min(len(data.columns), len(data) // 2))
                self.pca_model.fit(data.fillna(data.median()))
                self.pca_explained_variance = self.pca_model.explained_variance_ratio_

        except Exception as e:
            print(f"Warning: Correlation capture failed: {e}")
            self.correlation_matrix = pd.DataFrame()
            self.pca_model = None

    def _generate_base_data(self, num_rows: int) -> pd.DataFrame:
        synthetic_data = pd.DataFrame()

        for col, model_info in self.models.items():
            if model_info['type'] == 'categorical':
                values = np.random.choice(
                    model_info['values'],
                    size=num_rows,
                    p=model_info['probabilities']
                )
                synthetic_data[col] = values

            else:
                gmm = model_info['models']['gmm']
                values = gmm.sample(num_rows)[0].flatten()
                synthetic_data[col] = values

        return synthetic_data

    def _enforce_positivity(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()

        for col, info in self.column_info.items():
            if info['type'] == 'numerical':
                values = np.maximum(result[col].values, self.min_positive_value)
                orig_col = self.original_data[col]

                if 'shift_value' in info:
                    orig_col = orig_col - info['shift_value']

                target_mean = max(orig_col.mean(), self.min_positive_value)
                target_std = orig_col.std()

                if values.std() > 0:
                    values = (values - values.mean()) / values.std()
                    values = values * target_std + target_mean

                values = np.maximum(values, self.min_positive_value)
                result[col] = values

        return result

    def _preserve_correlations(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(data.columns) < 2:
            return data

        result = data.copy()
        numerical_cols = [col for col, info in self.column_info.items() 
                         if info['type'] == 'numerical' and col in data.columns]

        if len(numerical_cols) >= 2:
            current_data = result[numerical_cols]
            target_corr = self.original_data[numerical_cols].corr()

            eigenvals = np.maximum(np.linalg.eigvals(target_corr), self.precision_factor)
            target_corr = target_corr + np.eye(len(target_corr)) * self.precision_factor

            try:
                L = np.linalg.cholesky(target_corr)
                standardized = current_data.apply(lambda x: (x - x.mean()) / (x.std() + self.precision_factor))
                corrected = standardized @ L.T

                for i, col in enumerate(numerical_cols):
                    orig_mean = max(self.original_data[col].mean(), self.min_positive_value)
                    orig_std = self.original_data[col].std()
                    result[col] = corrected.iloc[:, i] * orig_std + orig_mean
                    result[col] = np.maximum(result[col], self.min_positive_value)

            except Exception as e:
                print(f"Warning: Correlation preservation failed: {e}")

        return result

    def _optimize_distributions(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()

        for col, info in self.column_info.items():
            if info['type'] == 'numerical':
                orig_col = self.original_data[col]
                synth_col = result[col]

                synth_col = np.maximum(synth_col, self.min_positive_value)
                orig_mean = max(orig_col.mean(), self.min_positive_value)
                orig_std = orig_col.std()

                if synth_col.std() > 0:
                    standardized = (synth_col - synth_col.mean()) / synth_col.std()
                    result[col] = standardized * orig_std + orig_mean
                    result[col] = np.maximum(result[col], self.min_positive_value)

        return result

    def _post_process(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()

        for col, info in self.column_info.items():
            if info['type'] == 'numerical':
                if col in self.scalers:
                    values = self.scalers[col].inverse_transform(
                        result[col].values.reshape(-1, 1)
                    ).flatten()
                else:
                    values = result[col].values

                values = np.maximum(values, self.min_positive_value)

                if 'shift_value' in info:
                    values = values - info['shift_value']
                    values = np.maximum(values, self.min_positive_value)

                result[col] = values

            elif info['type'] == 'categorical':
                if col in self.encoders:
                    result[col] = self.encoders[col].inverse_transform(
                        result[col].round().astype(int)
                    )

            elif info['type'] == 'datetime':
                if col in self.scalers and col in self.encoders:
                    days_values = self.scalers[col].inverse_transform(
                        result[col].values.reshape(-1, 1)
                    ).flatten()
                    days_values = np.maximum(days_values, 0)  # Ensure positive days
                    min_date = self.encoders[col]
                    result[col] = min_date + pd.to_timedelta(days_values.round(), unit='D')

            elif info['type'] == 'boolean':
                result[col] = result[col] > 0.5

        return result

    def _calculate_quality_score(self, synthetic_data: pd.DataFrame) -> float:
        score = 100.0
        total_weight = 0

        for col, info in self.column_info.items():
            if col not in synthetic_data.columns:
                score -= 20
                continue

            col_weight = 1.0
            col_score = 100.0

            if info['type'] == 'numerical':
                orig_data = self.original_data[col]
                synth_data = synthetic_data[col]

                if synth_data.min() < 0:
                    col_score = 0
                else:
                    # Statistical tests
                    ks_stat, _ = stats.ks_2samp(orig_data, synth_data)
                    col_score -= ks_stat * 20

                    # Moment matching
                    mean_diff = abs(orig_data.mean() - synth_data.mean()) / (abs(orig_data.mean()) + self.precision_factor)
                    std_diff = abs(orig_data.std() - synth_data.std()) / (orig_data.std() + self.precision_factor)

                    col_score -= mean_diff * 10
                    col_score -= std_diff * 10

            elif info['type'] == 'categorical':
                orig_counts = self.original_data[col].value_counts(normalize=True)
                synth_counts = synthetic_data[col].value_counts(normalize=True)

                for val in orig_counts.index:
                    orig_freq = orig_counts[val]
                    synth_freq = synth_counts.get(val, 0)
                    col_score -= abs(orig_freq - synth_freq) * 20

            score += (col_score - 100) * col_weight
            total_weight += col_weight

        if len(synthetic_data.columns) > 1:
            orig_corr = self.original_data.corr()
            synth_corr = synthetic_data.corr()
            corr_diff = np.abs(orig_corr - synth_corr).mean().mean()
            correlation_score = max(0, 100 - corr_diff * 100)
            score = 0.8 * score + 0.2 * correlation_score

        return max(0, min(100, score))

    def _enforce_uniqueness(self, data: pd.DataFrame, unique_columns: List[str]) -> pd.DataFrame:
        result = data.copy()

        for col in unique_columns:
            duplicates = result.duplicated(subset=[col], keep='first')

            if duplicates.any():
                dup_indices = result.index[duplicates]

                if self.column_info[col]['type'] == 'numerical':
                    base_values = result.loc[dup_indices, col].values
                    increments = np.arange(1, len(dup_indices) + 1) * 0.001
                    new_values = base_values + increments
                    new_values = np.maximum(new_values, self.min_positive_value)
                    result.loc[dup_indices, col] = new_values

                elif self.column_info[col]['type'] == 'categorical':
                    for i, idx in enumerate(dup_indices):
                        original_val = str(result.loc[idx, col])
                        result.loc[idx, col] = f"{original_val}_{i+1}"

                elif self.column_info[col]['type'] == 'datetime':
                    for idx in dup_indices:
                        random_seconds = np.random.randint(1, 3600)
                        result.loc[idx, col] += pd.Timedelta(seconds=random_seconds)

        return result
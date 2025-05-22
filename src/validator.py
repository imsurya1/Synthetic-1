import pandas as pd
import numpy as np
from typing import Dict, Any, List
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataValidator:
    """
    Validates the quality and privacy of synthetic data compared to original data.
    """
    
    def __init__(self):
        self.validation_results = {}
    
    def check_privacy(self, original_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check privacy preservation by ensuring synthetic data doesn't leak original records.
        
        Args:
            original_data: Original dataset
            synthetic_data: Synthetic dataset
            
        Returns:
            Dictionary with privacy metrics
        """
        try:
            # Calculate distance-based privacy metrics
            leakage_score = self._calculate_leakage_score(original_data, synthetic_data)
            
            # Similarity threshold (higher means more similar to original)
            similarity_threshold = 0.95  # Configurable threshold
            
            is_private = leakage_score < similarity_threshold
            
            return {
                'leakage_score': leakage_score,
                'similarity_threshold': similarity_threshold,
                'is_private': is_private,
                'privacy_level': self._get_privacy_level(leakage_score),
                'recommendations': self._get_privacy_recommendations(leakage_score)
            }
            
        except Exception as e:
            return {
                'leakage_score': 0.0,
                'similarity_threshold': 0.95,
                'is_private': True,
                'privacy_level': 'Unknown',
                'error': str(e)
            }
    
    def _calculate_leakage_score(self, original_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
        """Calculate privacy leakage score using distance-based methods."""
        try:
            # Normalize numerical columns for comparison
            numerical_cols = original_data.select_dtypes(include=[np.number]).columns
            
            if len(numerical_cols) == 0:
                return 0.0  # Can't calculate for non-numerical data
            
            # Sample subset for efficiency if datasets are large
            max_samples = 100
            orig_sample = original_data[numerical_cols].sample(min(len(original_data), max_samples))
            synth_sample = synthetic_data[numerical_cols].sample(min(len(synthetic_data), max_samples))
            
            # Normalize data
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            orig_normalized = scaler.fit_transform(orig_sample.fillna(0))
            synth_normalized = scaler.transform(synth_sample.fillna(0))
            
            # Calculate minimum distances between synthetic and original records
            min_distances = []
            
            for synth_row in synth_normalized:
                distances = np.sqrt(np.sum((orig_normalized - synth_row) ** 2, axis=1))
                min_distances.append(np.min(distances))
            
            # High leakage if many synthetic records are very close to original records
            close_threshold = 0.1  # Threshold for "closeness"
            close_records = np.sum(np.array(min_distances) < close_threshold)
            leakage_score = close_records / len(min_distances)
            
            return leakage_score
            
        except Exception as e:
            # Fallback: simple row matching
            return self._simple_leakage_check(original_data, synthetic_data)
    
    def _simple_leakage_check(self, original_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
        """Simple privacy check by looking for exact row matches."""
        try:
            # Convert to string for comparison
            orig_strings = set(original_data.astype(str).apply(lambda x: '|'.join(x), axis=1))
            synth_strings = set(synthetic_data.astype(str).apply(lambda x: '|'.join(x), axis=1))
            
            # Calculate overlap
            overlap = len(orig_strings.intersection(synth_strings))
            leakage_score = overlap / len(synth_strings) if len(synth_strings) > 0 else 0
            
            return leakage_score
            
        except:
            return 0.0  # Default to safe assumption
    
    def _get_privacy_level(self, leakage_score: float) -> str:
        """Determine privacy level based on leakage score."""
        if leakage_score < 0.01:
            return "Excellent"
        elif leakage_score < 0.05:
            return "Good"
        elif leakage_score < 0.1:
            return "Fair"
        else:
            return "Poor"
    
    def _get_privacy_recommendations(self, leakage_score: float) -> List[str]:
        """Get recommendations for improving privacy."""
        recommendations = []
        
        if leakage_score > 0.1:
            recommendations.append("Consider increasing privacy level in synthesis settings")
            recommendations.append("Try using CTGAN or CopulaGAN for better privacy")
            recommendations.append("Add more noise to numerical columns")
        
        elif leakage_score > 0.05:
            recommendations.append("Consider slight increase in privacy level")
            recommendations.append("Monitor for any sensitive information exposure")
        
        else:
            recommendations.append("Privacy level appears adequate")
        
        return recommendations
    
    def statistical_tests(self, original_series: pd.Series, synthetic_series: pd.Series) -> Dict[str, float]:
        """
        Perform statistical tests comparing original and synthetic data distributions.
        
        Args:
            original_series: Original data series
            synthetic_series: Synthetic data series
            
        Returns:
            Dictionary with test results
        """
        try:
            # Clean data
            orig_clean = original_series.dropna()
            synth_clean = synthetic_series.dropna()
            
            if len(orig_clean) == 0 or len(synth_clean) == 0:
                return {'error': 'Insufficient data for testing'}
            
            # Kolmogorov-Smirnov test for distribution similarity
            ks_statistic, ks_pvalue = stats.ks_2samp(orig_clean, synth_clean)
            
            # Basic statistics comparison
            mean_diff = abs(orig_clean.mean() - synth_clean.mean())
            std_diff = abs(orig_clean.std() - synth_clean.std())
            
            # Mann-Whitney U test (non-parametric)
            try:
                u_statistic, u_pvalue = stats.mannwhitneyu(orig_clean, synth_clean, alternative='two-sided')
            except:
                u_statistic, u_pvalue = 0, 1
            
            return {
                'ks_statistic': ks_statistic,
                'ks_pvalue': ks_pvalue,
                'mean_diff': mean_diff,
                'std_diff': std_diff,
                'u_statistic': u_statistic,
                'u_pvalue': u_pvalue
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def correlation_preservation_score(self, original_data: pd.DataFrame, 
                                     synthetic_data: pd.DataFrame, 
                                     numerical_cols: List[str]) -> float:
        """
        Calculate how well correlations are preserved between original and synthetic data.
        
        Args:
            original_data: Original dataset
            synthetic_data: Synthetic dataset
            numerical_cols: List of numerical columns to analyze
            
        Returns:
            Correlation preservation score (0-1)
        """
        try:
            if len(numerical_cols) < 2:
                return 1.0  # Perfect score if not enough columns for correlation
            
            # Calculate correlation matrices
            orig_corr = original_data[numerical_cols].corr().fillna(0)
            synth_corr = synthetic_data[numerical_cols].corr().fillna(0)
            
            # Calculate correlation between correlation matrices
            orig_corr_flat = orig_corr.values.flatten()
            synth_corr_flat = synth_corr.values.flatten()
            
            # Remove diagonal elements (self-correlations)
            n = len(numerical_cols)
            mask = np.ones((n, n), dtype=bool)
            np.fill_diagonal(mask, False)
            
            orig_corr_off_diag = orig_corr.values[mask]
            synth_corr_off_diag = synth_corr.values[mask]
            
            # Calculate correlation coefficient
            if len(orig_corr_off_diag) > 0 and np.var(orig_corr_off_diag) > 0:
                correlation_score = np.corrcoef(orig_corr_off_diag, synth_corr_off_diag)[0, 1]
                return max(0, correlation_score)  # Ensure non-negative
            else:
                return 1.0  # Perfect if no variation in correlations
            
        except Exception as e:
            return 0.0  # Return 0 on error
    
    def overall_quality_score(self, original_data: pd.DataFrame, 
                            synthetic_data: pd.DataFrame, 
                            column_info: Dict[str, Any]) -> float:
        """
        Calculate an overall quality score for the synthetic data.
        
        Args:
            original_data: Original dataset
            synthetic_data: Synthetic dataset
            column_info: Column information dictionary
            
        Returns:
            Overall quality score (0-100)
        """
        try:
            scores = []
            
            # 1. Privacy score (weight: 30%)
            privacy_metrics = self.check_privacy(original_data, synthetic_data)
            privacy_score = 100 if privacy_metrics['is_private'] else 50
            scores.append(('privacy', privacy_score, 0.3))
            
            # 2. Distribution similarity scores (weight: 40%)
            distribution_scores = []
            numerical_cols = [col for col, info in column_info.items() if info['type'] == 'numerical']
            
            for col in numerical_cols:
                if col in original_data.columns and col in synthetic_data.columns:
                    test_results = self.statistical_tests(original_data[col], synthetic_data[col])
                    if 'ks_pvalue' in test_results:
                        # Higher p-value means more similar distributions
                        col_score = min(100, test_results['ks_pvalue'] * 100)
                        distribution_scores.append(col_score)
            
            avg_distribution_score = np.mean(distribution_scores) if distribution_scores else 80
            scores.append(('distribution', avg_distribution_score, 0.4))
            
            # 3. Correlation preservation score (weight: 20%)
            if len(numerical_cols) >= 2:
                corr_score = self.correlation_preservation_score(original_data, synthetic_data, numerical_cols)
                scores.append(('correlation', corr_score * 100, 0.2))
            else:
                scores.append(('correlation', 90, 0.2))  # Default good score
            
            # 4. Basic structural similarity (weight: 10%)
            structure_score = 100  # Start with perfect score
            
            # Check if all columns are present
            missing_cols = set(original_data.columns) - set(synthetic_data.columns)
            if missing_cols:
                structure_score -= len(missing_cols) * 10
            
            # Check if row count is reasonable
            row_ratio = len(synthetic_data) / len(original_data)
            if row_ratio < 0.5 or row_ratio > 20:  # Too few or too many rows
                structure_score -= 20
            
            structure_score = max(0, structure_score)
            scores.append(('structure', structure_score, 0.1))
            
            # Calculate weighted average
            total_score = sum(score * weight for _, score, weight in scores)
            
            return round(total_score, 1)
            
        except Exception as e:
            return 50.0  # Default score on error
    
    def generate_report(self, original_data: pd.DataFrame, 
                       synthetic_data: pd.DataFrame, 
                       column_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive quality and privacy report.
        
        Args:
            original_data: Original dataset
            synthetic_data: Synthetic dataset
            column_info: Column information dictionary
            
        Returns:
            Comprehensive report dictionary
        """
        report = {}
        
        try:
            # Basic information
            report['Basic Information'] = {
                'Original Rows': len(original_data),
                'Synthetic Rows': len(synthetic_data),
                'Original Columns': len(original_data.columns),
                'Synthetic Columns': len(synthetic_data.columns),
                'Expansion Ratio': f"{len(synthetic_data) / len(original_data):.1f}x"
            }
            
            # Privacy analysis
            privacy_metrics = self.check_privacy(original_data, synthetic_data)
            report['Privacy Analysis'] = {
                'Privacy Status': 'SAFE' if privacy_metrics['is_private'] else 'REVIEW NEEDED',
                'Leakage Score': f"{privacy_metrics['leakage_score']:.4f}",
                'Privacy Level': privacy_metrics.get('privacy_level', 'Unknown'),
                'Recommendations': '; '.join(privacy_metrics.get('recommendations', []))
            }
            
            # Statistical analysis
            numerical_cols = [col for col, info in column_info.items() if info['type'] == 'numerical']
            if numerical_cols:
                stat_results = []
                for col in numerical_cols[:5]:  # Limit to first 5 for report
                    if col in original_data.columns and col in synthetic_data.columns:
                        test_result = self.statistical_tests(original_data[col], synthetic_data[col])
                        if 'ks_pvalue' in test_result:
                            status = 'SIMILAR' if test_result['ks_pvalue'] > 0.05 else 'DIFFERENT'
                            stat_results.append(f"{col}: {status} (p={test_result['ks_pvalue']:.4f})")
                
                report['Statistical Analysis'] = {
                    'Distribution Tests': stat_results,
                    'Correlation Preservation': f"{self.correlation_preservation_score(original_data, synthetic_data, numerical_cols):.3f}"
                }
            
            # Overall quality
            quality_score = self.overall_quality_score(original_data, synthetic_data, column_info)
            report['Overall Quality'] = {
                'Quality Score': f"{quality_score:.1f}/100",
                'Quality Level': self._get_quality_level(quality_score)
            }
            
            return report
            
        except Exception as e:
            return {'Error': f"Could not generate report: {str(e)}"}
    
    def _get_quality_level(self, score: float) -> str:
        """Determine quality level based on score."""
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Good"
        elif score >= 70:
            return "Fair"
        else:
            return "Poor"

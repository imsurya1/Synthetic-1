import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any

class DataVisualizer:
    """
    Creates visualizations for comparing original and synthetic data.
    """
    
    def __init__(self):
        self.color_palette = {
            'original': '#1f77b4',
            'synthetic': '#ff7f0e',
            'comparison': ['#1f77b4', '#ff7f0e']
        }
    
    def plot_distribution_comparison(self, original_series: pd.Series, 
                                   synthetic_series: pd.Series, 
                                   column_name: str) -> go.Figure:
        """
        Create overlaid histograms comparing original and synthetic data distributions.
        
        Args:
            original_series: Original data series
            synthetic_series: Synthetic data series
            column_name: Name of the column being compared
            
        Returns:
            Plotly figure object
        """
        try:
            # Clean data
            orig_clean = original_series.dropna()
            synth_clean = synthetic_series.dropna()
            
            if len(orig_clean) == 0 or len(synth_clean) == 0:
                return self._create_empty_figure("Insufficient data for visualization")
            
            # Create figure with secondary y-axis
            fig = go.Figure()
            
            # Determine bin edges based on combined data range
            combined_data = np.concatenate([orig_clean, synth_clean])
            bins = np.histogram_bin_edges(combined_data, bins='auto')
            
            # Add original data histogram
            fig.add_trace(go.Histogram(
                x=orig_clean,
                name='Original',
                opacity=0.7,
                nbinsx=len(bins)-1,
                histnorm='probability density',
                marker_color=self.color_palette['original']
            ))
            
            # Add synthetic data histogram
            fig.add_trace(go.Histogram(
                x=synth_clean,
                name='Synthetic',
                opacity=0.7,
                nbinsx=len(bins)-1,
                histnorm='probability density',
                marker_color=self.color_palette['synthetic']
            ))
            
            # Update layout
            fig.update_layout(
                title=f'Distribution Comparison: {column_name}',
                xaxis_title=column_name,
                yaxis_title='Probability Density',
                barmode='overlay',
                legend=dict(x=0.7, y=0.9),
                height=400,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_figure(f"Error creating distribution plot: {str(e)}")
    
    def plot_correlation_comparison(self, original_data: pd.DataFrame, 
                                  synthetic_data: pd.DataFrame, 
                                  numerical_cols: List[str]) -> go.Figure:
        """
        Create side-by-side correlation heatmaps for original and synthetic data.
        
        Args:
            original_data: Original dataset
            synthetic_data: Synthetic dataset
            numerical_cols: List of numerical columns to include
            
        Returns:
            Plotly figure object
        """
        try:
            if len(numerical_cols) < 2:
                return self._create_empty_figure("Need at least 2 numerical columns for correlation analysis")
            
            # Calculate correlation matrices
            orig_corr = original_data[numerical_cols].corr()
            synth_corr = synthetic_data[numerical_cols].corr()
            
            # Create subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Original Data Correlations', 'Synthetic Data Correlations'),
                horizontal_spacing=0.1
            )
            
            # Add original correlation heatmap
            fig.add_trace(
                go.Heatmap(
                    z=orig_corr.values,
                    x=orig_corr.columns,
                    y=orig_corr.columns,
                    colorscale='RdBu',
                    zmid=0,
                    zmin=-1,
                    zmax=1,
                    showscale=False,
                    text=np.round(orig_corr.values, 2),
                    texttemplate="%{text}",
                    textfont={"size": 10}
                ),
                row=1, col=1
            )
            
            # Add synthetic correlation heatmap
            fig.add_trace(
                go.Heatmap(
                    z=synth_corr.values,
                    x=synth_corr.columns,
                    y=synth_corr.columns,
                    colorscale='RdBu',
                    zmid=0,
                    zmin=-1,
                    zmax=1,
                    showscale=True,
                    text=np.round(synth_corr.values, 2),
                    texttemplate="%{text}",
                    textfont={"size": 10}
                ),
                row=1, col=2
            )
            
            # Update layout
            fig.update_layout(
                title='Correlation Matrix Comparison',
                height=500,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_figure(f"Error creating correlation plot: {str(e)}")
    
    def plot_categorical_comparison(self, original_series: pd.Series, 
                                  synthetic_series: pd.Series, 
                                  column_name: str) -> go.Figure:
        """
        Create side-by-side bar charts comparing categorical distributions.
        
        Args:
            original_series: Original categorical data
            synthetic_series: Synthetic categorical data
            column_name: Name of the column
            
        Returns:
            Plotly figure object
        """
        try:
            # Get value counts
            orig_counts = original_series.value_counts().sort_index()
            synth_counts = synthetic_series.value_counts().sort_index()
            
            # Normalize to percentages
            orig_pct = (orig_counts / orig_counts.sum() * 100).round(1)
            synth_pct = (synth_counts / synth_counts.sum() * 100).round(1)
            
            # Get all unique categories
            all_categories = sorted(set(orig_counts.index) | set(synth_counts.index))
            
            # Create data for plotting
            orig_values = [orig_pct.get(cat, 0) for cat in all_categories]
            synth_values = [synth_pct.get(cat, 0) for cat in all_categories]
            
            # Create figure
            fig = go.Figure()
            
            # Add bars for original data
            fig.add_trace(go.Bar(
                name='Original',
                x=all_categories,
                y=orig_values,
                marker_color=self.color_palette['original'],
                opacity=0.8
            ))
            
            # Add bars for synthetic data
            fig.add_trace(go.Bar(
                name='Synthetic',
                x=all_categories,
                y=synth_values,
                marker_color=self.color_palette['synthetic'],
                opacity=0.8
            ))
            
            # Update layout
            fig.update_layout(
                title=f'Categorical Distribution Comparison: {column_name}',
                xaxis_title=column_name,
                yaxis_title='Percentage (%)',
                barmode='group',
                height=400,
                legend=dict(x=0.7, y=0.9)
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_figure(f"Error creating categorical plot: {str(e)}")
    
    def plot_time_series_comparison(self, original_series: pd.Series, 
                                  synthetic_series: pd.Series, 
                                  column_name: str) -> go.Figure:
        """
        Create time series plots comparing datetime distributions.
        
        Args:
            original_series: Original datetime data
            synthetic_series: Synthetic datetime data
            column_name: Name of the column
            
        Returns:
            Plotly figure object
        """
        try:
            # Convert to datetime
            orig_dates = pd.to_datetime(original_series).dropna()
            synth_dates = pd.to_datetime(synthetic_series).dropna()
            
            if len(orig_dates) == 0 or len(synth_dates) == 0:
                return self._create_empty_figure("Insufficient datetime data for visualization")
            
            # Create date frequency analysis
            orig_freq = orig_dates.dt.date.value_counts().sort_index()
            synth_freq = synth_dates.dt.date.value_counts().sort_index()
            
            # Create figure
            fig = go.Figure()
            
            # Add original timeline
            fig.add_trace(go.Scatter(
                x=orig_freq.index,
                y=orig_freq.values,
                mode='lines+markers',
                name='Original',
                line=dict(color=self.color_palette['original']),
                marker=dict(size=4)
            ))
            
            # Add synthetic timeline
            fig.add_trace(go.Scatter(
                x=synth_freq.index,
                y=synth_freq.values,
                mode='lines+markers',
                name='Synthetic',
                line=dict(color=self.color_palette['synthetic']),
                marker=dict(size=4)
            ))
            
            # Update layout
            fig.update_layout(
                title=f'Temporal Distribution Comparison: {column_name}',
                xaxis_title='Date',
                yaxis_title='Frequency',
                height=400,
                legend=dict(x=0.7, y=0.9)
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_figure(f"Error creating time series plot: {str(e)}")
    
    def plot_summary_statistics(self, original_data: pd.DataFrame, 
                              synthetic_data: pd.DataFrame, 
                              numerical_cols: List[str]) -> go.Figure:
        """
        Create a comparison of summary statistics between original and synthetic data.
        
        Args:
            original_data: Original dataset
            synthetic_data: Synthetic dataset
            numerical_cols: List of numerical columns
            
        Returns:
            Plotly figure object
        """
        try:
            if not numerical_cols:
                return self._create_empty_figure("No numerical columns available for summary statistics")
            
            # Calculate summary statistics
            stats_comparison = []
            
            for col in numerical_cols:
                if col in original_data.columns and col in synthetic_data.columns:
                    orig_col = original_data[col].dropna()
                    synth_col = synthetic_data[col].dropna()
                    
                    if len(orig_col) > 0 and len(synth_col) > 0:
                        stats_comparison.append({
                            'Column': col,
                            'Original_Mean': orig_col.mean(),
                            'Synthetic_Mean': synth_col.mean(),
                            'Original_Std': orig_col.std(),
                            'Synthetic_Std': synth_col.std(),
                            'Original_Min': orig_col.min(),
                            'Synthetic_Min': synth_col.min(),
                            'Original_Max': orig_col.max(),
                            'Synthetic_Max': synth_col.max()
                        })
            
            if not stats_comparison:
                return self._create_empty_figure("No valid numerical data for comparison")
            
            # Create comparison chart for means and standard deviations
            df_stats = pd.DataFrame(stats_comparison)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Mean Comparison', 'Standard Deviation Comparison', 
                              'Minimum Values', 'Maximum Values'),
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # Mean comparison
            fig.add_trace(go.Bar(
                name='Original Mean',
                x=df_stats['Column'],
                y=df_stats['Original_Mean'],
                marker_color=self.color_palette['original'],
                showlegend=True
            ), row=1, col=1)
            
            fig.add_trace(go.Bar(
                name='Synthetic Mean',
                x=df_stats['Column'],
                y=df_stats['Synthetic_Mean'],
                marker_color=self.color_palette['synthetic'],
                showlegend=True
            ), row=1, col=1)
            
            # Standard deviation comparison
            fig.add_trace(go.Bar(
                name='Original Std',
                x=df_stats['Column'],
                y=df_stats['Original_Std'],
                marker_color=self.color_palette['original'],
                showlegend=False
            ), row=1, col=2)
            
            fig.add_trace(go.Bar(
                name='Synthetic Std',
                x=df_stats['Column'],
                y=df_stats['Synthetic_Std'],
                marker_color=self.color_palette['synthetic'],
                showlegend=False
            ), row=1, col=2)
            
            # Min values
            fig.add_trace(go.Bar(
                name='Original Min',
                x=df_stats['Column'],
                y=df_stats['Original_Min'],
                marker_color=self.color_palette['original'],
                showlegend=False
            ), row=2, col=1)
            
            fig.add_trace(go.Bar(
                name='Synthetic Min',
                x=df_stats['Column'],
                y=df_stats['Synthetic_Min'],
                marker_color=self.color_palette['synthetic'],
                showlegend=False
            ), row=2, col=1)
            
            # Max values
            fig.add_trace(go.Bar(
                name='Original Max',
                x=df_stats['Column'],
                y=df_stats['Original_Max'],
                marker_color=self.color_palette['original'],
                showlegend=False
            ), row=2, col=2)
            
            fig.add_trace(go.Bar(
                name='Synthetic Max',
                x=df_stats['Column'],
                y=df_stats['Synthetic_Max'],
                marker_color=self.color_palette['synthetic'],
                showlegend=False
            ), row=2, col=2)
            
            # Update layout
            fig.update_layout(
                title='Summary Statistics Comparison',
                height=600,
                barmode='group'
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_figure(f"Error creating summary statistics plot: {str(e)}")
    
    def _create_empty_figure(self, message: str) -> go.Figure:
        """Create an empty figure with an error message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            height=400
        )
        return fig

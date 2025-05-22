import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_processor import DataProcessor
from src.synthesizer import DataSynthesizer
from src.validator import DataValidator
from src.visualizer import DataVisualizer
from utils.helpers import download_button_with_data

# Page configuration
st.set_page_config(
    page_title="Synthetic Financial Data Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'synthetic_data' not in st.session_state:
    st.session_state.synthetic_data = None
if 'synthesis_complete' not in st.session_state:
    st.session_state.synthesis_complete = False
if 'column_info' not in st.session_state:
    st.session_state.column_info = None

def main():
    st.title("🔄 Synthetic Financial Data Generator")
    st.markdown("""
    Generate high-quality synthetic financial data that preserves statistical distributions 
    and relationships from your Excel input while maintaining complete privacy.
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Excel File",
            type=['xlsx', 'xls'],
            help="Upload your Excel file containing sensitive financial data (10-15 rows minimum)"
        )
        
        if uploaded_file is not None:
            try:
                # Process uploaded file
                processor = DataProcessor()
                original_data, column_info = processor.load_and_analyze(uploaded_file)
                
                st.session_state.original_data = original_data
                st.session_state.column_info = column_info
                st.session_state.synthesis_complete = False
                
                st.success(f"✅ File loaded successfully!")
                st.info(f"**Rows:** {len(original_data)}")
                st.info(f"**Columns:** {len(original_data.columns)}")
                
                # Display column information
                st.subheader("Detected Column Types")
                for col, info in column_info.items():
                    icon = "🔢" if info['type'] == 'numerical' else "📅" if info['type'] == 'datetime' else "📝"
                    st.text(f"{icon} {col}: {info['type']}")
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return

    # Main content area
    if st.session_state.original_data is not None:
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Overview", "⚙️ Synthesis Configuration", "📊 Results", "📈 Quality Validation"])
        
        with tab1:
            show_data_overview()
        
        with tab2:
            show_synthesis_configuration()
        
        with tab3:
            show_results()
        
        with tab4:
            show_quality_validation()
    else:
        st.info("👆 Please upload an Excel file to get started.")
        
        # Show example of what the tool can do
        with st.expander("ℹ️ What does this tool do?"):
            st.markdown("""
            This tool generates synthetic financial data that:
            
            **🔒 Privacy Preserving**
            - Processes data locally - no external services
            - Doesn't expose original records
            - Creates statistically similar but entirely new data
            
            **📈 Relationship Preserving**
            - Maintains correlations between columns
            - Preserves statistical distributions
            - Handles complex financial relationships
            
            **🎯 Customizable**
            - Specify unique constraint columns at runtime
            - Configure output size (500-5,000 rows)
            - Support for any Excel schema
            
            **✅ Quality Assured**
            - Statistical validation dashboards
            - Distribution comparison charts
            - Privacy verification metrics
            """)

def show_data_overview():
    st.header("📋 Original Data Overview")
    
    original_data = st.session_state.original_data
    column_info = st.session_state.column_info
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Data Preview")
        st.dataframe(original_data.head(10), use_container_width=True)
    
    with col2:
        st.subheader("Statistics")
        st.metric("Total Rows", len(original_data))
        st.metric("Total Columns", len(original_data.columns))
        
        # Column type breakdown
        type_counts = {}
        for col_info in column_info.values():
            col_type = col_info['type']
            type_counts[col_type] = type_counts.get(col_type, 0) + 1
        
        for col_type, count in type_counts.items():
            st.metric(f"{col_type.title()} Columns", count)
    
    # Show detailed column analysis
    st.subheader("Column Analysis")
    
    for col, info in column_info.items():
        with st.expander(f"📊 {col} ({info['type']})"):
            if info['type'] == 'numerical':
                col_data = original_data[col].dropna()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean", f"{col_data.mean():.2f}")
                with col2:
                    st.metric("Std Dev", f"{col_data.std():.2f}")
                with col3:
                    st.metric("Range", f"{col_data.min():.1f} - {col_data.max():.1f}")
                
                # Mini histogram
                fig = px.histogram(x=col_data, nbins=10, title=f"Distribution of {col}")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
            elif info['type'] == 'categorical':
                value_counts = original_data[col].value_counts()
                st.write(f"**Unique values:** {len(value_counts)}")
                if len(value_counts) <= 10:
                    st.write("**Value distribution:**")
                    st.bar_chart(value_counts)
                else:
                    st.write("**Top 10 values:**")
                    st.bar_chart(value_counts.head(10))
            
            elif info['type'] == 'datetime':
                col_data = pd.to_datetime(original_data[col]).dropna()
                st.write(f"**Date range:** {col_data.min().strftime('%Y-%m-%d')} to {col_data.max().strftime('%Y-%m-%d')}")
                st.write(f"**Total span:** {(col_data.max() - col_data.min()).days} days")

def show_synthesis_configuration():
    st.header("⚙️ Synthesis Configuration")
    
    original_data = st.session_state.original_data
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Generation Parameters")
        
        # Number of synthetic rows
        num_rows = st.slider(
            "Number of synthetic rows to generate",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="Specify how many synthetic rows to generate"
        )
        
        # Synthesis method
        synthesis_method = st.selectbox(
            "Synthesis Method",
            ["GaussianCopula", "CTGAN", "CopulaGAN"],
            help="Choose the synthesis algorithm. GaussianCopula is faster, CTGAN/CopulaGAN are more accurate for complex relationships"
        )
        
        # Privacy level
        privacy_level = st.selectbox(
            "Privacy Level",
            ["Standard", "High", "Maximum"],
            help="Higher privacy levels add more noise but reduce similarity to original data"
        )
    
    with col2:
        st.subheader("Unique Constraints")
        
        # Select columns that must be unique
        available_columns = list(original_data.columns)
        unique_columns = st.multiselect(
            "Select columns that must have unique values",
            available_columns,
            help="These columns will be enforced to have unique values in the synthetic data"
        )
        
        if unique_columns:
            st.info(f"✅ Selected {len(unique_columns)} column(s) for uniqueness constraints")
            for col in unique_columns:
                st.text(f"• {col}")
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        # Correlation preservation
        preserve_correlations = st.checkbox(
            "Preserve inter-column correlations",
            value=True,
            help="Maintain statistical relationships between columns"
        )
        
        # Constraint handling
        constraint_handling = st.selectbox(
            "Constraint Handling Strategy",
            ["Reject Sampling", "Transform", "Post-process"],
            help="How to handle constraint violations during generation"
        )
        
        # Random seed for reproducibility
        use_seed = st.checkbox("Use random seed for reproducibility")
        random_seed = None
        if use_seed:
            random_seed = st.number_input("Random Seed", value=42, min_value=0)
    
    # Generate button
    if st.button("🚀 Generate Synthetic Data", type="primary", use_container_width=True):
        if len(original_data) < 5:
            st.error("❌ Need at least 5 rows of original data for synthesis")
            return
        
        generate_synthetic_data(
            num_rows=num_rows,
            synthesis_method=synthesis_method,
            privacy_level=privacy_level,
            unique_columns=unique_columns,
            preserve_correlations=preserve_correlations,
            constraint_handling=constraint_handling,
            random_seed=random_seed
        )

def generate_synthetic_data(num_rows, synthesis_method, privacy_level, unique_columns, 
                          preserve_correlations, constraint_handling, random_seed):
    """Generate synthetic data with the specified parameters and constraints"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize synthesizer with advanced settings
        status_text.text("🔧 Initializing synthesizer...")
        progress_bar.progress(10)
        
        synthesizer = DataSynthesizer(
            method=synthesis_method,
            privacy_level=privacy_level,
            random_seed=random_seed
        )
        
        # Detect columns that must be non-negative
        numeric_cols = st.session_state.column_info.keys()
        unique_columns = [col for col in numeric_cols if col in (unique_columns or [])]
        
        # Fit the model
        status_text.text("🧠 Training synthesis model...")
        progress_bar.progress(30)
        
        synthesizer.fit(
            st.session_state.original_data,
            st.session_state.column_info,
            unique_columns=unique_columns,
            preserve_correlations=preserve_correlations
        )
        
        # Generate synthetic data
        status_text.text("🎲 Generating synthetic data...")
        progress_bar.progress(70)
        
        synthetic_data = synthesizer.generate(
            num_rows=num_rows,
            constraint_handling=constraint_handling
        )
        
        # Store results
        st.session_state.synthetic_data = synthetic_data
        st.session_state.synthesis_complete = True
        
        status_text.text("✅ Synthesis complete!")
        progress_bar.progress(100)
        
        st.success(f"🎉 Successfully generated {len(synthetic_data)} synthetic rows!")
        st.success(f"📊 Data shape: {synthetic_data.shape}")
        st.success("✅ Results saved! Check the 'Synthesis Results' tab.")
        
        # Show a preview of generated data
        st.subheader("Preview of Generated Data:")
        st.dataframe(synthetic_data.head(3))
        
        # Don't auto-switch tabs to avoid confusion
        st.info("👆 Go to the 'Synthesis Results' tab to see full results and download options.")
        
    except Exception as e:
        st.error(f"❌ Error during synthesis: {str(e)}")
        st.error(f"Error details: {type(e).__name__}")
        
        # Show more helpful debugging info
        if "column_info" in str(e):
            st.error("Issue with column analysis. Please check your Excel file format.")
        elif "synthesizer" in str(e):
            st.error("Issue with data synthesis. Trying fallback method...")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Try to show what data was loaded
        if st.session_state.original_data is not None:
            st.write("**Original data shape:**", st.session_state.original_data.shape)
            st.write("**Available columns:**", list(st.session_state.original_data.columns))

def show_results():
    st.header("📊 Synthesis Results")
    
    # Debug info to help troubleshoot
    st.write("**Debug Info:**")
    st.write(f"Synthesis complete: {st.session_state.get('synthesis_complete', False)}")
    st.write(f"Synthetic data exists: {st.session_state.get('synthetic_data') is not None}")
    
    # If synthetic data exists but flag isn't set, fix it
    if st.session_state.get('synthetic_data') is not None and not st.session_state.get('synthesis_complete', False):
        st.session_state.synthesis_complete = True
        st.success("✅ Found generated data! Displaying results now.")
    
    if st.session_state.get('synthetic_data') is None:
        st.info("🔄 Generate synthetic data first in the 'Synthesis Configuration' tab")
        return
    
    synthetic_data = st.session_state.synthetic_data
    original_data = st.session_state.original_data
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Original Rows", len(original_data))
    with col2:
        st.metric("Synthetic Rows", len(synthetic_data))
    with col3:
        expansion_ratio = len(synthetic_data) / len(original_data)
        st.metric("Expansion Ratio", f"{expansion_ratio:.1f}x")
    with col4:
        st.metric("Columns Preserved", len(synthetic_data.columns))
    
    # Data preview
    st.subheader("📋 Synthetic Data Preview")
    st.dataframe(synthetic_data.head(20), use_container_width=True)
    
    # Download options
    st.subheader("💾 Download Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Excel download
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            synthetic_data.to_excel(writer, sheet_name='Synthetic_Data', index=False)
            original_data.to_excel(writer, sheet_name='Original_Data', index=False)
        
        st.download_button(
            label="📥 Download as Excel",
            data=excel_buffer.getvalue(),
            file_name=f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col2:
        # CSV download
        csv_buffer = io.StringIO()
        synthetic_data.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="📥 Download as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col3:
        # Generate report
        if st.button("📊 Generate Quality Report"):
            generate_quality_report()

def generate_quality_report():
    """Generate a comprehensive quality report"""
    validator = DataValidator()
    
    report = validator.generate_report(
        st.session_state.original_data,
        st.session_state.synthetic_data,
        st.session_state.column_info
    )
    
    # Create downloadable report
    report_buffer = io.StringIO()
    report_buffer.write("SYNTHETIC DATA QUALITY REPORT\n")
    report_buffer.write("=" * 50 + "\n\n")
    
    for section, content in report.items():
        report_buffer.write(f"{section.upper()}\n")
        report_buffer.write("-" * len(section) + "\n")
        if isinstance(content, dict):
            for key, value in content.items():
                report_buffer.write(f"{key}: {value}\n")
        else:
            report_buffer.write(f"{content}\n")
        report_buffer.write("\n")
    
    st.download_button(
        label="📋 Download Quality Report",
        data=report_buffer.getvalue(),
        file_name=f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

def show_quality_validation():
    st.header("📈 Quality Validation")
    
    if not st.session_state.synthesis_complete or st.session_state.synthetic_data is None:
        st.info("🔄 Generate synthetic data first to see quality validation")
        return
    
    original_data = st.session_state.original_data
    synthetic_data = st.session_state.synthetic_data
    column_info = st.session_state.column_info
    
    # Initialize validator and visualizer
    validator = DataValidator()
    visualizer = DataVisualizer()
    
    # Privacy validation
    st.subheader("🔒 Privacy Validation")
    privacy_metrics = validator.check_privacy(original_data, synthetic_data)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Data Leakage Score", f"{privacy_metrics['leakage_score']:.3f}")
    with col2:
        st.metric("Similarity Threshold", f"{privacy_metrics['similarity_threshold']:.3f}")
    with col3:
        privacy_status = "✅ SAFE" if privacy_metrics['is_private'] else "⚠️ REVIEW"
        st.metric("Privacy Status", privacy_status)
    
    if not privacy_metrics['is_private']:
        st.warning("⚠️ Some synthetic records may be too similar to original data. Consider increasing privacy level.")
    
    # Statistical validation
    st.subheader("📊 Statistical Validation")
    
    # Distribution comparisons for numerical columns
    numerical_cols = [col for col, info in column_info.items() if info['type'] == 'numerical']
    
    if numerical_cols:
        selected_num_col = st.selectbox("Select numerical column for distribution comparison", numerical_cols)
        
        if selected_num_col:
            fig = visualizer.plot_distribution_comparison(
                original_data[selected_num_col],
                synthetic_data[selected_num_col],
                selected_num_col
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical tests
            stats_results = validator.statistical_tests(
                original_data[selected_num_col],
                synthetic_data[selected_num_col]
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("KS Test p-value", f"{stats_results['ks_pvalue']:.4f}")
                ks_status = "✅ SIMILAR" if stats_results['ks_pvalue'] > 0.05 else "⚠️ DIFFERENT"
                st.caption(f"Distribution similarity: {ks_status}")
            
            with col2:
                st.metric("Mean Difference", f"{stats_results['mean_diff']:.4f}")
                st.metric("Std Difference", f"{stats_results['std_diff']:.4f}")
    
    # Correlation analysis
    st.subheader("🔗 Correlation Analysis")
    
    if len(numerical_cols) >= 2:
        fig = visualizer.plot_correlation_comparison(original_data, synthetic_data, numerical_cols)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation preservation score
        corr_score = validator.correlation_preservation_score(original_data, synthetic_data, numerical_cols)
        st.metric("Correlation Preservation Score", f"{corr_score:.3f}")
        
        if corr_score > 0.8:
            st.success("✅ Excellent correlation preservation")
        elif corr_score > 0.6:
            st.info("ℹ️ Good correlation preservation")
        else:
            st.warning("⚠️ Correlation preservation could be improved")
    
    # Categorical column analysis
    categorical_cols = [col for col, info in column_info.items() if info['type'] == 'categorical']
    
    if categorical_cols:
        st.subheader("📝 Categorical Analysis")
        selected_cat_col = st.selectbox("Select categorical column for analysis", categorical_cols)
        
        if selected_cat_col:
            fig = visualizer.plot_categorical_comparison(
                original_data[selected_cat_col],
                synthetic_data[selected_cat_col],
                selected_cat_col
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Overall quality score
    st.subheader("🎯 Overall Quality Score")
    quality_score = validator.overall_quality_score(original_data, synthetic_data, column_info)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Quality gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = quality_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Overall Quality Score"},
            delta = {'reference': 80},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Quality interpretation
    if quality_score >= 90:
        st.success("🌟 Excellent quality! Synthetic data is highly representative of the original.")
    elif quality_score >= 80:
        st.success("✅ Good quality! Synthetic data preserves most important characteristics.")
    elif quality_score >= 70:
        st.info("ℹ️ Fair quality. Consider adjusting synthesis parameters for better results.")
    else:
        st.warning("⚠️ Quality could be improved. Try different synthesis methods or parameters.")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import io
from typing import Any, Dict, List
import base64

def download_button_with_data(data: Any, filename: str, label: str, mime_type: str = "application/octet-stream"):
    """
    Create a download button for data with proper formatting.
    
    Args:
        data: Data to download
        filename: Filename for download
        label: Button label
        mime_type: MIME type for the file
    """
    try:
        if isinstance(data, pd.DataFrame):
            # Convert DataFrame to appropriate format based on file extension
            if filename.endswith('.csv'):
                buffer = io.StringIO()
                data.to_csv(buffer, index=False)
                file_data = buffer.getvalue()
                mime_type = "text/csv"
            elif filename.endswith('.xlsx'):
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    data.to_excel(writer, index=False)
                file_data = buffer.getvalue()
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            else:
                file_data = str(data)
        else:
            file_data = data
        
        st.download_button(
            label=label,
            data=file_data,
            file_name=filename,
            mime=mime_type
        )
    except Exception as e:
        st.error(f"Error creating download button: {str(e)}")

def format_number(number: float, decimal_places: int = 2) -> str:
    """
    Format a number for display with appropriate decimal places.
    
    Args:
        number: Number to format
        decimal_places: Number of decimal places
        
    Returns:
        Formatted number string
    """
    try:
        if pd.isna(number):
            return "N/A"
        
        if abs(number) >= 1000000:
            return f"{number/1000000:.{decimal_places}f}M"
        elif abs(number) >= 1000:
            return f"{number/1000:.{decimal_places}f}K"
        else:
            return f"{number:.{decimal_places}f}"
    except:
        return str(number)

def validate_excel_file(file) -> Dict[str, Any]:
    """
    Validate uploaded Excel file and return validation results.
    
    Args:
        file: Uploaded file object
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': False,
        'error_message': None,
        'warnings': [],
        'info': {}
    }
    
    try:
        # Check file type
        if not file.name.endswith(('.xlsx', '.xls')):
            validation_result['error_message'] = "File must be an Excel file (.xlsx or .xls)"
            return validation_result
        
        # Try to read the file
        df = pd.read_excel(file)
        
        # Check if file is empty
        if df.empty:
            validation_result['error_message'] = "Excel file is empty"
            return validation_result
        
        # Check minimum rows
        if len(df) < 3:
            validation_result['error_message'] = "Need at least 3 rows of data for synthesis"
            return validation_result
        
        # Check for at least one non-null column
        if df.isnull().all().all():
            validation_result['error_message'] = "All cells are empty"
            return validation_result
        
        # Warnings for data quality
        null_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if null_percentage > 50:
            validation_result['warnings'].append(f"High percentage of missing data ({null_percentage:.1f}%)")
        
        if len(df) < 10:
            validation_result['warnings'].append("Small dataset may produce less realistic synthetic data")
        
        # File info
        validation_result['info'] = {
            'rows': len(df),
            'columns': len(df.columns),
            'file_size_mb': file.size / (1024*1024) if hasattr(file, 'size') else 0,
            'null_percentage': null_percentage
        }
        
        validation_result['is_valid'] = True
        
    except Exception as e:
        validation_result['error_message'] = f"Error reading Excel file: {str(e)}"
    
    return validation_result

def get_column_type_icon(column_type: str) -> str:
    """
    Get appropriate icon for column type.
    
    Args:
        column_type: Type of the column
        
    Returns:
        Icon string
    """
    icons = {
        'numerical': '🔢',
        'categorical': '📝',
        'datetime': '📅',
        'boolean': '✅',
        'text': '📄'
    }
    return icons.get(column_type, '❓')

def create_progress_bar_with_steps(steps: List[str], current_step: int = 0) -> None:
    """
    Create a progress bar with step indicators.
    
    Args:
        steps: List of step names
        current_step: Current step index (0-based)
    """
    try:
        total_steps = len(steps)
        progress_percentage = (current_step + 1) / total_steps
        
        # Create progress bar
        progress_bar = st.progress(progress_percentage)
        
        # Create step indicators
        cols = st.columns(total_steps)
        for i, (col, step) in enumerate(zip(cols, steps)):
            with col:
                if i < current_step:
                    st.success(f"✅ {step}")
                elif i == current_step:
                    st.info(f"🔄 {step}")
                else:
                    st.write(f"⏳ {step}")
        
        return progress_bar
        
    except Exception as e:
        st.error(f"Error creating progress bar: {str(e)}")
        return None

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default value if division is not possible.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value to return if division fails
        
    Returns:
        Division result or default value
    """
    try:
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return default
        return numerator / denominator
    except:
        return default

def truncate_text(text: str, max_length: int = 50) -> str:
    """
    Truncate text to specified length with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    try:
        text_str = str(text)
        if len(text_str) <= max_length:
            return text_str
        return text_str[:max_length-3] + "..."
    except:
        return str(text)

def calculate_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate memory usage statistics for a DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with memory usage information
    """
    try:
        memory_info = {
            'total_mb': df.memory_usage(deep=True).sum() / (1024*1024),
            'per_column_mb': {},
            'largest_columns': []
        }
        
        # Calculate per-column memory usage
        for col in df.columns:
            col_memory = df[col].memory_usage(deep=True) / (1024*1024)
            memory_info['per_column_mb'][col] = col_memory
        
        # Find largest columns
        sorted_cols = sorted(memory_info['per_column_mb'].items(), 
                           key=lambda x: x[1], reverse=True)
        memory_info['largest_columns'] = sorted_cols[:5]
        
        return memory_info
        
    except Exception as e:
        return {'error': str(e)}

def generate_sample_data_info() -> Dict[str, Any]:
    """
    Generate information about what constitutes good sample data.
    
    Returns:
        Dictionary with sample data guidelines
    """
    return {
        'minimum_requirements': {
            'rows': 'At least 10-15 rows',
            'columns': 'At least 3-5 columns',
            'data_types': 'Mix of numerical, categorical, and date columns'
        },
        'quality_tips': [
            'Include representative data that covers your typical use cases',
            'Ensure columns have meaningful relationships (e.g., transaction amount affects balance)',
            'Include various data types: numbers, categories, dates, text',
            'Minimize missing values where possible',
            'Include edge cases and outliers if they exist in your real data'
        ],
        'privacy_recommendations': [
            'Remove or mask any personally identifiable information (PII)',
            'Use representative but not actual sensitive values',
            'Ensure the sample covers the statistical patterns you want to preserve',
            'Consider using data from a test environment if available'
        ],
        'examples': {
            'financial_transactions': [
                'TransactionID', 'AccountNumber', 'TransactionType', 
                'Amount', 'Balance', 'Date', 'MerchantCategory'
            ],
            'customer_data': [
                'CustomerID', 'Age', 'Income', 'City', 'ProductCategory', 
                'PurchaseAmount', 'SignupDate'
            ],
            'sales_data': [
                'OrderID', 'ProductID', 'Quantity', 'UnitPrice', 
                'Region', 'SalesRep', 'OrderDate'
            ]
        }
    }

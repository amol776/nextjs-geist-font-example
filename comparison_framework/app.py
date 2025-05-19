import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import os
from datetime import datetime

# Import our modules
from data_reader import DataReader
from mapping_manager import MappingManager
from report_generator import ReportGenerator
from db_connector import DatabaseConnector
from api_fetcher import APIFetcher
from utils import log_error, format_timestamp

# Set page config
st.set_page_config(
    page_title="Data Comparison Framework",
    page_icon="📊",
    layout="wide"
)

# Define source types
SOURCE_TYPES = [
    "CSV File",
    "DAT File",
    "SQL Server",
    "Stored Procedure",
    "Teradata",
    "API",
    "Parquet File",
    "Zipped Flat Files"
]

def main():
    """Main application function"""
    
    # Add header with modern styling
    st.markdown(
        """
        <div style='text-align: center; background-color: #f0f2f6; padding: 1rem; border-radius: 10px;'>
            <h1 style='color: #1f77b4;'>Data Comparison Framework</h1>
            <p style='color: #666;'>Compare data from multiple sources with advanced mapping and reporting capabilities</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Initialize session state
    if 'source_df' not in st.session_state:
        st.session_state.source_df = None
    if 'target_df' not in st.session_state:
        st.session_state.target_df = None
    if 'column_mapping' not in st.session_state:
        st.session_state.column_mapping = {}
    if 'excluded_columns' not in st.session_state:
        st.session_state.excluded_columns = []

    # Create two columns for source and target selection
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Source Configuration")
        source_type = st.selectbox("Select Source Type", SOURCE_TYPES, key="source_type")
        source_data = handle_data_source(source_type, "source")

    with col2:
        st.subheader("Target Configuration")
        target_type = st.selectbox("Select Target Type", SOURCE_TYPES, key="target_type")
        target_data = handle_data_source(target_type, "target")

    # If both source and target data are loaded
    if source_data is not None and target_data is not None:
        st.session_state.source_df = source_data
        st.session_state.target_df = target_data
        
        # Show column mapping interface
        st.subheader("Column Mapping")
        show_column_mapping_interface(source_data, target_data)

        # Add join key selection
        st.subheader("Join Key Selection")
        common_columns = list(set(source_data.columns) & set(target_data.columns))
        if not common_columns:
            st.warning("No common columns found between source and target data")
            join_keys = None
        else:
            # Show common columns in a more readable format
            st.write("Common columns available for joining:")
            for col in common_columns:
                st.write(f"- {col}")
            
            selected_keys = st.multiselect(
                "Select Join Key(s)",
                options=common_columns,
                help="Select one or more columns to use as join keys for comparison. These columns should uniquely identify records."
            )
            
            # Validate selected join keys
            if selected_keys:
                # Check if selected keys exist in both dataframes
                invalid_keys = [key for key in selected_keys 
                              if key not in source_data.columns or key not in target_data.columns]
                if invalid_keys:
                    st.error(f"Invalid join keys selected: {', '.join(invalid_keys)}")
                    join_keys = None
                else:
                    join_keys = selected_keys
                    # Show preview of join keys
                    st.write("Preview of selected join keys:")
                    preview_df = pd.DataFrame({
                        'Join Key': selected_keys,
                        'Source Values (sample)': [str(source_data[key].head(3).tolist()) for key in selected_keys],
                        'Target Values (sample)': [str(target_data[key].head(3).tolist()) for key in selected_keys]
                    })
                    st.dataframe(preview_df)
            else:
                st.warning("No join keys selected. Comparison will be done row by row.")
                join_keys = None

        # Store join keys in session state
        st.session_state.join_keys = join_keys

        # Compare button
        if st.button("Compare Data", type="primary"):
            perform_comparison()

def handle_data_source(source_type: str, prefix: str) -> Optional[pd.DataFrame]:
    """Handle different types of data sources"""
    
    try:
        if source_type in ["CSV File", "DAT File", "Parquet File", "Zipped Flat Files"]:
            return handle_file_upload(source_type, prefix)
        elif source_type in ["SQL Server", "Teradata", "Stored Procedure"]:
            return handle_database_connection(source_type, prefix)
        elif source_type == "API":
            return handle_api_connection(prefix)
        
    except Exception as e:
        log_error(f"Error handling {source_type}: {str(e)}")
        st.error(f"Error processing {source_type}: {str(e)}")
    
    return None

def handle_file_upload(file_type: str, prefix: str) -> Optional[pd.DataFrame]:
    """Handle file upload for different file types"""
    
    uploaded_file = st.file_uploader(f"Upload {file_type}", key=f"{prefix}_file")
    
    if uploaded_file:
        delimiter = st.text_input(f"Delimiter (for {file_type})", 
                                value=',' if file_type == "CSV File" else '|',
                                key=f"{prefix}_delimiter")
        
        try:
            if file_type == "CSV File":
                return DataReader.load_csv(uploaded_file, delimiter=delimiter)
            elif file_type == "DAT File":
                return DataReader.load_dat(uploaded_file, delimiter=delimiter)
            elif file_type == "Parquet File":
                return DataReader.load_parquet(uploaded_file)
            elif file_type == "Zipped Flat Files":
                return DataReader.load_zipped_flat_files(uploaded_file, separator=delimiter)
                
        except Exception as e:
            st.error(f"Error reading {file_type}: {str(e)}")
    
    return None

def handle_database_connection(db_type: str, prefix: str) -> Optional[pd.DataFrame]:
    """Handle database connections"""
    
    with st.expander(f"{db_type} Connection Details"):
        host = st.text_input("Host", key=f"{prefix}_host")
        database = st.text_input("Database", key=f"{prefix}_database")
        username = st.text_input("Username", key=f"{prefix}_username")
        password = st.text_input("Password", type="password", key=f"{prefix}_password")
        
        if db_type == "Stored Procedure":
            proc_name = st.text_input("Stored Procedure Name", key=f"{prefix}_proc")
            params = st.text_area("Parameters (JSON format)", key=f"{prefix}_params")
        else:
            query = st.text_area("SQL Query", key=f"{prefix}_query")

        if st.button("Connect", key=f"{prefix}_connect"):
            try:
                conn_params = {
                    'host': host,
                    'database': database,
                    'username': username,
                    'password': password
                }
                
                if db_type == "SQL Server":
                    return DatabaseConnector.get_sqlserver_data(conn_params, query)
                elif db_type == "Teradata":
                    return DatabaseConnector.get_teradata_data(conn_params, query)
                elif db_type == "Stored Procedure":
                    return DatabaseConnector.get_data_from_stored_proc(
                        conn_params, proc_name, eval(params) if params else None)
                        
            except Exception as e:
                st.error(f"Database connection error: {str(e)}")
    
    return None

def handle_api_connection(prefix: str) -> Optional[pd.DataFrame]:
    """Handle API connections"""
    
    with st.expander("API Connection Details"):
        api_url = st.text_input("API URL", key=f"{prefix}_api_url")
        method = st.selectbox("Method", ["GET", "POST"], key=f"{prefix}_method")
        headers = st.text_area("Headers (JSON format)", key=f"{prefix}_headers")
        params = st.text_area("Parameters (JSON format)", key=f"{prefix}_params")
        
        if st.button("Connect", key=f"{prefix}_connect"):
            try:
                return APIFetcher.fetch_api_data(
                    api_url=api_url,
                    method=method,
                    headers=eval(headers) if headers else None,
                    params=eval(params) if params else None
                )
            except Exception as e:
                st.error(f"API connection error: {str(e)}")
    
    return None

def show_column_mapping_interface(source_df: pd.DataFrame, target_df: pd.DataFrame):
    """Show interface for column mapping"""
    
    # Auto-generate initial mapping if not already done
    if not st.session_state.column_mapping:
        st.session_state.column_mapping = MappingManager.auto_map_columns(source_df, target_df)

    st.markdown("### Column Mapping Configuration")
    
    # Create three columns for the mapping interface
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown("**Source Column**")
    with col2:
        st.markdown("**Target Column**")
    with col3:
        st.markdown("**Exclude**")

    # Show mapping interface for each source column
    for source_col in source_df.columns:
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.text(source_col)
        
        with col2:
            # Dropdown for target column selection
            target_col = st.selectbox(
                f"Map to",
                options=[''] + list(target_df.columns),
                index=0 if source_col not in st.session_state.column_mapping 
                else list(target_df.columns).index(st.session_state.column_mapping[source_col]) + 1,
                key=f"mapping_{source_col}"
            )
            
            if target_col:
                st.session_state.column_mapping[source_col] = target_col
            elif source_col in st.session_state.column_mapping:
                del st.session_state.column_mapping[source_col]
        
        with col3:
            # Checkbox for excluding column from comparison
            excluded = st.checkbox(
                "Exclude",
                key=f"exclude_{source_col}",
                value=source_col in st.session_state.excluded_columns
            )
            
            if excluded and source_col not in st.session_state.excluded_columns:
                st.session_state.excluded_columns.append(source_col)
            elif not excluded and source_col in st.session_state.excluded_columns:
                st.session_state.excluded_columns.remove(source_col)

def perform_comparison():
    """Perform the comparison and generate reports"""
    
    try:
        st.markdown("### Comparison Results")
        
        # Create reports directory if it doesn't exist
        os.makedirs("reports", exist_ok=True)
        timestamp = format_timestamp()
        
        # Generate reports
        join_keys = st.session_state.get('join_keys', None)
        
        with st.spinner("Generating comparison reports..."):
                # Generate difference report
                diff_df, has_differences = ReportGenerator.generate_diff_report(
                    st.session_state.source_df,
                    st.session_state.target_df,
                    st.session_state.column_mapping,
                    st.session_state.excluded_columns,
                    join_keys=join_keys
                )
                
                # Generate profiling report
                profile_df = ReportGenerator.generate_profiling_report(
                    st.session_state.source_df,
                    st.session_state.target_df,
                    st.session_state.column_mapping,
                    join_keys=join_keys
                )
                
                # Generate regression report (independent of join keys)
                regression_path = f"reports/RegressionReport_{timestamp}.xlsx"
                ReportGenerator.generate_regression_report(
                    st.session_state.source_df,
                    st.session_state.target_df,
                    st.session_state.column_mapping,
                    regression_path
                )
                
                # Generate side-by-side report
                side_by_side_path = f"reports/DifferenceReport_{timestamp}.xlsx"
                side_by_side_df, _ = ReportGenerator.generate_side_by_side_report(
                    st.session_state.source_df,
                    st.session_state.target_df,
                    st.session_state.column_mapping,
                    side_by_side_path,
                    join_keys=join_keys
                )

        # Display results and download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Data Comparison Results")
            st.dataframe(diff_df)
            
            if os.path.exists(side_by_side_path):
                with open(side_by_side_path, 'rb') as f:
                    st.download_button(
                        "Download Difference Report",
                        f,
                        file_name=f"DifferenceReport_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        with col2:
            st.markdown("#### Y-Data Profiling Results")
            if not profile_df.empty:
                # Format the dataframe for better display
                st.write("Column-wise Statistical Comparison:")
                
                # Create an expander for detailed view
                with st.expander("View Detailed Profiling Report", expanded=True):
                    # Display main metrics
                    st.write("Main Metrics:")
                    main_metrics = profile_df[['Column', 'Source_Count', 'Target_Count', 'Match_Percentage']]
                    st.dataframe(main_metrics, use_container_width=True)
                    
                    # Display detailed statistics
                    st.write("Detailed Statistics:")
                    st.dataframe(profile_df, use_container_width=True)
                    
                    # Download button for profiling report
                    profiling_path = f"reports/ProfilingReport_{timestamp}.xlsx"
                    profile_df.to_excel(profiling_path, index=False)
                    with open(profiling_path, 'rb') as f:
                        st.download_button(
                            "Download Complete Profiling Report",
                            f,
                            file_name=f"ProfilingReport_{timestamp}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            else:
                st.error("Unable to generate profiling report. Please check the data and try again.")
            
            if os.path.exists(regression_path):
                with open(regression_path, 'rb') as f:
                    st.download_button(
                        "Download Regression Report",
                        f,
                        file_name=f"RegressionReport_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

    except Exception as e:
        log_error(f"Error performing comparison: {str(e)}")
        st.error(f"Error performing comparison: {str(e)}")

if __name__ == "__main__":
    main()

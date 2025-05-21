import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import os
from datetime import datetime
import datacompy
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components

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

    # Initialize session state variables
    if 'source_df' not in st.session_state:
        st.session_state.source_df = None
    if 'target_df' not in st.session_state:
        st.session_state.target_df = None
    if 'column_mapping' not in st.session_state:
        st.session_state.column_mapping = {}
    if 'excluded_columns' not in st.session_state:
        st.session_state.excluded_columns = []
    if 'report_paths' not in st.session_state:
        st.session_state.report_paths = {}

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
        
        # Initialize or update column mapping
        if not st.session_state.column_mapping:
            st.session_state.column_mapping = MappingManager.auto_map_columns(source_data, target_data)

        # Show column mapping interface
        st.subheader("Column Mapping")
        show_column_mapping_interface(source_data, target_data)

        # Add join key selection based on mapped columns
        st.subheader("Join Key Selection")
        
        # Get mapped columns (where source and target are mapped)
        mapped_columns = {source_col: target_col 
                        for source_col, target_col in st.session_state.column_mapping.items()
                        if source_col in source_data.columns and target_col in target_data.columns}
        
        if not mapped_columns:
            st.warning("No mapped columns available for join keys. Please map columns first.")
            join_keys = None
        else:
            # Show mapped columns available for joining
            st.write("Mapped columns available for joining:")
            mapped_preview = []
            for source_col, target_col in mapped_columns.items():
                st.write(f"- Source: {source_col} → Target: {target_col}")
                # Get sample values
                source_sample = source_data[source_col].head(3).tolist()
                target_sample = target_data[target_col].head(3).tolist()
                mapped_preview.append({
                    'Source Column': source_col,
                    'Target Column': target_col,
                    'Source Sample': str(source_sample),
                    'Target Sample': str(target_sample)
                })
            
            # Show preview of mapped columns
            if mapped_preview:
                st.write("Preview of mapped columns:")
                st.dataframe(pd.DataFrame(mapped_preview))
            
            # Select join keys from mapped columns
            selected_keys = st.multiselect(
                "Select Join Key(s)",
                options=list(mapped_columns.keys()),
                help="Select one or more mapped columns to use as join keys for comparison. These columns should uniquely identify records."
            )
            
            if selected_keys:
                # Create the join keys mapping (source to target columns)
                join_keys = [(source_col, mapped_columns[source_col]) for source_col in selected_keys]
                st.write("Selected join keys:")
                for source_col, target_col in join_keys:
                    st.write(f"- {source_col} → {target_col}")
            else:
                st.warning("No join keys selected. Comparison will be done row by row.")
                join_keys = None

        # Store join keys in session state
        st.session_state.join_keys = join_keys

        # Compare button
        if st.button("Compare Data", type="primary"):
            with st.spinner("Generating comparison reports..."):
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
        
        if db_type in ["SQL Server", "Stored Procedure"]:
            use_windows_auth = st.checkbox("Use Windows Authentication", value=True, key=f"{prefix}_use_windows_auth")
            if not use_windows_auth:
                username = st.text_input("Username", key=f"{prefix}_username")
                password = st.text_input("Password", type="password", key=f"{prefix}_password")
        else:
            use_windows_auth = False
            username = st.text_input("Username", key=f"{prefix}_username")
            password = st.text_input("Password", type="password", key=f"{prefix}_password")
        
        if db_type == "Stored Procedure":
            proc_name = st.text_input("Stored Procedure Name", key=f"{prefix}_proc")
            params = st.text_area("Parameters (JSON format)", key=f"{prefix}_params")
        else:
            query = st.text_area("SQL Query", key=f"{prefix}_query")

        if st.button("Connect", key=f"{prefix}_connect"):
            try:
                # Build connection parameters
                conn_params = {
                    'host': host,
                    'database': database,
                    'use_windows_auth': use_windows_auth if db_type in ["SQL Server", "Stored Procedure"] else False
                }
                
                # Add username/password only if not using Windows Auth
                if not (db_type in ["SQL Server", "Stored Procedure"] and use_windows_auth):
                    conn_params.update({
                        'username': username,
                        'password': password
                    })
                
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
    
    st.markdown("### Column Mapping Configuration")
    
    # Add auto-map button
    if st.button("Auto-Map Columns", key="auto_map_btn"):
        st.session_state.column_mapping = MappingManager.auto_map_columns(source_df, target_df)
        st.success("Columns automatically mapped!")
    
    # Show current mapping status
    if st.session_state.column_mapping:
        st.write(f"Currently mapped: {len(st.session_state.column_mapping)} columns")
        
        # Create a DataFrame to display the mapping
        mapping_data = []
        for source_col in source_df.columns:
            source_sample = str(source_df[source_col].head(2).tolist())
            target_col = st.session_state.column_mapping.get(source_col, '')
            target_sample = str(target_df[target_col].head(2).tolist()) if target_col else ''
            
            mapping_data.append({
                'Source Column': source_col,
                'Source Sample': source_sample,
                'Target Column': target_col,
                'Target Sample': target_sample,
                'Mapped': '✓' if target_col else '✗'
            })
        
        mapping_df = pd.DataFrame(mapping_data)
        st.dataframe(mapping_df, use_container_width=True)
    
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
            st.caption(f"Sample: {str(source_df[source_col].head(2).tolist())}")
        
        with col2:
            # Dropdown for target column selection
            current_mapping = st.session_state.column_mapping.get(source_col, '')
            target_options = [''] + list(target_df.columns)
            target_index = target_options.index(current_mapping) if current_mapping in target_options else 0
            
            target_col = st.selectbox(
                f"Map to",
                options=target_options,
                index=target_index,
                key=f"mapping_{source_col}"
            )
            
            if target_col:
                if target_col != current_mapping:
                    st.session_state.column_mapping[source_col] = target_col
                    # Show sample of selected target column
                    st.caption(f"Sample: {str(target_df[target_col].head(2).tolist())}")
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
            try:
                from ydata_profiling import ProfileReport
                import streamlit.components.v1 as components
                
                # Generate Y-Data HTML profiling reports
                st.write("Generating Y-Data Profiling Reports...")
                
                # Source profiling
                source_profile = ProfileReport(
                    st.session_state.source_df,
                    title="Source Data Profile Report",
                    minimal=True
                )
                source_html = f"reports/SourceProfile_{timestamp}.html"
                source_profile.to_file(source_html)
                
                # Target profiling
                target_profile = ProfileReport(
                    st.session_state.target_df,
                    title="Target Data Profile Report",
                    minimal=True
                )
                target_html = f"reports/TargetProfile_{timestamp}.html"
                target_profile.to_file(target_html)
                
                # Generate other reports
                st.write("Generating Comparison Reports...")
                
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
                
                # Display Y-Data Profiling Reports in expandable sections
                st.subheader("Y-Data Profiling Reports")

                # Generate comparison profile
                comparison_profile = ProfileReport(
                    pd.concat([
                        st.session_state.source_df.add_prefix('Source_'),
                        st.session_state.target_df.add_prefix('Target_')
                    ], axis=1),
                    title="Source vs Target Comparison Profile",
                    minimal=True
                )
                comparison_html = f"reports/ComparisonProfile_{timestamp}.html"
                comparison_profile.to_file(comparison_html)

                # Generate DataCompy comparison report
                try:
                    # Prepare join keys for DataCompy
                    join_columns = []
                    source_df = st.session_state.source_df.copy()
                    target_df = st.session_state.target_df.copy()
                    
                    if join_keys:
                        join_columns = [key[0] for key in join_keys]  # Use source column names
                    else:
                        # If no join keys specified, use index
                        source_df['_index'] = source_df.index
                        target_df['_index'] = target_df.index
                        join_columns = ['_index']

                    # Create DataCompy comparison
                    comparison = datacompy.Compare(
                        source_df,
                        target_df,
                        join_columns=join_columns,
                        df1_name='Source',
                        df2_name='Target'
                    )

                    # Get column statistics safely
                    column_stats_html = '<p>No column statistics available.</p>'
                    try:
                        if (hasattr(comparison, 'column_stats') and 
                            isinstance(comparison.column_stats, pd.DataFrame) and 
                            not comparison.column_stats.empty):
                            column_stats_html = comparison.column_stats.to_html()
                    except Exception as e:
                        st.warning(f"Could not generate column statistics: {str(e)}")

                    # Generate HTML report
                    datacompy_html = f"reports/DataCompyReport_{timestamp}.html"
                    with open(datacompy_html, 'w') as f:
                        f.write(f"""
                        <html>
                        <head>
                            <title>DataCompy Comparison Report</title>
                            <style>
                                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                                .report {{ max-width: 1200px; margin: 0 auto; }}
                                .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                                .match {{ color: green; }}
                                .mismatch {{ color: red; }}
                                table {{ border-collapse: collapse; width: 100%; }}
                                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                                th {{ background-color: #f5f5f5; }}
                            </style>
                        </head>
                        <body>
                            <div class="report">
                                <h1>DataCompy Comparison Report</h1>
                                <div class="section">
                                    <h2>Summary</h2>
                                    <pre>{comparison.report()}</pre>
                                </div>
                                <div class="section">
                                    <h2>Detailed Statistics</h2>
                                    <h3>Matches</h3>
                                    <div class="match">
                                        <p>Number of rows match: {comparison.count_matching_rows()}</p>
                                        <p>Number of columns match: {len(set(source_df.columns).intersection(target_df.columns))}</p>
                                    </div>
                                    <h3>Mismatches</h3>
                                    <div class="mismatch">
                                        <p>Rows only in Source: {len(comparison.df1_unq_rows)}</p>
                                        <p>Rows only in Target: {len(comparison.df2_unq_rows)}</p>
                                        <p>Source-only columns: {len(set(source_df.columns) - set(target_df.columns))}</p>
                                        <p>Target-only columns: {len(set(target_df.columns) - set(source_df.columns))}</p>
                                    </div>
                                </div>
                                <div class="section">
                                    <h2>Column Statistics</h2>
                                    {column_stats_html}
                                </div>
                            </div>
                        </body>
                        </html>
                        """)
                except Exception as e:
                    st.error(f"Error generating DataCompy report: {str(e)}")
                    datacompy_html = None

                # Store DataCompy report path
                st.session_state.report_paths['datacompy_html'] = datacompy_html
                
                # Store report paths in session state to prevent page reset
                if 'report_paths' not in st.session_state:
                    st.session_state.report_paths = {}
                st.session_state.report_paths.update({
                    'source_html': source_html,
                    'target_html': target_html,
                    'comparison_html': comparison_html
                })
                
                tab1, tab2, tab3, tab4 = st.tabs(["Source Profile", "Target Profile", "Comparison Profile", "DataCompy Report"])
                
                with tab1:
                    with open(source_html, 'r', encoding='utf-8') as f:
                        components.html(f.read(), height=600, scrolling=True)
                    
                    # Download button for source profile
                    with open(source_html, 'rb') as f:
                        st.download_button(
                            "Download Source Profile Report",
                            f,
                            file_name=f"SourceProfile_{timestamp}.html",
                            mime="text/html",
                            key="download_source"
                        )
                
                with tab2:
                    with open(target_html, 'r', encoding='utf-8') as f:
                        components.html(f.read(), height=600, scrolling=True)
                    
                    # Download button for target profile
                    with open(target_html, 'rb') as f:
                        st.download_button(
                            "Download Target Profile Report",
                            f,
                            file_name=f"TargetProfile_{timestamp}.html",
                            mime="text/html",
                            key="download_target"
                        )

                with tab3:
                    with open(comparison_html, 'r', encoding='utf-8') as f:
                        components.html(f.read(), height=600, scrolling=True)
                    
                    # Download button for comparison profile
                    with open(comparison_html, 'rb') as f:
                        st.download_button(
                            "Download Comparison Profile Report",
                            f,
                            file_name=f"ComparisonProfile_{timestamp}.html",
                            mime="text/html",
                            key="download_comparison"
                        )

                with tab4:
                    if datacompy_html and os.path.exists(datacompy_html):
                        with open(datacompy_html, 'r', encoding='utf-8') as f:
                            components.html(f.read(), height=600, scrolling=True)
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            # Download button for DataCompy report
                            with open(datacompy_html, 'rb') as f:
                                st.download_button(
                                    "📥 Download DataCompy Report",
                                    f,
                                    file_name=f"DataCompyReport_{timestamp}.html",
                                    mime="text/html",
                                    key="download_datacompy",
                                    use_container_width=True
                                )
                        
                        # Show additional DataCompy insights
                        with st.expander("📊 DataCompy Insights", expanded=False):
                            try:
                                st.write("### Match Statistics")
                                col1, col2, col3 = st.columns(3)
                                
                                matching_rows = comparison.count_matching_rows()
                                source_only_rows = len(comparison.df1_unq_rows)
                                target_only_rows = len(comparison.df2_unq_rows)
                                
                                with col1:
                                    st.metric(
                                        "Matching Rows",
                                        matching_rows,
                                        delta=f"{matching_rows / len(source_df) * 100:.1f}%"
                                    )
                                
                                with col2:
                                    st.metric(
                                        "Source Only Rows",
                                        source_only_rows,
                                        delta=f"{source_only_rows / len(source_df) * 100:.1f}%"
                                    )
                                
                                with col3:
                                    st.metric(
                                        "Target Only Rows",
                                        target_only_rows,
                                        delta=f"{target_only_rows / len(target_df) * 100:.1f}%"
                                    )
                                
                                st.write("### Column Analysis")
                                
                                # Get column sets
                                source_cols = set(source_df.columns)
                                target_cols = set(target_df.columns)
                                common_cols = source_cols.intersection(target_cols)
                                source_only = source_cols - target_cols
                                target_only = target_cols - source_cols

                                # Display column sets
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("Source-only Columns:")
                                    if source_only:
                                        for col in sorted(source_only):
                                            st.info(f"- {col}")
                                    else:
                                        st.success("No source-only columns")
                                
                                with col2:
                                    st.write("Target-only Columns:")
                                    if target_only:
                                        for col in sorted(target_only):
                                            st.info(f"- {col}")
                                    else:
                                        st.success("No target-only columns")
                                
                                st.write("### Common Column Analysis")
                                if common_cols:
                                    for col in sorted(common_cols):
                                        # Calculate match rate for common columns
                                        source_values = set(source_df[col].dropna())
                                        target_values = set(target_df[col].dropna())
                                        common_values = source_values.intersection(target_values)
                                        all_values = source_values.union(target_values)
                                        match_rate = len(common_values) / len(all_values) if all_values else 1.0
                                        
                                        # Display progress bar with details
                                        st.progress(
                                            match_rate,
                                            text=f"{col}: {match_rate*100:.1f}% unique values match"
                                        )
                                        
                                        with st.expander(f"Details for {col}", expanded=False):
                                            st.write(f"- Unique values in Source: {len(source_values)}")
                                            st.write(f"- Unique values in Target: {len(target_values)}")
                                            st.write(f"- Common unique values: {len(common_values)}")
                                            if len(source_values - target_values) > 0:
                                                st.write("- Values only in Source:", list(source_values - target_values)[:5])
                                            if len(target_values - source_values) > 0:
                                                st.write("- Values only in Target:", list(target_values - source_values)[:5])
                                else:
                                    st.warning("No common columns found between source and target datasets")
                            except Exception as e:
                                st.error(f"Error displaying insights: {str(e)}")
                    else:
                        st.error("❌ DataCompy report generation failed. Please check the data and try again.")
                        
                        with col2:
                            st.write("Target-only Columns:")
                            if target_only:
                                for col in sorted(target_only):
                                    st.info(f"- {col}")
                            else:
                                st.success("No target-only columns")
                        
                        st.write("### Common Column Analysis")
                        if common_cols:
                            for col in sorted(common_cols):
                                # Calculate match rate for common columns
                                source_values = set(st.session_state.source_df[col].dropna())
                                target_values = set(st.session_state.target_df[col].dropna())
                                common_values = source_values.intersection(target_values)
                                all_values = source_values.union(target_values)
                                match_rate = len(common_values) / len(all_values) if all_values else 1.0
                                
                                # Display progress bar with details
                                st.progress(
                                    match_rate,
                                    text=f"{col}: {match_rate*100:.1f}% unique values match"
                                )
                                
                                with st.expander(f"Details for {col}", expanded=False):
                                    st.write(f"- Unique values in Source: {len(source_values)}")
                                    st.write(f"- Unique values in Target: {len(target_values)}")
                                    st.write(f"- Common unique values: {len(common_values)}")
                                    if len(source_values - target_values) > 0:
                                        st.write("- Values only in Source:", list(source_values - target_values)[:5])
                                    if len(target_values - source_values) > 0:
                                        st.write("- Values only in Target:", list(target_values - source_values)[:5])
                        else:
                            st.warning("No common columns found between source and target datasets")
                
            except ImportError:
                st.warning("Y-Data Profiling package not installed. Please install ydata-profiling package.")
                # Continue with other reports...

        # Store report paths in session state
        st.session_state.report_paths.update({
            'side_by_side_path': side_by_side_path,
            'profiling_path': f"reports/ProfilingReport_{timestamp}.xlsx",
            'regression_path': regression_path
        })

        # Create zip file containing all reports
        st.divider()
        zip_path = f"reports/FinalComparison_{timestamp}.zip"
        try:
            import zipfile
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                # Add all reports to zip
                for report_name, report_path in st.session_state.report_paths.items():
                    if report_path and os.path.exists(report_path):
                        zipf.write(report_path, os.path.basename(report_path))
            
            # Add download button for zip file
            st.markdown("### 📦 Download Complete Comparison Package")
            st.caption("Contains all reports: Regression, Side by Side, DataCompy, and Profile Reports")
            with open(zip_path, 'rb') as f:
                st.download_button(
                    "📥 Download All Reports (ZIP)",
                    f,
                    file_name=f"FinalComparison_{timestamp}.zip",
                    mime="application/zip",
                    key="download_all",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Error creating zip file: {str(e)}")
        st.divider()

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
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_difference"
                    )
        
        with col2:
            st.markdown("#### Y-Data Profiling Results")
            if not profile_df.empty:
                # Format the dataframe for better display
                st.write("Column-wise Statistical Comparison:")
                
                # Create tabs for different views
                metrics_tab, details_tab = st.tabs(["Main Metrics", "Detailed Statistics"])
                
                with metrics_tab:
                    main_metrics = profile_df[['Column', 'Source_Count', 'Target_Count', 'Match_Percentage']]
                    st.dataframe(main_metrics, use_container_width=True)
                
                with details_tab:
                    st.dataframe(profile_df, use_container_width=True)
                
                # Save profiling report
                profiling_path = st.session_state.report_paths['profiling_path']
                profile_df.to_excel(profiling_path, index=False)
                
                # Download buttons with unique keys
                col3, col4 = st.columns(2)
                with col3:
                    with open(profiling_path, 'rb') as f:
                        st.download_button(
                            "Download Profiling Report",
                            f,
                            file_name=f"ProfilingReport_{timestamp}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_profiling"
                        )
                
                with col4:
                    if os.path.exists(regression_path):
                        with open(regression_path, 'rb') as f:
                            st.download_button(
                                "Download Regression Report",
                                f,
                                file_name=f"RegressionReport_{timestamp}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="download_regression"
                            )
            else:
                st.error("Unable to generate profiling report. Please check the data and try again.")

    except Exception as e:
        log_error(f"Error performing comparison: {str(e)}")
        st.error(f"Error performing comparison: {str(e)}")

if __name__ == "__main__":
    main()

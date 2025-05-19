import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import openpyxl
from openpyxl.styles import PatternFill, Font
from utils import log_error, format_timestamp

class ReportGenerator:
    """Class to handle generation of comparison reports"""

    # Define colors for Excel reports
    PASS_COLOR = PatternFill(start_color='90EE90', end_color='90EE90', fill_type='solid')  # Light green
    FAIL_COLOR = PatternFill(start_color='FFB6C1', end_color='FFB6C1', fill_type='solid')  # Light pink
    HEADER_COLOR = PatternFill(start_color='4682B4', end_color='4682B4', fill_type='solid')  # Steel blue
    HEADER_FONT = Font(color='FFFFFF', bold=True)  # White, bold

    @staticmethod
    def generate_diff_report(
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        column_mapping: Dict[str, str],
        exclude_columns: List[str] = None
    ) -> Tuple[pd.DataFrame, bool]:
        """
        Generate a difference report between source and target DataFrames.
        
        Args:
            source_df: Source DataFrame
            target_df: Target DataFrame
            column_mapping: Dictionary mapping source columns to target columns
            exclude_columns: List of columns to exclude from comparison
            
        Returns:
            Tuple[pd.DataFrame, bool]: (Difference report DataFrame, Whether differences were found)
        """
        try:
            # Filter out excluded columns
            if exclude_columns:
                column_mapping = {k: v for k, v in column_mapping.items() 
                                if k not in exclude_columns}

            differences = []
            
            # Compare mapped columns
            for source_col, target_col in column_mapping.items():
                source_values = source_df[source_col]
                target_values = target_df[target_col]
                
                # Find rows where values don't match
                mask = source_values != target_values
                if mask.any():
                    diff_indices = mask[mask].index
                    for idx in diff_indices:
                        differences.append({
                            'Column': source_col,
                            'Row_Index': idx,
                            'Source_Value': source_values[idx],
                            'Target_Value': target_values[idx]
                        })

            # Create difference report DataFrame
            if differences:
                diff_df = pd.DataFrame(differences)
                return diff_df, True
            else:
                return pd.DataFrame(columns=['Message']).append(
                    {'Message': 'No differences found'}, ignore_index=True), False

        except Exception as e:
            log_error(f"Error generating difference report: {str(e)}")
            return pd.DataFrame(), False

    @staticmethod
    def generate_profiling_report(
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        column_mapping: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Generate Y-Data profiling comparison report.
        
        Args:
            source_df: Source DataFrame
            target_df: Target DataFrame
            column_mapping: Dictionary mapping source columns to target columns
            
        Returns:
            pd.DataFrame: Profiling report DataFrame
        """
        try:
            profile_data = []
            
            for source_col, target_col in column_mapping.items():
                source_stats = ReportGenerator._calculate_column_stats(source_df[source_col])
                target_stats = ReportGenerator._calculate_column_stats(target_df[target_col])
                
                profile_data.append({
                    'Column': source_col,
                    'Source_Count': source_stats['count'],
                    'Target_Count': target_stats['count'],
                    'Source_Nulls': source_stats['null_count'],
                    'Target_Nulls': target_stats['null_count'],
                    'Source_Distinct': source_stats['distinct_count'],
                    'Target_Distinct': target_stats['distinct_count'],
                    'Source_Min': source_stats['min'],
                    'Target_Min': target_stats['min'],
                    'Source_Max': source_stats['max'],
                    'Target_Max': target_stats['max'],
                    'Source_Mean': source_stats.get('mean', 'N/A'),
                    'Target_Mean': target_stats.get('mean', 'N/A')
                })
            
            return pd.DataFrame(profile_data)

        except Exception as e:
            log_error(f"Error generating profiling report: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def _calculate_column_stats(series: pd.Series) -> Dict[str, Any]:
        """Calculate statistical measures for a column."""
        stats = {
            'count': len(series),
            'null_count': series.isnull().sum(),
            'distinct_count': series.nunique(),
            'min': series.min() if series.dtype in ['int64', 'float64'] else str(series.min()),
            'max': series.max() if series.dtype in ['int64', 'float64'] else str(series.max())
        }
        
        if series.dtype in ['int64', 'float64']:
            stats['mean'] = series.mean()
            
        return stats

    @staticmethod
    def generate_regression_report(
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        column_mapping: Dict[str, str],
        output_path: str
    ) -> bool:
        """
        Generate Excel-based regression report with multiple tabs.
        
        Args:
            source_df: Source DataFrame
            target_df: Target DataFrame
            column_mapping: Dictionary mapping source columns to target columns
            output_path: Path to save the Excel report
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Generate AggregationCheck tab
                ReportGenerator._generate_aggregation_check(
                    source_df, target_df, column_mapping, writer)
                
                # Generate CountCheck tab
                ReportGenerator._generate_count_check(
                    source_df, target_df, writer)
                
                # Generate DistinctCheck tab
                ReportGenerator._generate_distinct_check(
                    source_df, target_df, column_mapping, writer)
                
                # Apply formatting
                workbook = writer.book
                for sheet_name in workbook.sheetnames:
                    ReportGenerator._apply_excel_formatting(workbook[sheet_name])
                
            return True

        except Exception as e:
            log_error(f"Error generating regression report: {str(e)}")
            return False

    @staticmethod
    def _generate_aggregation_check(
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        column_mapping: Dict[str, str],
        writer: pd.ExcelWriter
    ) -> None:
        """Generate AggregationCheck tab in the regression report."""
        agg_data = []
        
        for source_col, target_col in column_mapping.items():
            if source_df[source_col].dtype in ['int64', 'float64']:
                source_sum = source_df[source_col].sum()
                target_sum = target_df[target_col].sum()
                
                agg_data.append({
                    'Source_Column': source_col,
                    'Target_Column': target_col,
                    'Source_Sum': source_sum,
                    'Target_Sum': target_sum,
                    'Result': 'PASS' if np.isclose(source_sum, target_sum) else 'FAIL'
                })
        
        pd.DataFrame(agg_data).to_excel(writer, sheet_name='AggregationCheck', index=False)

    @staticmethod
    def _generate_count_check(
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        writer: pd.ExcelWriter
    ) -> None:
        """Generate CountCheck tab in the regression report."""
        count_data = [{
            'Source_File_Name': 'Source Dataset',
            'Source_Path': 'N/A',
            'Data_Count': len(source_df),
            'Target_Count': len(target_df),
            'Result': 'PASS' if len(source_df) == len(target_df) else 'FAIL'
        }]
        
        pd.DataFrame(count_data).to_excel(writer, sheet_name='CountCheck', index=False)

    @staticmethod
    def _generate_distinct_check(
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        column_mapping: Dict[str, str],
        writer: pd.ExcelWriter
    ) -> None:
        """Generate DistinctCheck tab in the regression report."""
        distinct_data = []
        
        for source_col, target_col in column_mapping.items():
            if source_df[source_col].dtype not in ['int64', 'float64']:
                source_distinct = set(source_df[source_col].dropna().unique())
                target_distinct = set(target_df[target_col].dropna().unique())
                
                distinct_data.append({
                    'Column': source_col,
                    'Source_Distinct_Count': len(source_distinct),
                    'Target_Distinct_Count': len(target_distinct),
                    'Source_Distinct_Values': ', '.join(map(str, sorted(source_distinct))),
                    'Target_Distinct_Values': ', '.join(map(str, sorted(target_distinct))),
                    'Result': 'PASS' if source_distinct == target_distinct else 'FAIL'
                })
        
        pd.DataFrame(distinct_data).to_excel(writer, sheet_name='DistinctCheck', index=False)

    @staticmethod
    def _apply_excel_formatting(worksheet: openpyxl.worksheet.worksheet.Worksheet) -> None:
        """Apply formatting to Excel worksheet."""
        # Format headers
        for cell in worksheet[1]:
            cell.fill = ReportGenerator.HEADER_COLOR
            cell.font = ReportGenerator.HEADER_FONT

        # Format PASS/FAIL cells
        for row in worksheet.iter_rows(min_row=2):
            for cell in row:
                if cell.value == 'PASS':
                    cell.fill = ReportGenerator.PASS_COLOR
                elif cell.value == 'FAIL':
                    cell.fill = ReportGenerator.FAIL_COLOR

    @staticmethod
    def generate_side_by_side_report(
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        column_mapping: Dict[str, str],
        output_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, str]:
        """
        Generate side-by-side difference report.
        
        Args:
            source_df: Source DataFrame
            target_df: Target DataFrame
            column_mapping: Dictionary mapping source columns to target columns
            output_path: Optional path to save Excel report
            
        Returns:
            Tuple[pd.DataFrame, str]: (Difference report DataFrame, Output path if saved)
        """
        try:
            # Create side-by-side comparison
            comparison_data = []
            
            for source_col, target_col in column_mapping.items():
                source_values = source_df[source_col]
                target_values = target_df[target_col]
                
                # Find differences
                mask = source_values != target_values
                if mask.any():
                    diff_indices = mask[mask].index
                    for idx in diff_indices:
                        comparison_data.append({
                            'Row_Index': idx,
                            f'Source_{source_col}': source_values[idx],
                            f'Target_{target_col}': target_values[idx]
                        })

            if not comparison_data:
                diff_df = pd.DataFrame({'Message': ['No differences found']})
            else:
                diff_df = pd.DataFrame(comparison_data)

            # Save to Excel if path provided
            if output_path:
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    diff_df.to_excel(writer, index=False)
                    ReportGenerator._apply_excel_formatting(writer.book.active)
                return diff_df, output_path
            
            return diff_df, ''

        except Exception as e:
            log_error(f"Error generating side-by-side report: {str(e)}")
            return pd.DataFrame(), ''

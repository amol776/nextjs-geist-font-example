import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from utils import log_error

class MappingManager:
    """Class to handle column mappings and data type conversions"""
    
    # Define the type mapping dictionary as a class variable
    TYPE_MAPPING = {
        'int': 'int32',
        'int64': 'int64',
        'numeric': 'int64',
        'bigint': 'int64',
        'smallint': 'int64',
        'varchar': 'string',
        'nvarchar': 'string',
        'char': 'string',
        'date': 'datetime64[ns]',
        'datetime': 'datetime64[ns]',
        'decimal': 'float',
        'float': 'float',
        'bit': 'bool',
        'nchar': 'char',
        'boolean': 'bool'
    }

    @classmethod
    def auto_map_columns(cls, source_df: pd.DataFrame, target_df: pd.DataFrame) -> Dict[str, str]:
        """
        Automatically map columns between source and target DataFrames.
        
        Args:
            source_df: Source DataFrame
            target_df: Target DataFrame
            
        Returns:
            dict: Mapping of source columns to target columns
        """
        mapping = {}
        source_cols = [col.lower() for col in source_df.columns]
        target_cols = [col.lower() for col in target_df.columns]
        
        # First pass: exact matches
        for idx, source_col in enumerate(source_cols):
            if source_col in target_cols:
                mapping[source_df.columns[idx]] = target_df.columns[target_cols.index(source_col)]
        
        # Second pass: fuzzy matches for unmapped columns
        unmapped_source = [col for col in source_df.columns if col not in mapping]
        for source_col in unmapped_source:
            # Simple fuzzy matching (remove special characters and spaces)
            clean_source = ''.join(e.lower() for e in source_col if e.isalnum())
            best_match = None
            best_score = 0
            
            for target_col in target_df.columns:
                clean_target = ''.join(e.lower() for e in target_col if e.isalnum())
                
                # Calculate similarity score
                score = cls._calculate_similarity(clean_source, clean_target)
                if score > best_score and score > 0.8:  # 80% similarity threshold
                    best_score = score
                    best_match = target_col
            
            if best_match:
                mapping[source_col] = best_match
        
        return mapping

    @staticmethod
    def _calculate_similarity(str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings using Levenshtein distance.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if not str1 or not str2:
            return 0
            
        # Simple implementation of Levenshtein distance
        if len(str1) < len(str2):
            str1, str2 = str2, str1
        
        if not str2:
            return 0
            
        previous_row = range(len(str2) + 1)
        for i, c1 in enumerate(str1):
            current_row = [i + 1]
            for j, c2 in enumerate(str2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        # Convert distance to similarity score
        max_len = max(len(str1), len(str2))
        similarity = 1 - (previous_row[-1] / max_len)
        return similarity

    @classmethod
    def generate_data_type_mapping(cls, df: pd.DataFrame) -> Dict[str, str]:
        """
        Generate data type mapping for DataFrame columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            dict: Mapping of column names to suggested data types
        """
        type_mapping = {}
        for column in df.columns:
            current_type = str(df[column].dtype)
            
            # Map pandas/numpy types to our type system
            if 'int' in current_type:
                type_mapping[column] = 'int64'
            elif 'float' in current_type:
                type_mapping[column] = 'float'
            elif 'datetime' in current_type:
                type_mapping[column] = 'datetime64[ns]'
            elif 'bool' in current_type:
                type_mapping[column] = 'bool'
            else:
                type_mapping[column] = 'string'
        
        return type_mapping

    @classmethod
    def apply_mapping(cls, df: pd.DataFrame, type_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Apply data type mapping to DataFrame.
        
        Args:
            df: Input DataFrame
            type_mapping: Dictionary mapping column names to desired data types
            
        Returns:
            pd.DataFrame: DataFrame with applied type conversions
        """
        try:
            df_copy = df.copy()
            for column, dtype in type_mapping.items():
                if column in df_copy.columns:
                    try:
                        if dtype == 'datetime64[ns]':
                            df_copy[column] = pd.to_datetime(df_copy[column], errors='coerce')
                        else:
                            df_copy[column] = df_copy[column].astype(dtype)
                    except Exception as e:
                        log_error(f"Error converting column {column} to type {dtype}: {str(e)}")
            return df_copy
        except Exception as e:
            log_error(f"Error applying type mapping: {str(e)}")
            return df

    @staticmethod
    def validate_mapping(source_df: pd.DataFrame, target_df: pd.DataFrame, 
                        column_mapping: Dict[str, str]) -> Tuple[bool, List[str]]:
        """
        Validate column mapping between source and target DataFrames.
        
        Args:
            source_df: Source DataFrame
            target_df: Target DataFrame
            column_mapping: Dictionary mapping source columns to target columns
            
        Returns:
            tuple: (is_valid, list of validation messages)
        """
        messages = []
        is_valid = True
        
        # Check if all mapped columns exist
        for source_col, target_col in column_mapping.items():
            if source_col not in source_df.columns:
                messages.append(f"Source column '{source_col}' not found")
                is_valid = False
            if target_col not in target_df.columns:
                messages.append(f"Target column '{target_col}' not found")
                is_valid = False
        
        # Check for duplicate mappings
        if len(set(column_mapping.values())) != len(column_mapping.values()):
            messages.append("Duplicate target columns in mapping")
            is_valid = False
        
        return is_valid, messages

    @staticmethod
    def get_unmapped_columns(source_df: pd.DataFrame, target_df: pd.DataFrame, 
                            column_mapping: Dict[str, str]) -> Tuple[List[str], List[str]]:
        """
        Get lists of unmapped columns from both source and target DataFrames.
        
        Args:
            source_df: Source DataFrame
            target_df: Target DataFrame
            column_mapping: Dictionary mapping source columns to target columns
            
        Returns:
            tuple: (unmapped source columns, unmapped target columns)
        """
        unmapped_source = [col for col in source_df.columns if col not in column_mapping]
        unmapped_target = [col for col in target_df.columns if col not in column_mapping.values()]
        return unmapped_source, unmapped_target

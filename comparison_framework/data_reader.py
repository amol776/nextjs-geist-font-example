import pandas as pd
import zipfile
import io
from typing import Union, Dict, Any
from utils import log_error, check_file_size

class DataReader:
    """Class to handle reading data from various file sources"""
    
    @staticmethod
    def load_csv(file: Union[str, io.BytesIO], delimiter: str = ',', **kwargs) -> Union[pd.DataFrame, None]:
        """
        Load data from CSV file.
        
        Args:
            file: File path or file object
            delimiter: CSV delimiter character
            **kwargs: Additional parameters for pd.read_csv
            
        Returns:
            pd.DataFrame or None: DataFrame containing the data, None if error occurs
        """
        try:
            if not check_file_size(file):
                log_error("File size exceeds 3GB limit")
                return None

            # Try to read with the specified delimiter and handle bad lines
            try:
                df = pd.read_csv(file, delimiter=delimiter, on_bad_lines='warn', encoding='utf-8', engine='python', **kwargs)
            except (pd.errors.ParserError, UnicodeDecodeError):
                # If that fails, try to detect the delimiter
                if hasattr(file, 'seek'):
                    file.seek(0)
                actual_delimiter = DataReader.infer_delimiter(file)
                if hasattr(file, 'seek'):
                    file.seek(0)
                df = pd.read_csv(file, delimiter=actual_delimiter, on_bad_lines='warn', encoding='utf-8', engine='python', **kwargs)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Handle any missing values
            df = df.fillna('')
            
            return df

        except Exception as e:
            log_error(f"Error reading CSV file: {str(e)}")
            return None

    @staticmethod
    def load_dat(file: Union[str, io.BytesIO], delimiter: str = '|', **kwargs) -> Union[pd.DataFrame, None]:
        """
        Load data from DAT file.
        
        Args:
            file: File path or file object
            delimiter: File delimiter character
            **kwargs: Additional parameters for pd.read_csv
            
        Returns:
            pd.DataFrame or None: DataFrame containing the data, None if error occurs
        """
        try:
            if not check_file_size(file):
                log_error("File size exceeds 3GB limit")
                return None
            
            # Try to read with the specified delimiter and handle bad lines
            try:
                df = pd.read_csv(file, delimiter=delimiter, on_bad_lines='warn', encoding='utf-8', engine='python', **kwargs)
            except (pd.errors.ParserError, UnicodeDecodeError):
                # If that fails, try to detect the delimiter
                if hasattr(file, 'seek'):
                    file.seek(0)
                actual_delimiter = DataReader.infer_delimiter(file)
                if hasattr(file, 'seek'):
                    file.seek(0)
                df = pd.read_csv(file, delimiter=actual_delimiter, on_bad_lines='warn', encoding='utf-8', engine='python', **kwargs)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Handle any missing values
            df = df.fillna('')
            
            return df

        except Exception as e:
            log_error(f"Error reading DAT file: {str(e)}")
            return None

    @staticmethod
    def load_parquet(file: Union[str, io.BytesIO]) -> Union[pd.DataFrame, None]:
        """
        Load data from Parquet file.
        
        Args:
            file: File path or file object
            
        Returns:
            pd.DataFrame or None: DataFrame containing the data, None if error occurs
        """
        try:
            if not check_file_size(file):
                log_error("File size exceeds 3GB limit")
                return None
            
            df = pd.read_parquet(file)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Handle any missing values
            df = df.fillna('')
            
            return df

        except Exception as e:
            log_error(f"Error reading Parquet file: {str(e)}")
            return None

    @staticmethod
    def load_zipped_flat_files(zip_file: Union[str, io.BytesIO], separator: str = ',') -> Union[pd.DataFrame, None]:
        """
        Load and combine flat files from a zip archive.
        
        Args:
            zip_file: Zip file path or file object
            separator: Delimiter used in flat files
            
        Returns:
            pd.DataFrame or None: Combined DataFrame containing data from all files, None if error occurs
        """
        try:
            if not check_file_size(zip_file):
                log_error("File size exceeds 3GB limit")
                return None

            dfs = []
            with zipfile.ZipFile(zip_file) as z:
                for filename in z.namelist():
                    with z.open(filename) as f:
                        try:
                            # Try to read with the specified delimiter
                            df = pd.read_csv(io.TextIOWrapper(f), delimiter=separator, on_bad_lines='warn', encoding='utf-8', engine='python')
                        except (pd.errors.ParserError, UnicodeDecodeError):
                            # If that fails, try to detect the delimiter
                            actual_delimiter = DataReader.infer_delimiter(io.TextIOWrapper(f))
                            f.seek(0)
                            df = pd.read_csv(io.TextIOWrapper(f), delimiter=actual_delimiter, on_bad_lines='warn', encoding='utf-8', engine='python')
                        
                        # Clean column names
                        df.columns = df.columns.str.strip()
                        
                        # Handle any missing values
                        df = df.fillna('')
                        
                        dfs.append(df)

            if not dfs:
                log_error("No valid files found in zip archive")
                return None

            # Combine all DataFrames
            return pd.concat(dfs, ignore_index=True)

        except Exception as e:
            log_error(f"Error processing zipped files: {str(e)}")
            return None

    @staticmethod
    def infer_delimiter(file: Union[str, io.BytesIO], sample_size: int = 1024) -> str:
        """
        Attempt to infer the delimiter used in a text file.
        
        Args:
            file: File path or file object
            sample_size: Number of bytes to sample for delimiter detection
            
        Returns:
            str: Inferred delimiter character
        """
        try:
            if hasattr(file, 'read'):
                sample = file.read(sample_size).decode('utf-8')
                file.seek(0)  # Reset file pointer
            else:
                with open(file, 'r') as f:
                    sample = f.read(sample_size)

            # Common delimiters to check
            delimiters = [',', '|', '\t', ';']
            counts = {d: sample.count(d) for d in delimiters}
            
            # Return the delimiter with highest count
            max_count = max(counts.values())
            if max_count > 0:
                return max(counts.items(), key=lambda x: x[1])[0]
            
            return ','  # Default to comma if no delimiter found

        except Exception as e:
            log_error(f"Error inferring delimiter: {str(e)}")
            return ','

    @staticmethod
    def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get basic information about the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            dict: Dictionary containing DataFrame information
        """
        return {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024)  # in MB
        }

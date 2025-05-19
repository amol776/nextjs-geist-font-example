import pandas as pd
import requests
import json
from typing import Dict, Optional, Union, Any
from utils import log_error
import io

class APIFetcher:
    """Class to handle API data fetching and processing"""

    @staticmethod
    def fetch_api_data(
        api_url: str,
        method: str = 'GET',
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        response_format: str = 'json',
        flatten: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from an API endpoint and convert to DataFrame.
        
        Args:
            api_url: URL of the API endpoint
            method: HTTP method (GET, POST, etc.)
            headers: Request headers
            params: Query parameters
            body: Request body for POST/PUT requests
            response_format: Expected response format ('json' or 'csv')
            flatten: Whether to flatten nested JSON structures
            
        Returns:
            Optional[pd.DataFrame]: DataFrame containing API response data, None if error occurs
        """
        try:
            # Set default headers if none provided
            if headers is None:
                headers = {'Content-Type': 'application/json'}

            # Make the API request
            response = requests.request(
                method=method.upper(),
                url=api_url,
                headers=headers,
                params=params,
                json=body if body else None,
                timeout=30  # 30 seconds timeout
            )

            # Check if request was successful
            response.raise_for_status()

            # Process response based on format
            if response_format.lower() == 'json':
                return APIFetcher._process_json_response(response.json(), flatten)
            elif response_format.lower() == 'csv':
                return APIFetcher._process_csv_response(response.content)
            else:
                log_error(f"Unsupported response format: {response_format}")
                return None

        except requests.exceptions.RequestException as e:
            log_error(f"API request failed: {str(e)}")
            return None
        except Exception as e:
            log_error(f"Error processing API response: {str(e)}")
            return None

    @staticmethod
    def _process_json_response(data: Union[Dict, list], flatten: bool) -> pd.DataFrame:
        """
        Process JSON response data into a DataFrame.
        
        Args:
            data: JSON response data
            flatten: Whether to flatten nested structures
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        # Handle list of dictionaries
        if isinstance(data, list):
            df = pd.json_normalize(data) if flatten else pd.DataFrame(data)
        # Handle single dictionary
        elif isinstance(data, dict):
            # Check if the response has a data/results key
            for key in ['data', 'results', 'items', 'records']:
                if key in data and isinstance(data[key], list):
                    df = pd.json_normalize(data[key]) if flatten else pd.DataFrame(data[key])
                    break
            else:
                # If no recognized key found, process the entire dictionary
                df = pd.json_normalize(data) if flatten else pd.DataFrame([data])
        else:
            raise ValueError("Unsupported JSON structure")

        return df

    @staticmethod
    def _process_csv_response(content: bytes) -> pd.DataFrame:
        """
        Process CSV response data into a DataFrame.
        
        Args:
            content: CSV content in bytes
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        return pd.read_csv(io.BytesIO(content))

    @staticmethod
    def validate_api_response(df: pd.DataFrame, expected_columns: Optional[list] = None) -> bool:
        """
        Validate API response DataFrame against expected schema.
        
        Args:
            df: DataFrame to validate
            expected_columns: List of expected column names
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            # Check if DataFrame is empty
            if df.empty:
                log_error("API response resulted in empty DataFrame")
                return False

            # Validate against expected columns if provided
            if expected_columns:
                missing_columns = set(expected_columns) - set(df.columns)
                if missing_columns:
                    log_error(f"Missing expected columns: {missing_columns}")
                    return False

            return True

        except Exception as e:
            log_error(f"Error validating API response: {str(e)}")
            return False

    @staticmethod
    def handle_pagination(
        api_url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        page_param: str = 'page',
        limit_param: str = 'limit',
        total_pages_key: Optional[str] = None,
        max_pages: int = 100
    ) -> Optional[pd.DataFrame]:
        """
        Handle paginated API responses.
        
        Args:
            api_url: Base API URL
            headers: Request headers
            params: Base query parameters
            page_param: Name of the pagination parameter
            limit_param: Name of the limit parameter
            total_pages_key: JSON key containing total pages info
            max_pages: Maximum number of pages to fetch
            
        Returns:
            Optional[pd.DataFrame]: Combined DataFrame from all pages, None if error occurs
        """
        try:
            all_data = []
            params = params or {}
            page = 1

            while page <= max_pages:
                # Update parameters with current page
                current_params = {**params, page_param: page}
                
                # Fetch current page
                df = APIFetcher.fetch_api_data(
                    api_url=api_url,
                    headers=headers,
                    params=current_params
                )

                if df is None or df.empty:
                    break

                all_data.append(df)

                # Check if we've reached the last page
                if len(df) == 0 or (total_pages_key and page >= total_pages_key):
                    break

                page += 1

            if not all_data:
                return None

            # Combine all pages
            return pd.concat(all_data, ignore_index=True)

        except Exception as e:
            log_error(f"Error handling pagination: {str(e)}")
            return None

    @staticmethod
    def retry_failed_request(
        api_url: str,
        max_retries: int = 3,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Retry failed API requests with exponential backoff.
        
        Args:
            api_url: API URL
            max_retries: Maximum number of retry attempts
            **kwargs: Additional arguments for fetch_api_data
            
        Returns:
            Optional[pd.DataFrame]: API response data, None if all retries fail
        """
        import time
        
        for attempt in range(max_retries):
            try:
                df = APIFetcher.fetch_api_data(api_url, **kwargs)
                if df is not None:
                    return df
                
            except Exception as e:
                if attempt == max_retries - 1:
                    log_error(f"All retry attempts failed: {str(e)}")
                    return None
                    
                # Exponential backoff
                wait_time = (2 ** attempt) + 1
                time.sleep(wait_time)
                continue

        return None

import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('comparison_framework.log')
    ]
)

logger = logging.getLogger(__name__)

def log_error(error_message: str) -> None:
    """
    Log error messages to both console and file.
    
    Args:
        error_message (str): The error message to log
    """
    logger.error(error_message)

def check_file_size(file) -> bool:
    """
    Check if file size exceeds 3GB threshold.
    
    Args:
        file: File object or path
        
    Returns:
        bool: True if file size is within limits, False otherwise
    """
    try:
        if hasattr(file, 'seek'):
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(0)  # Reset file pointer
        else:
            size = os.path.getsize(file)
        
        # 3GB in bytes
        GB_3 = 3 * 1024 * 1024 * 1024
        return size <= GB_3
    except Exception as e:
        log_error(f"Error checking file size: {str(e)}")
        return False

def format_timestamp() -> str:
    """
    Get formatted timestamp for file naming.
    
    Returns:
        str: Formatted timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def validate_connection_params(params: dict) -> bool:
    """
    Validate database connection parameters.
    
    Args:
        params (dict): Dictionary containing connection parameters
        
    Returns:
        bool: True if parameters are valid, False otherwise
    """
    # Basic required parameters for all connections
    basic_params = ['host', 'database']
    
    # Check if using Windows Authentication
    if params.get('use_windows_auth', True):
        # Only host and database are required for Windows Auth
        return all(param in params for param in basic_params)
    else:
        # Username and password required for SQL authentication
        sql_auth_params = basic_params + ['username', 'password']
        return all(param in params for param in sql_auth_params)

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Sanitized filename
    """
    # Replace invalid characters with underscore
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename.
    
    Args:
        filename (str): Name of the file
        
    Returns:
        str: File extension without dot
    """
    return os.path.splitext(filename)[1][1:].lower()

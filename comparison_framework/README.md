# Data Comparison Framework

A robust and flexible framework for comparing data from multiple sources with an intuitive Streamlit UI. This framework supports various data sources and provides comprehensive comparison reports.

## Features

- **Multiple Data Source Support:**
  - CSV files
  - DAT files
  - SQL Server databases
  - Stored Procedures
  - Teradata databases
  - REST APIs
  - Parquet files
  - Flat files inside zipped folders

- **Smart Column Mapping:**
  - Automatic column mapping suggestion
  - Manual mapping adjustment
  - Column exclusion options
  - Data type mapping support

- **Comprehensive Reporting:**
  1. Data Comparison Difference Report
  2. Y-Data Profiling Comparison Report
  3. Excel-based Regression Report including:
     - AggregationCheck (numeric column comparisons)
     - CountCheck (record count validation)
     - DistinctCheck (unique value analysis)
  4. Side-by-side Difference Report

- **Large File Support:**
  - Handles files up to 3GB
  - Efficient memory management
  - Progress indicators for long-running operations

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd comparison-framework
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Access the UI through your web browser (typically http://localhost:8501)

3. Select source and target data types:
   - Choose from available data source types
   - Configure connection details or upload files
   - Set delimiters for file-based sources

4. Review and adjust column mappings:
   - Verify automatic column mappings
   - Make manual adjustments if needed
   - Exclude columns from comparison
   - Configure data type mappings

5. Generate comparison reports:
   - Click "Compare Data" to generate all reports
   - Download reports in Excel format
   - Review results in the UI

## Project Structure

```
comparison_framework/
├── app.py                  # Main Streamlit application
├── data_reader.py          # Data source reading utilities
├── mapping_manager.py      # Column mapping functionality
├── report_generator.py     # Report generation logic
├── db_connector.py         # Database connection handlers
├── api_fetcher.py         # API interaction utilities
├── utils.py               # Common utility functions
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Configuration

### Database Connections

For database connections (SQL Server, Teradata), ensure you have:
- Proper network access to the database servers
- Required credentials and permissions
- Appropriate database drivers installed

### API Connections

For API connections:
- Ensure proper authentication details
- Valid API endpoints
- Required headers and parameters

## Report Types

### 1. Data Comparison Difference Report
- Shows exact differences between source and target data
- Highlights mismatched values
- Includes row indices for easy reference

### 2. Y-Data Profiling Report
- Statistical analysis of both datasets
- Column-level metrics
- Data quality indicators

### 3. Regression Report
- **AggregationCheck Tab:**
  - Numeric column comparisons
  - Sum comparisons
  - Pass/Fail indicators
- **CountCheck Tab:**
  - Record count validation
  - Source vs target counts
  - Pass/Fail status
- **DistinctCheck Tab:**
  - Unique value analysis
  - Distinct value counts
  - Value distribution comparison

### 4. Side-by-side Difference Report
- Detailed view of differences
- Source and target values side by side
- Excel format with color coding

## Best Practices

1. **File Size Management:**
   - For large files (>3GB), consider splitting into smaller chunks
   - Monitor system memory usage
   - Use appropriate data types to optimize memory

2. **Performance Optimization:**
   - Index key columns in databases
   - Use efficient SQL queries
   - Implement pagination for large API responses

3. **Error Handling:**
   - Check for data quality issues
   - Validate connection parameters
   - Monitor comparison progress

## Troubleshooting

Common issues and solutions:

1. **Memory Issues:**
   - Reduce chunk size for large files
   - Close unused connections
   - Clear session state when needed

2. **Connection Errors:**
   - Verify network connectivity
   - Check credentials
   - Confirm server availability

3. **Report Generation Issues:**
   - Ensure write permissions
   - Verify file paths
   - Check disk space

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

import argparse
import os
import pandas as pd
import boto3
# Use BytesIO to treat bytes as a file for Pandas
from io import BytesIO
import traceback # For better error reporting

# --- Configuration Variables ---
# MODIFY FOR EUROSTAT DATA: List all columns potentially present in the raw file
EXPECTED_COLUMNS = [
    'DATAFLOW', 'LAST UPDATE', 'freq', 'sex', 'age', 'unit', 'geo',
    'TIME_PERIOD', 'OBS_VALUE', 'OBS_FLAG', 'CONF_STATUS'
]

# MODIFY FOR EUROSTAT DATA: Set the name of the target variable column
TARGET_COLUMN = 'OBS_VALUE'

# MODIFY FOR EUROSTAT DATA: Define columns absolutely necessary for the *preprocessing* script downstream
# Should include Target + Sensitive (if used) + All Features used by preprocessor
# Based on the updated preprocess.py config:
MANDATORY_COLUMNS = [
    'OBS_VALUE',      # Target
    'sex',            # Sensitive Attribute
    'TIME_PERIOD',    # Numerical Feature
    'freq',           # Categorical Feature
    'age',            # Categorical Feature
    'unit',           # Categorical Feature
    'geo'             # Categorical Feature
]

# --- Keep these configuration variables ---
NULL_THRESHOLD_TARGET = 0.1 # Max allowed null proportion in target column
INPUT_DATA_FORMAT = 'csv'   # Keep as 'csv' unless input is different

# --- Encoding/Delimiter Configuration ---
INPUT_ENCODING = 'utf-8'    # DEFAULT: Common standard, change if data uses 'latin-1', 'iso-8859-1', etc.
INPUT_DELIMITER = ','       # DEFAULT: Common CSV delimiter, change if data uses ';', '\t', etc.
# --- End Configuration ---

def validate_data(bucket, key, endpoint_url, access_key, secret_key):
    print(f"--- Starting Validation ---")
    print(f"Input: s3://{bucket}/{key}")
    print(f"Format: {INPUT_DATA_FORMAT}, Encoding: {INPUT_ENCODING}, Delimiter: '{INPUT_DELIMITER}'")
    print(f"Expected Columns (subset): {EXPECTED_COLUMNS[:5]}...") # Print first few expected
    print(f"Mandatory Columns: {MANDATORY_COLUMNS}")
    print(f"Target Column: {TARGET_COLUMN} (Null Threshold: {NULL_THRESHOLD_TARGET:.1%})")

    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    try:
        print("Attempting to load data from S3...")
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        # Read raw bytes first
        raw_body = obj['Body'].read()

        # Let Pandas handle decoding using the configured encoding and delimiter
        if INPUT_DATA_FORMAT == 'csv':
            # Pass raw bytes wrapped in BytesIO and specify encoding/delimiter
            df = pd.read_csv(BytesIO(raw_body),
                             encoding=INPUT_ENCODING,
                             delimiter=INPUT_DELIMITER)

        # Add elif blocks here if supporting other formats like json, parquet
        # elif INPUT_DATA_FORMAT == 'parquet':
        #     df = pd.read_parquet(BytesIO(raw_body))
        else:
             raise ValueError(f"Unsupported INPUT_DATA_FORMAT: {INPUT_DATA_FORMAT}")

        print("File loaded successfully.")
        print(f"Shape: {df.shape}")
        print(f"Columns found ({len(df.columns)}): {df.columns.tolist()}")

        # --- Validation Logic using configured variables ---

        # 1. Check for MANDATORY columns
        print("Checking for mandatory columns...")
        missing_mandatory = [col for col in MANDATORY_COLUMNS if col not in df.columns]
        if missing_mandatory:
             # Provide more context in error message
             raise ValueError(f"CRITICAL: Missing mandatory columns required for preprocessing. Missing: {missing_mandatory}. Found columns: {df.columns.tolist()}")
        print("-> OK: All mandatory columns found.")

        # 2. Check for other EXPECTED columns (optional warning)
        print("Checking for other expected columns (warning if missing)...")
        missing_expected = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_expected:
            # This is just a warning, doesn't stop the process
            print(f"-> WARNING: Missing some expected (but not mandatory) columns: {missing_expected}")
        else:
            print("-> OK: All expected columns listed were found.")

        # 3. Check for excessive nulls in the TARGET column
        print(f"Checking null ratio in target column '{TARGET_COLUMN}'...")
        if TARGET_COLUMN in df.columns:
            if df.empty:
                 print("-> INFO: DataFrame is empty, skipping null check for target.")
                 null_ratio = 0.0
            else:
                 null_count = df[TARGET_COLUMN].isnull().sum()
                 total_count = len(df)
                 null_ratio = null_count / total_count
                 print(f"   Null count: {null_count}, Total rows: {total_count}")

            if null_ratio > NULL_THRESHOLD_TARGET:
                raise ValueError(f"CRITICAL: Excessive null values ({null_ratio:.2%}) found in target column '{TARGET_COLUMN}'. Threshold is {NULL_THRESHOLD_TARGET:.2%}")
            print(f"-> OK: Null ratio check for target '{TARGET_COLUMN}' passed ({null_ratio:.2%}).")
        else:
            # This case should ideally be caught by the MANDATORY_COLUMNS check if TARGET_COLUMN is mandatory
            print(f"-> WARNING: Target column '{TARGET_COLUMN}' not found in DataFrame, skipping null check. (Should have been caught by mandatory check if configured correctly).")

        print("--- Validation Checks Passed ---")

    except UnicodeDecodeError as e:
        print(f"--- VALIDATION FAILED: ENCODING ERROR ---")
        print(f"Error message: {e}")
        print(f"The file s3://{bucket}/{key} could not be decoded using '{INPUT_ENCODING}'.")
        print(f"--> ACTION REQUIRED: Verify the file's actual encoding. Update the INPUT_ENCODING variable in validate.py (and possibly preprocess.py) if it's different (e.g., 'latin-1', 'iso-8859-1', 'cp1252').")
        raise # Re-raise to fail the step
    except pd.errors.ParserError as e:
        print(f"--- VALIDATION FAILED: PARSING ERROR ---")
        print(f"Error message: {e}")
        print(f"Pandas encountered an error trying to parse the CSV file with delimiter '{INPUT_DELIMITER}'.")
        print(f"--> ACTION REQUIRED: Check if the delimiter is correct. Update INPUT_DELIMITER if needed. Also check for file corruption or rows with unexpected numbers of columns.")
        raise
    except FileNotFoundError: # More specific than generic Exception for S3 object not found
        print(f"--- VALIDATION FAILED: FILE NOT FOUND ---")
        print(f"The specified key 's3://{bucket}/{key}' was not found.")
        # boto3 usually raises ClientError for this, let's catch that too
        raise
    except boto3.exceptions.ClientError as e:
         error_code = e.response.get("Error", {}).get("Code")
         if error_code == 'NoSuchKey':
              print(f"--- VALIDATION FAILED: FILE NOT FOUND ---")
              print(f"The specified key 's3://{bucket}/{key}' was not found.")
         else:
              print(f"--- VALIDATION FAILED: AWS S3 CLIENT ERROR ---")
              print(f"{type(e).__name__}: {e}")
         raise
    except pd.errors.EmptyDataError:
        print(f"--- VALIDATION FAILED: EMPTY FILE ---")
        print(f"The file s3://{bucket}/{key} is empty or contains only headers.")
        raise
    except ValueError as e: # Catch specific ValueErrors raised by our checks
        print(f"--- VALIDATION FAILED: DATA CONTENT ISSUE ---")
        print(f"{e}") # Prints the detailed error message we created
        raise
    except Exception as e:
        print(f"--- VALIDATION FAILED: UNEXPECTED ERROR ---")
        print(f"An unexpected error occurred during validation.")
        print(f"{type(e).__name__}: {e}")
        traceback.print_exc() # Print full traceback
        raise

    print("--- Validation Finished Successfully ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, required=True, help='Input S3 bucket')
    parser.add_argument('--key', type=str, required=True, help='Input S3 key')
    args = parser.parse_args()

    s3_endpoint = os.environ.get("S3_ENDPOINT_URL")
    s3_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    s3_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    if not all([s3_endpoint, s3_access_key, s3_secret_key]):
        raise ValueError("S3 credentials (endpoint, access key, secret key) not found in environment variables.")

    validate_data(args.bucket, args.key, s3_endpoint, s3_access_key, s3_secret_key)
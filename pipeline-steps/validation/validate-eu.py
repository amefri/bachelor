import argparse
import os
import pandas as pd
import boto3
from io import BytesIO
import traceback

# --- Configuration for Eurostat data validation ---
EXPECTED_COLUMNS = [
    'DATAFLOW', 'LAST UPDATE', 'freq', 'sex', 'age', 'unit', 'geo',
    'TIME_PERIOD', 'OBS_VALUE', 'OBS_FLAG', 'CONF_STATUS'
]
TARGET_COLUMN = 'OBS_VALUE'
MANDATORY_COLUMNS = [
    'OBS_VALUE', 'sex', 'TIME_PERIOD', 'freq', 'age', 'unit', 'geo'
]

# --- General validation configurations ---
NULL_THRESHOLD_TARGET = 0.1
INPUT_DATA_FORMAT = 'csv'
INPUT_ENCODING = 'utf-8'
INPUT_DELIMITER = ','

# Main function to validate input data from S3.
def validate_data(bucket, key, endpoint_url, access_key, secret_key):
    print(f"--- Starting Validation ---")
    print(f"Input: s3://{bucket}/{key}")
    # Log configuration details for the validation run.
    print(f"Format: {INPUT_DATA_FORMAT}, Encoding: {INPUT_ENCODING}, Delimiter: '{INPUT_DELIMITER}'")
    print(f"Expected Columns (subset): {EXPECTED_COLUMNS[:5]}...")
    print(f"Mandatory Columns: {MANDATORY_COLUMNS}")
    print(f"Target Column: {TARGET_COLUMN} (Null Threshold: {NULL_THRESHOLD_TARGET:.1%})")

    # Initialize Minio.
    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    try:
        print("Attempting to load data from S3...")
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        raw_body = obj['Body'].read()

        # Load data based on the specified format, encoding, and delimiter.
        if INPUT_DATA_FORMAT == 'csv':
            df = pd.read_csv(BytesIO(raw_body),
                             encoding=INPUT_ENCODING,
                             delimiter=INPUT_DELIMITER)
        else:
             raise ValueError(f"Unsupported INPUT_DATA_FORMAT: {INPUT_DATA_FORMAT}")

        print("File loaded successfully.")
        print(f"Shape: {df.shape}")
        print(f"Columns found ({len(df.columns)}): {df.columns.tolist()}")

        # --- Data Validation Checks ---

        # 1. Check for presence of all mandatory columns.
        print("Checking for mandatory columns...")
        missing_mandatory = [col for col in MANDATORY_COLUMNS if col not in df.columns]
        if missing_mandatory:
             raise ValueError(f"CRITICAL: Missing mandatory columns required for preprocessing. Missing: {missing_mandatory}. Found columns: {df.columns.tolist()}")
        print("-> OK: All mandatory columns found.")

        # 2. Check for other expected columns (issues a warning if missing).
        print("Checking for other expected columns (warning if missing)...")
        missing_expected = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_expected:
            print(f"-> WARNING: Missing some expected (but not mandatory) columns: {missing_expected}")
        else:
            print("-> OK: All expected columns listed were found.")

        # 3. Check for excessive null values in the target column.
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
            print(f"-> WARNING: Target column '{TARGET_COLUMN}' not found in DataFrame, skipping null check.")

        print("--- Validation Checks Passed ---")

    # --- Error Handling for specific validation failures ---
    except UnicodeDecodeError as e:
        print(f"--- VALIDATION FAILED: ENCODING ERROR ---")
        print(f"Error message: {e}")
        print(f"The file s3://{bucket}/{key} could not be decoded using '{INPUT_ENCODING}'.")
        print(f"--> ACTION REQUIRED: Verify file encoding and update INPUT_ENCODING if necessary.")
        raise
    except pd.errors.ParserError as e:
        print(f"--- VALIDATION FAILED: PARSING ERROR ---")
        print(f"Error message: {e}")
        print(f"Pandas encountered an error parsing the CSV with delimiter '{INPUT_DELIMITER}'.")
        print(f"--> ACTION REQUIRED: Check delimiter and file structure.")
        raise
    except FileNotFoundError:
        print(f"--- VALIDATION FAILED: FILE NOT FOUND ---")
        print(f"The specified key 's3://{bucket}/{key}' was not found.")
        raise
    except boto3.exceptions.ClientError as e:
         error_code = e.response.get("Error", {}).get("Code")
         if error_code == 'NoSuchKey':
              print(f"--- VALIDATION FAILED: FILE NOT FOUND (S3 NoSuchKey) ---")
              print(f"The specified key 's3://{bucket}/{key}' was not found.")
         else:
              print(f"--- VALIDATION FAILED: AWS S3 CLIENT ERROR ---")
              print(f"{type(e).__name__}: {e}")
         raise
    except pd.errors.EmptyDataError:
        print(f"--- VALIDATION FAILED: EMPTY FILE ---")
        print(f"The file s3://{bucket}/{key} is empty or contains only headers.")
        raise
    except ValueError as e: # Catches custom ValueErrors from our checks.
        print(f"--- VALIDATION FAILED: DATA CONTENT ISSUE ---")
        print(f"{e}")
        raise
    except Exception as e: # Catches any other unexpected errors.
        print(f"--- VALIDATION FAILED: UNEXPECTED ERROR ---")
        print(f"An unexpected error occurred during validation.")
        print(f"{type(e).__name__}: {e}")
        traceback.print_exc()
        raise

    print("--- Validation Finished Successfully ---")

if __name__ == "__main__":
    # Define and parse command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, required=True, help='Input S3 bucket')
    parser.add_argument('--key', type=str, required=True, help='Input S3 key')
    args = parser.parse_args()

    # Retrieve S3 credentials from environment variables.
    s3_endpoint = os.environ.get("S3_ENDPOINT_URL")
    s3_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    s3_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    if not all([s3_endpoint, s3_access_key, s3_secret_key]):
        raise ValueError("S3 credentials (endpoint, access key, secret key) not found in environment variables.")

    # Run the validation function.
    validate_data(args.bucket, args.key, s3_endpoint, s3_access_key, s3_secret_key)
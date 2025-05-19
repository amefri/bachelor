import argparse
import os
import pandas as pd
import boto3
from io import BytesIO
import traceback 

# --- Configuration for AMS data validation ---
# Define all columns expected to be present in the raw input file.
EXPECTED_COLUMNS = ['Datum', 'RGSCode', 'RGSName', 'Geschlecht', 'Verweildauer', 'Altersgruppe', 'ABGANG', 'DS_VWD']
# Specify the name of the target variable column.
TARGET_COLUMN = 'ABGANG'
# List columns that are essential for downstream processing (features, sensitive attributes, target).
MANDATORY_COLUMNS = ['RGSCode', 'Geschlecht', 'Verweildauer', 'Altersgruppe', 'ABGANG', 'DS_VWD']

# --- General validation configurations ---
# Maximum allowable proportion of null values in the target column.
NULL_THRESHOLD_TARGET = 0.1
# Format of the input data file.
INPUT_DATA_FORMAT = 'csv'
# Character encoding of the input file (e.g., 'utf-8', 'latin1').
INPUT_ENCODING = 'latin1'
# Delimiter used in the CSV file (e.g., ',', ';', '\t').
INPUT_DEMLIMITER = ';'

# Main function to perform data validation.
def validate_data(bucket, key, endpoint_url, access_key, secret_key):
    print(f"Starting validation for s3://{bucket}/{key}")
    print(f"Attempting to read with encoding: {INPUT_ENCODING} and delimiter: '{INPUT_DEMLIMITER}'")

    # Initialize S3 client for accessing data.
    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    try:
        # Load data from S3.
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        raw_body = obj['Body'].read() # Read raw bytes first.

        # Parse data based on configured format, encoding, and delimiter.
        if INPUT_DATA_FORMAT == 'csv':
            df = pd.read_csv(BytesIO(raw_body), encoding=INPUT_ENCODING, delimiter=INPUT_DEMLIMITER)
        else:
             raise ValueError(f"Unsupported INPUT_DATA_FORMAT: {INPUT_DATA_FORMAT}")

        print("File loaded successfully.")
        print(f"Shape: {df.shape}")
        print(f"Columns found: {df.columns.tolist()}")

        # --- Begin Validation Logic ---

        # 1. Check for the presence of all mandatory columns.
        if not all(col in df.columns for col in MANDATORY_COLUMNS):
             missing_mandatory = set(MANDATORY_COLUMNS) - set(df.columns)
             raise ValueError(f"CRITICAL: Missing mandatory columns. Missing: {missing_mandatory}")
        print("-> OK: All mandatory columns found.")

        # 2. Check for other expected columns (issues a warning if missing).
        if not all(col in df.columns for col in EXPECTED_COLUMNS):
            missing_expected = set(EXPECTED_COLUMNS) - set(df.columns)
            print(f"-> WARNING: Missing some expected (but not mandatory) columns: {missing_expected}")
        else:
            print("-> OK: All expected columns found.")


        # 3. Check for excessive null values in the target column.
        if TARGET_COLUMN in df.columns:
            null_ratio = df[TARGET_COLUMN].isnull().sum() / len(df) if len(df) > 0 else 0
            if null_ratio > NULL_THRESHOLD_TARGET:
                raise ValueError(f"CRITICAL: Excessive null values ({null_ratio:.2%}) in target '{TARGET_COLUMN}'. Threshold: {NULL_THRESHOLD_TARGET:.2%}")
            print(f"-> OK: Null ratio for target '{TARGET_COLUMN}' passed ({null_ratio:.2%}).")
        else:
            # This should ideally be caught by the mandatory column check if target is mandatory.
            print(f"-> WARNING: Target column '{TARGET_COLUMN}' not found, skipping null check.")

        print("--- Validation Checks Passed ---")

    # --- Error Handling for specific validation failures ---
    except UnicodeDecodeError as e:
        print(f"--- VALIDATION FAILED: ENCODING ERROR ---")
        print(f"{e}")
        print(f"File s3://{bucket}/{key} likely not encoded as '{INPUT_ENCODING}'.")
        print(f"--> ACTION: Verify file encoding and update INPUT_ENCODING in this script if needed.")
        raise
    except FileNotFoundError: # Less common for S3 (ClientError is typical), but included.
        print(f"--- VALIDATION FAILED: FILE NOT FOUND ---")
        print(f"Key 's3://{bucket}/{key}' not found.")
        raise
    except boto3.exceptions.ClientError as e: # Catches S3 specific errors like NoSuchKey.
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
        print(f"File s3://{bucket}/{key} is empty or contains only headers.")
        raise
    except pd.errors.ParserError as e: # Handles issues with CSV structure, delimiters.
        print(f"--- VALIDATION FAILED: PARSING ERROR ---")
        print(f"Error parsing CSV file. Check delimiter ('{INPUT_DEMLIMITER}') and file structure.")
        print(f"{e}")
        raise
    except ValueError as e: # Catches custom ValueErrors from our checks.
        print(f"--- VALIDATION FAILED: DATA CONTENT ISSUE ---")
        print(f"{e}") # Prints the detailed error message from the raising check.
        raise
    except Exception as e: # Catches any other unexpected errors.
        print(f"--- VALIDATION FAILED: UNEXPECTED ERROR ---")
        print(f"{type(e).__name__}: {e}")
        traceback.print_exc()
        raise

    print("--- Validation Finished Successfully ---")

if __name__ == "__main__":
    # Define and parse command-line arguments.
    parser = argparse.ArgumentParser(description="Validate input data from S3.")
    parser.add_argument('--bucket', type=str, required=True, help='Input S3 bucket name.')
    parser.add_argument('--key', type=str, required=True, help='Input S3 key (path to file).')
    args = parser.parse_args()

    # Retrieve S3 credentials and endpoint from environment variables.
    s3_endpoint = os.environ.get("S3_ENDPOINT_URL")
    s3_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    s3_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    # Ensure S3 configuration is available.
    if not all([s3_endpoint, s3_access_key, s3_secret_key]):
        raise ValueError("S3 endpoint and/or credentials not found in environment variables.")

    # Execute the validation function.
    validate_data(args.bucket, args.key, s3_endpoint, s3_access_key, s3_secret_key)
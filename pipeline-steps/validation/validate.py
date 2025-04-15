import argparse
import os
import pandas as pd
import boto3
# Use BytesIO to treat bytes as a file for Pandas
from io import BytesIO

# --- Configuration Variables ---
# MODIFY FOR AMS DATA: List all columns expected in the raw CSV file
EXPECTED_COLUMNS = ['Datum', 'RGSCode', 'RGSName', 'Geschlecht', 'Verweildauer', 'Altersgruppe', 'ABGANG', 'DS_VWD']

# MODIFY FOR AMS DATA: Set the name of the target variable column
TARGET_COLUMN = 'ABGANG'

# MODIFY FOR AMS DATA: Define columns absolutely necessary for the pipeline downstream
# (Features to use + sensitive attribute + target)
MANDATORY_COLUMNS = ['RGSCode', 'Geschlecht', 'Verweildauer', 'Altersgruppe', 'ABGANG', 'DS_VWD']

# --- Keep these configuration variables ---
NULL_THRESHOLD_TARGET = 0.1 # Max allowed null proportion in target
INPUT_DATA_FORMAT = 'csv' # Keep as 'csv'

# --- Encoding Configuration ---

INPUT_ENCODING = 'latin1' # DEFAULT: Change if needed!

INPUT_DEMLIMITER = ';' # DEFAULT: Change if needed!

# --- End Configuration ---

def validate_data(bucket, key, endpoint_url, access_key, secret_key):
    print(f"Starting validation for s3://{bucket}/{key}")
    # Use configured encoding for logging
    print(f"Attempting to read with encoding: {INPUT_ENCODING}") # Logging added

    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        # Read raw bytes first
        raw_body = obj['Body'].read()

        # Let Pandas handle decoding using the configured encoding
        if INPUT_DATA_FORMAT == 'csv':
            # Pass raw bytes wrapped in BytesIO and specify encoding
            # Add other pd.read_csv options if needed (e.g., delimiter=';')
            df = pd.read_csv(BytesIO(raw_body), encoding=INPUT_ENCODING, delimiter=INPUT_DEMLIMITER)
            # Optional: Add delimiter if it's not comma (e.g., delimiter=';')
            # df = pd.read_csv(BytesIO(raw_body), encoding=INPUT_ENCODING, delimiter=';')
        # Add elif blocks here if supporting other formats like json, parquet
        # elif INPUT_DATA_FORMAT == 'parquet':
        #     df = pd.read_parquet(BytesIO(raw_body))
        else:
             raise ValueError(f"Unsupported INPUT_DATA_FORMAT: {INPUT_DATA_FORMAT}")

        print("File loaded successfully using specified encoding.")
        print(f"Shape: {df.shape}")
        print(f"Columns found: {df.columns.tolist()}")

        # --- Validation Logic using configured variables ---

        # Check for MANDATORY columns
        if not all(col in df.columns for col in MANDATORY_COLUMNS):
             missing_mandatory = set(MANDATORY_COLUMNS) - set(df.columns)
             raise ValueError(f"Missing mandatory columns defined in config. Missing: {missing_mandatory}")
        print("All mandatory columns found.")

        # Check for other EXPECTED columns (optional warning)
        if not all(col in df.columns for col in EXPECTED_COLUMNS):
            missing_expected = set(EXPECTED_COLUMNS) - set(df.columns)
            # This is just a warning, doesn't stop the process
            print(f"WARNING: Missing some expected columns defined in config: {missing_expected}")

        # Check for excessive nulls in the TARGET column
        if TARGET_COLUMN in df.columns:
            # Calculate null ratio safely, avoiding division by zero for empty dataframe
            null_ratio = df[TARGET_COLUMN].isnull().sum() / len(df) if len(df) > 0 else 0
            if null_ratio > NULL_THRESHOLD_TARGET:
                raise ValueError(f"Excessive null values ({null_ratio:.2%}) found in target column '{TARGET_COLUMN}'. Threshold: {NULL_THRESHOLD_TARGET:.2%}")
            print(f"Null ratio check for target '{TARGET_COLUMN}' passed ({null_ratio:.2%}).")
        else:
            # This case should ideally be caught by the MANDATORY_COLUMNS check if TARGET_COLUMN is mandatory
            print(f"WARNING: Target column '{TARGET_COLUMN}' not found, skipping null check.")

        print("Validation checks passed.")

    except UnicodeDecodeError as e: # Catch specific error
        print(f"--- VALIDATION FAILED: ENCODING ERROR ---")
        print(f"{e}")
        print(f"The file s3://{bucket}/{key} is likely NOT encoded as '{INPUT_ENCODING}'.")
        print("--> ACTION REQUIRED: Try changing the INPUT_ENCODING variable in validate.py to the correct encoding (e.g., 'latin-1', 'iso-8859-1', 'cp1252') and rebuild the step-validation Docker image.")
        raise # Re-raise to fail the step
    except FileNotFoundError as e: # Catch if key doesn't exist (less likely with S3 but good practice)
        print(f"--- VALIDATION FAILED: FILE NOT FOUND ---")
        print(f"The specified key 's3://{bucket}/{key}' likely does not exist.")
        raise
    except pd.errors.EmptyDataError:
        print(f"--- VALIDATION FAILED: EMPTY FILE ---")
        print(f"The file s3://{bucket}/{key} is empty.")
        raise
    # Add specific exceptions for other potential read errors if needed (e.g., pd.errors.ParserError)
    except Exception as e:
        print(f"--- VALIDATION FAILED: UNEXPECTED ERROR ---")
        print(f"{type(e).__name__}: {e}")
        # Include traceback for unexpected errors
        import traceback
        traceback.print_exc()
        raise

    print("Validation finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, required=True, help='Input S3 bucket')
    parser.add_argument('--key', type=str, required=True, help='Input S3 key')
    args = parser.parse_args()

    s3_endpoint = os.environ.get("S3_ENDPOINT_URL")
    s3_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    s3_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    if not all([s3_endpoint, s3_access_key, s3_secret_key]):
        raise ValueError("S3 credentials or endpoint not found in environment variables.")

    validate_data(args.bucket, args.key, s3_endpoint, s3_access_key, s3_secret_key)
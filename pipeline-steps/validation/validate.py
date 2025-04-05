# pipeline-steps/validation/validate.py

import argparse
import os
import pandas as pd
import boto3
# Use BytesIO to treat bytes as a file for Pandas
from io import BytesIO # CHANGED: Import BytesIO

# --- Configuration Variables ---
EXPECTED_COLUMNS = ['feature1', 'feature2', 'sensitive_attr', 'target'] # MODIFY FOR  DATA
TARGET_COLUMN = 'target'                                              # MODIFY FOR  DATA
MANDATORY_COLUMNS = ['feature1', 'target', 'sensitive_attr']          # MODIFY FOR  DATA (subset of EXPECTED)
NULL_THRESHOLD_TARGET = 0.1 # Max allowed null proportion in target
INPUT_DATA_FORMAT = 'csv' # 'csv', 'json', 'parquet', etc.

# >>> NEW CONFIGURATION VARIABLE <<<
INPUT_ENCODING = 'latin1'  # DEFAULT: Change this if  file uses a different encoding!
                          # Common alternatives: 'latin-1', 'iso-8859-1', 'cp1252'
# --- End Configuration ---

def validate_data(bucket, key, endpoint_url, access_key, secret_key):
    print(f"Starting validation for s3://{bucket}/{key}")
    # Use configured encoding for logging
    print(f"Attempting to read with encoding: {INPUT_ENCODING}") # ADDED logging

    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        # CHANGED: Read raw bytes first
        raw_body = obj['Body'].read()

        # CHANGED: Let Pandas handle decoding using the configured encoding
        if INPUT_DATA_FORMAT == 'csv':
            # Pass raw bytes wrapped in BytesIO and specify encoding
            df = pd.read_csv(BytesIO(raw_body), encoding=INPUT_ENCODING)
        # elif INPUT_DATA_FORMAT == 'json':
        #    # For JSON, manual decoding might still be needed depending on structure
        #    # Or use pd.read_json(BytesIO(raw_body), encoding=INPUT_ENCODING, ...)
        #     body_str = raw_body.decode(INPUT_ENCODING)
        #     df = pd.read_json(StringIO(body_str), lines=True) # Adjust params
        # elif INPUT_DATA_FORMAT == 'parquet':
        #     # Parquet read doesn't typically need explicit encoding arg for text within
        #     df = pd.read_parquet(BytesIO(raw_body))
        else:
             raise ValueError(f"Unsupported INPUT_DATA_FORMAT: {INPUT_DATA_FORMAT}")

        print("File loaded successfully using specified encoding.")
        print(f"Shape: {df.shape}")
        # ...(rest of the validation logic remains the same)...

        # Check for expected/mandatory columns
        if not all(col in df.columns for col in MANDATORY_COLUMNS):
             missing = set(MANDATORY_COLUMNS) - set(df.columns)
             raise ValueError(f"Missing mandatory columns (config). Missing: {missing}")

        if not all(col in df.columns for col in EXPECTED_COLUMNS):
            missing_expected = set(EXPECTED_COLUMNS) - set(df.columns)
            print(f"WARNING: Missing some expected columns (config): {missing_expected}")

        if TARGET_COLUMN in df.columns:
            null_ratio = df[TARGET_COLUMN].isnull().sum() / len(df) if len(df) > 0 else 0
            if null_ratio > NULL_THRESHOLD_TARGET:
                raise ValueError(f"Excessive null values ({null_ratio:.2%}) found in target column '{TARGET_COLUMN}'. Threshold: {NULL_THRESHOLD_TARGET:.2%}")
            print(f"Null ratio in target '{TARGET_COLUMN}': {null_ratio:.2%}")
        else:
            print(f"WARNING: Target column '{TARGET_COLUMN}' not found, skipping null check.")

        print("Validation checks passed.")

    except UnicodeDecodeError as e: # Catch specific error
        print(f"Validation FAILED: Encoding error - {e}")
        print(f"The file s3://{bucket}/{key} is likely NOT encoded as '{INPUT_ENCODING}'.")
        print("Try changing the INPUT_ENCODING variable in validate.py to 'latin-1', 'cp1252', or the correct encoding.")
        raise # Re-raise to fail the step
    except Exception as e:
        print(f"Validation FAILED: {e}")
        raise

    print("Validation finished successfully.")

if __name__ == "__main__":
    # ... (argument parsing and S3 credential loading remain the same) ...
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
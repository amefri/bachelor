import argparse
import os
import pandas as pd
import boto3
from io import StringIO


# Data Structure & Validation Rules
# Adjust these based on  dataset needs
EXPECTED_COLUMNS = ['Datum', 'RGSCode', 'Geschlecht', 'DS_VWD'] 
TARGET_COLUMN = 'DS_VWD'                                             
MANDATORY_COLUMNS = ['Datum', 'DS_VWD', 'Geschlecht']          
NULL_THRESHOLD_TARGET = 0.1 # Max allowed null proportion in target
INPUT_DATA_FORMAT = 'csv' # if not csv change pd.read !!!!

def validate_data(bucket, key, endpoint_url, access_key, secret_key):
    print(f"Starting validation for s3://{bucket}/{key}")
    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        body = obj['Body'].read().decode('utf-8')

        # Read data based on configured format
        if INPUT_DATA_FORMAT == 'csv':
            df = pd.read_csv(StringIO(body))
        # elif INPUT_DATA_FORMAT == 'json':
        #     df = pd.read_json(StringIO(body), lines=True) # Adjust read_json params as needed
        # elif INPUT_DATA_FORMAT == 'parquet':
        #     # Reading parquet from stream might need BytesIO
        #     from io import BytesIO
        #     df = pd.read_parquet(BytesIO(obj['Body'].read()))
        else:
             raise ValueError(f"Unsupported INPUT_DATA_FORMAT: {INPUT_DATA_FORMAT}")

        print("File loaded successfully.")
        print(f"Shape: {df.shape}")
        print(f"Columns found: {df.columns.tolist()}")
        print(f"Expected columns based on config: {EXPECTED_COLUMNS}")

        # --- Validation Logic using Configuration ---
        # Check for expected/mandatory columns
        if not all(col in df.columns for col in MANDATORY_COLUMNS):
             missing = set(MANDATORY_COLUMNS) - set(df.columns)
             raise ValueError(f"Missing mandatory columns (config). Missing: {missing}")

        # Check if all expected columns are present (optional, could be warning)
        if not all(col in df.columns for col in EXPECTED_COLUMNS):
            missing_expected = set(EXPECTED_COLUMNS) - set(df.columns)
            print(f"WARNING: Missing some expected columns (config): {missing_expected}")
            # Depending on strictness, you might raise ValueError here too

        # Check for excessive nulls in the configured target column
        if TARGET_COLUMN in df.columns:
            null_ratio = df[TARGET_COLUMN].isnull().sum() / len(df) if len(df) > 0 else 0
            if null_ratio > NULL_THRESHOLD_TARGET:
                raise ValueError(f"Excessive null values ({null_ratio:.2%}) found in target column '{TARGET_COLUMN}'. Threshold: {NULL_THRESHOLD_TARGET:.2%}")
            print(f"Null ratio in target '{TARGET_COLUMN}': {null_ratio:.2%}")
        else:
            print(f"WARNING: Target column '{TARGET_COLUMN}' not found, skipping null check.")

        # --- Add more checks using config variables ---

        print("Validation checks passed.")

    except Exception as e:
        print(f"Validation FAILED: {e}")
        raise

    print("Validation finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, required=True, help='Input S3 bucket')
    parser.add_argument('--key', type=str, required=True, help='Input S3 key')
    # No need to pass column names etc. via args if using config variables above
    args = parser.parse_args()

    s3_endpoint = os.environ.get("S3_ENDPOINT_URL")
    s3_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    s3_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    if not all([s3_endpoint, s3_access_key, s3_secret_key]):
        raise ValueError("S3 credentials or endpoint not found in environment variables.")

    validate_data(args.bucket, args.key, s3_endpoint, s3_access_key, s3_secret_key)
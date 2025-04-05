import argparse
import os
import pandas as pd
import boto3
from io import StringIO




def validate_data(bucket, key, endpoint_url, access_key, secret_key):
    print(f"Validation start for s3://{bucket}/{key}")
    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )
    print(f"erfolgreich")
    
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        body = obj['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(body))

        # --- Basic Validation Logic ---
        print("File loaded successfully.")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Example: Check for expected columns (replace with your schema)
        expected_cols = [feature1, feature2, sensitive_attr, target]
        if not all(col in df.columns for col in expected_cols):
             raise ValueError(f"Missing expected columns. Found: {df.columns.tolist()}, Expected: {expected_cols}")

        # Example: Check for excessive nulls in target
        if df['target'].isnull().sum() > len(df) * 0.1: # More than 10% nulls
            raise ValueError("Excessive null values found in 'target' column.")

        print("Validation checks passed.")
        # In a real scenario, might output a status file or just rely on logs/exit code

    except Exception as e:
        print(f"Validation FAILED: {e}")
        raise # Re-raise exception to fail the Argo step

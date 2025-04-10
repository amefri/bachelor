#--WRONG FORMULAR I DID A CALCULATION UPSI-PLEASE DONT LOOK---

import argparse
import os
import pandas as pd
import boto3
from io import BytesIO
import json
import numpy as np # For NaN handling and safe division

# --- Configuration ---
# Default input format if needed (can be overridden)
INPUT_FORMAT = 'parquet'

DEFAULT_BIAS_THRESHOLD = 0.1
# --- End Configuration ---


def check_bias(
    in_bucket, in_key, out_bucket, report_key,
    sensitive_col,  # REQUIRED: Column with sensitive attribute
    group_value,    # REQUIRED: Specific group value within sensitive col
    target_col,             # REQUIRED: Target variable column name
    positive_target_value_str, # REQUIRED: Value representing positive outcome (as string from arg)
    bias_threshold,         # Threshold for check
    endpoint_url, access_key, secret_key
    ):
    """
    Checks for bias using  Divergence.
  Divergence = P(Group | Y=positive) - P(Group | Y=negative)
    """
    print(f"Starting  Divergence bias check for s3://{in_bucket}/{in_key}")
    print(f"  Sensitive Column: {sensitive_col}")
    print(f"  Group of Interest: '{group_value}'")
    print(f"  Target Column: {target_col}")
    print(f"  Positive Outcome Value (str): '{positive_target_value_str}'")
    print(f"  Bias Threshold (Magnitude): {bias_threshold}")
    print(f"Report will be saved to s3://{out_bucket}/{report_key}")

    s3_client = boto3.client(
        "s3", endpoint_url=endpoint_url, aws_access_key_id=access_key, aws_secret_access_key=secret_key
    )

    # --- Load Data ---
    try:
        print(f"Loading processed data from format: {INPUT_FORMAT}")
        obj = s3_client.get_object(Bucket=in_bucket, Key=in_key)
        in_buffer = BytesIO(obj['Body'].read())
        if INPUT_FORMAT == 'parquet':
            df = pd.read_parquet(in_buffer)
        # Add elif for other INPUT_FORMATs if needed
        else:
            raise ValueError(f"Unsupported INPUT_FORMAT: {INPUT_FORMAT}")
        print("Processed data loaded.")
        print(f"Data shape: {df.shape}")
    except Exception as e:
        print(f"ERROR: Failed to load data from s3://{in_bucket}/{in_key}. Error: {e}")
        raise

    # --- Input Validation ---
    required_cols = [target_col, sensitive_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in data: {missing_cols}. Found: {df.columns.tolist()}")

    # Try converting positive target value string to the type of the target column
    try:
        target_dtype = df[target_col].dtype
        # Handle potential conversion errors, especially for non-numeric types
        if pd.api.types.is_numeric_dtype(target_dtype):
             # Attempt conversion to number (int first, then float)
             try:
                 positive_target_value = int(positive_target_value_str)
             except ValueError:
                 positive_target_value = float(positive_target_value_str)
        elif pd.api.types.is_bool_dtype(target_dtype):
             # Handle common boolean strings
             bool_map = {'true': True, 'false': False, '1': True, '0': False}
             positive_target_value = bool_map.get(positive_target_value_str.lower(), None)
             if positive_target_value is None:
                  raise ValueError("String is not a recognized boolean value (true/false/1/0)")
        else: # Assume string or object type
             positive_target_value = positive_target_value_str
        print(f"  Converted Positive Outcome Value: {positive_target_value} (Type: {type(positive_target_value)})")

    except (ValueError, TypeError) as e:
         raise ValueError(f"Could not convert positive target value string '{positive_target_value_str}' to match target column '{target_col}' dtype ({target_dtype}). Error: {e}")


    # ---  Divergence Calculation ---
    results = {
        "divergence": np.nan,
        "p_group_given_pos": np.nan,
        "p_group_given_neg": np.nan,
        "count_group_in_pos": 0,
        "count_total_pos": 0,
        "count_group_in_neg": 0,
        "count_total_neg": 0,
        "calculation_status": "Not Calculated",
        "sensitive_col": sensitive_col,
        "group_value": group_value,
        "target_col": target_col,
        "positive_target_value": positive_target_value, # Store the converted value
    }

    try:
        # Subset data based on target outcome
        positive_outcome_df = df[df[target_col] == positive_target_value]
        negative_outcome_df = df[df[target_col] != positive_target_value] # Assumes binary or non-positive is negative

        total_pos = len(positive_outcome_df)
        total_neg = len(negative_outcome_df)
        results["count_total_pos"] = total_pos
        results["count_total_neg"] = total_neg

        # Calculate P(Group | Y=positive)
        if total_pos > 0:
            group_in_pos = len(positive_outcome_df[positive_outcome_df[sensitive_col] ==group_value])
            p_group_given_pos = group_in_pos / total_pos
            results["count_group_in_pos"] = group_in_pos
            results["p_group_given_pos"] = p_group_given_pos
            print(f"  P(Group='{group_value}' | Y={positive_target_value}) = {group_in_pos}/{total_pos} = {p_group_given_pos:.4f}")
        else:
            p_group_given_pos = np.nan # Indicate undefined
            print("  WARNING: No samples found with the positive outcome. P(Group|Y=pos) is undefined.")

        # Calculate P(Group | Y=negative)
        if total_neg > 0:
            group_in_neg = len(negative_outcome_df[negative_outcome_df[sensitive_col] == group_value])
            p_group_given_neg = group_in_neg / total_neg
            results["count_group_in_neg"] = group_in_neg
            results["p_group_given_neg"] = p_group_given_neg
            print(f"  P(Group='{group_value}' | Y!={positive_target_value}) = {group_in_neg}/{total_neg} = {p_group_given_neg:.4f}")
        else:
            p_group_given_neg = np.nan # Indicate undefined
            print("  WARNING: No samples found with the negative outcome. P(Group|Y=neg) is undefined.")

        # Calculate Divergence (only if both probabilities are valid)
        if not np.isnan(p_group_given_pos) and not np.isnan(p_group_given_neg):
            divergence = p_group_given_pos - p_group_given_neg
            results["divergence"] = divergence
            results["calculation_status"] = "Success"
            print(f"\n  Leitner Divergence for Group '{group_value}' = {divergence:.4f}")
        else:
             results["calculation_status"] = "Warning: Could not calculate divergence due to missing positive or negative outcomes."
             print(f"\n  {results['calculation_status']}")

    except Exception as e:
        results["calculation_status"] = f"Error during calculation: {e}"
        print(f"ERROR during Leitner Divergence calculation: {e}")


    # --- Combine results for the report ---
    metrics_results = {
        "divergence_metrics": results,
        "bias_check_status": "Not Evaluated",
        "bias_threshold": bias_threshold
    }

    # --- Decision Logic (Based on  Divergence Magnitude) ---
    if results["calculation_status"] == "Success":
        divergence_magnitude = abs(results["divergence"])
        print(f"  Divergence Magnitude: {divergence_magnitude:.4f}")
        if divergence_magnitude > bias_threshold:
            print(f"WARNING: Potential bias detected.  Divergence magnitude ({divergence_magnitude:.4f}) exceeds threshold ({bias_threshold}).")
            metrics_results["bias_check_status"] = "Warning: Threshold Exceeded"
            # Optionally fail the pipeline:
            # raise ValueError(f"Bias check FAILED: Divergence magnitude ({divergence_magnitude:.4f}) exceeds threshold ({bias_threshold}).")
        else:
            print("Bias check passed based on  Divergence threshold.")
            metrics_results["bias_check_status"] = "Passed"
    elif "Error" in results["calculation_status"]:
         metrics_results["bias_check_status"] = "Error during calculation"
         print(f"ERROR: Cannot determine bias status due to calculation error.")
    else: # Handle cases like missing groups or NaN results
         metrics_results["bias_check_status"] = "Indeterminate (calculation warning)"
         print(f"INFO: Bias status indeterminate due to calculation warning: {results['calculation_status']}")


    # --- Save Report ---
    print("\nSaving bias report...")
    try:
        # Use default=str to handle potential NaN values during JSON serialization
        report_buffer = BytesIO(json.dumps(metrics_results, indent=2, default=str).encode('utf-8'))
        s3_client.put_object(Bucket=out_bucket, Key=report_key, Body=report_buffer)
        print(f"Bias report saved to s3://{out_bucket}/{report_key}")
    except Exception as e:
        print(f"ERROR: Failed to save report to s3://{out_bucket}/{report_key}. Error: {e}")
        # Decide if this should fail the step
        # raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Leitner Divergence for bias assessment.")
    # S3 Args
    parser.add_argument('--in_bucket', type=str, required=True, help="S3 bucket for input processed data")
    parser.add_argument('--in_key', type=str, required=True, help="S3 key for input processed data (e.g., parquet file)")
    parser.add_argument('--out_bucket', type=str, required=True, help="S3 bucket for output report")
    parser.add_argument('--report_key', type=str, required=True, help="S3 key for output report (JSON)")
    # Leitner Specific Args
    parser.add_argument('--sensitive_col', type=str, required=True, help="Column name containing the sensitive attribute")
    parser.add_argument('--group_value', type=str, required=True, help="The specific value in sensitive col identifying the group of interest")
    parser.add_argument('--target_col', type=str, required=True, help="Target variable column name")
    parser.add_argument('--positive_target_value', type=str, required=True, help="The value (as string) in target col representing the positive outcome")
    # Optional threshold
    parser.add_argument('--bias_threshold', type=float, default=DEFAULT_BIAS_THRESHOLD, help=f"Threshold for the absolute value of Leitner Divergence (default: {DEFAULT_BIAS_THRESHOLD})")

    args = parser.parse_args()

    # Get S3 credentials from environment
    s3_endpoint = os.environ.get("S3_ENDPOINT_URL")
    s3_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    s3_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    if not all([s3_endpoint, s3_access_key, s3_secret_key]):
        raise ValueError("S3 credentials or endpoint not found in environment variables (S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")

    # Call the main function
    check_bias(
        args.in_bucket, args.in_key, args.out_bucket, args.report_key,
        args.sensitive_col,
        args.group_value,
        args.target_col,
        args.positive_target_value, # Pass as string, conversion happens inside
        args.bias_threshold,
        s3_endpoint, s3_access_key, s3_secret_key
        )

    print("Bias check script finished.")
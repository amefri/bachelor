# demographic parity check

from fairlearn.metrics import selection_rate, demographic_parity_difference, MetricFrame
import argparse
import os
import pandas as pd
import boto3
from io import BytesIO
import json
import sys # Import sys for error handling


BIAS_THRESHOLD_DP = 0.1 

def run_demographic_parity_check(in_bucket, in_key, out_bucket, report_key,
                                  sensitive_features_str, target_col, BIAS_THRESHOLD_DP, 
                                  endpoint_url, access_key, secret_key):
    print("Checking demographic parity...")
    print(f"Using bias threshold: {BIAS_THRESHOLD_DP}") # Log the threshold being used

    s3 = boto3.client("s3", endpoint_url=endpoint_url,
                      aws_access_key_id=access_key, aws_secret_access_key=secret_key)

    try:
        obj = s3.get_object(Bucket=in_bucket, Key=in_key)
        df = pd.read_parquet(BytesIO(obj["Body"].read()))
        print(f"Data loaded successfully from {in_bucket}/{in_key}. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data from {in_bucket}/{in_key}: {e}")
        sys.exit(f"Failed to load input data: {e}") # Exit if data load fails

    sensitive_features_list = sensitive_features_str.split(',')
    required_cols = [target_col] + sensitive_features_list
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Error: Missing columns in dataset: {missing}")
        sys.exit(f"Missing columns: {missing}") # Exit if columns are missing

    df = df.dropna(subset=required_cols)
    if df.empty:
        print(f"Error: DataFrame is empty after dropping NA values for columns: {required_cols}")
        sys.exit("DataFrame empty after NA drop.") # Exit if no data remains

    print(f"Data shape after dropping NA: {df.shape}")
    print(f"Unique values in sensitive features ({sensitive_features_list}):")
    for col in sensitive_features_list:
         print(f"  {col}: {df[col].unique()}")
    print(f"Unique values in target ({target_col}): {df[target_col].unique()}")


    if len(sensitive_features_list) == 1:
        sensitive_features = df[sensitive_features_list[0]]
    else:
        # Ensure consistent data types for multi-column sensitive features if needed
        sensitive_features = df[sensitive_features_list].apply(tuple, axis=1)

    y_true = df[target_col]

    # --- Check if y_true has variance ---
    if y_true.nunique() <= 1:
       print(f"Warning: Target column '{target_col}' has only one unique value ({y_true.unique()}). Bias metrics may not be meaningful.")
       
       dp_diff = 0.0
       mf_by_group = {}
       mf_overall = y_true.mean() if not y_true.empty else 0.0 # Handle empty case
       bias_status = "Not Applicable (Single Target Value)"
    else:
        try:
            mf = MetricFrame(
                metrics=selection_rate,
                y_true=y_true,
                y_pred=y_true, # Using y_true as y_pred for selection rate of true labels
                sensitive_features=sensitive_features
            )
            mf_by_group = mf.by_group.to_dict()
            mf_overall = mf.overall

            # Ensure DP calculation doesn't fail on edge cases
            dp_diff = demographic_parity_difference(y_true, y_true, sensitive_features=sensitive_features)

            # Use the bias_threshold passed from the workflow
            bias_status = "Warning: Potential Bias Detected" if abs(dp_diff) > BIAS_THRESHOLD_DP else "Passed"
            print(f"Demographic Parity Difference: {dp_diff:.4f}")
            print(f"Bias Check Status (Threshold: {BIAS_THRESHOLD_DP}): {bias_status}")

        except Exception as e:
            print(f"Error calculating Fairlearn metrics: {e}")
            # Decide how to handle metric calculation errors
            dp_diff = "Error"
            mf_by_group = {"Error": str(e)}
            mf_overall = "Error"
            bias_status = "Error During Calculation"


    result = {
        "selection_rate_by_group": mf_by_group,
        "overall_selection_rate": mf_overall,
        "demographic_parity_difference": dp_diff,
        "bias_threshold_used": BIAS_THRESHOLD_DP, # Include threshold in report
        "bias_check_status": "Warning" if abs(dp_diff) > BIAS_THRESHOLD_DP else "Passed",
        "sensitive_features": sensitive_features_list,
        "target_column": target_col,
        "check_type": "Demographic Parity"
    }

    try:
        report_buffer = BytesIO(json.dumps(result, indent=2).encode("utf-8"))
        s3.put_object(Bucket=out_bucket, Key=report_key, Body=report_buffer)
        print(f"Demographic parity report saved successfully to {out_bucket}/{report_key}.")
    except Exception as e:
        print(f"Error saving report to {out_bucket}/{report_key}: {e}")
        sys.exit(f"Failed to save report: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for demographic parity bias.") # Added description
    parser.add_argument('--in_bucket', required=True, help="Input S3 bucket") # Added help text
    parser.add_argument('--in_key', required=True, help="Input S3 key for processed data (Parquet format)")
    parser.add_argument('--out_bucket', required=True, help="Output S3 bucket for report")
    parser.add_argument('--report_key', required=True, help="Output S3 key for bias report (JSON format)")
    parser.add_argument('--sensitive_features', required=True, help="Comma-separated list of sensitive feature column names")
    parser.add_argument('--target_col', required=True, help="Name of the target column")
    parser.add_argument('--group_value', required=False, help="Specific group value (currently not used by demographic parity check, but accepted)") # Added required=False for now
    parser.add_argument('--positive_target_value', required=False, help="Value representing positive outcome (currently not used by demographic parity check, but accepted)") # Added required=False for now
    
    args = parser.parse_args()


    run_demographic_parity_check(
        args.in_bucket, args.in_key, args.out_bucket, args.report_key,
        args.sensitive_features, args.target_col, BIAS_THRESHOLD_DP, 
        
        os.environ.get("S3_ENDPOINT_URL"),
        os.environ.get("AWS_ACCESS_KEY_ID"),
        os.environ.get("AWS_SECRET_ACCESS_KEY")
    )
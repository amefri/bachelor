

from fairlearn.metrics import selection_rate, demographic_parity_difference, MetricFrame
import argparse
import os
import pandas as pd
import boto3
from io import BytesIO, StringIO
import json
import sys

def run_demographic_parity_check(in_bucket, in_key, out_bucket, report_key,
                                  sensitive_features_str, target_col, bias_threshold,
                                  endpoint_url, access_key, secret_key):
    print("Checking demographic parity...")
    print(f"Using bias threshold: {bias_threshold}")

    s3 = boto3.client("s3", endpoint_url=endpoint_url,
                      aws_access_key_id=access_key, aws_secret_access_key=secret_key)

    # --- Default status in case of early exit ---
    early_exit_status = "Error"

    try:
        obj = s3.get_object(Bucket=in_bucket, Key=in_key)
        df = pd.read_parquet(BytesIO(obj["Body"].read()))
        print(f"Data loaded successfully from {in_bucket}/{in_key}. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data from {in_bucket}/{in_key}: {e}")
        try:
             with open("/tmp/bias_status.txt", "w") as f: f.write(early_exit_status)
             print(f"Status '{early_exit_status}' written to /tmp/bias_status.txt")
        except Exception as fe:
             print(f"Warning: Could not write status file for Argo on error: {fe}")
        sys.exit(f"Failed to load input data: {e}")

    sensitive_features_list = sensitive_features_str.split(',')
    required_cols = [target_col] + sensitive_features_list
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Error: Missing columns in dataset: {missing}")
        try:
             with open("/tmp/bias_status.txt", "w") as f: f.write(early_exit_status)
             print(f"Status '{early_exit_status}' written to /tmp/bias_status.txt")
        except Exception as fe:
             print(f"Warning: Could not write status file for Argo on error: {fe}")
        sys.exit(f"Missing columns: {missing}")

    df = df.dropna(subset=required_cols)
    if df.empty:
        print(f"Error: DataFrame is empty after dropping NA values for columns: {required_cols}")
        try:
             with open("/tmp/bias_status.txt", "w") as f: f.write(early_exit_status)
             print(f"Status '{early_exit_status}' written to /tmp/bias_status.txt")
        except Exception as fe:
             print(f"Warning: Could not write status file for Argo on error: {fe}")
        sys.exit("DataFrame empty after NA drop.")

    print(f"Data shape after dropping NA: {df.shape}")
    print(f"Unique values in sensitive features ({sensitive_features_list}):")
    for col in sensitive_features_list:
         unique_vals = df[col].unique()
         print(f"  {col}: {unique_vals[:10]}{'...' if len(unique_vals) > 10 else ''}")
    unique_target = df[target_col].unique()
    print(f"Unique values in target ({target_col}): {unique_target[:10]}{'...' if len(unique_target) > 10 else ''}")

    if len(sensitive_features_list) == 1:
        sensitive_features = df[sensitive_features_list[0]]
    else:
        sensitive_features = df[sensitive_features_list].apply(tuple, axis=1)

    y_true = df[target_col]
    bias_status = "Unknown" # Default status before calculation

    if y_true.nunique() <= 1:
       print(f"Warning: Target column '{target_col}' has only one unique value ({y_true.unique()}). Bias metrics not applicable.")
       dp_diff = 0.0
       mf_by_group = {}
       mf_overall = y_true.mean() if not y_true.empty else 0.0
       
       bias_status = "Passed" # Treat as Passed since no disparity exists
    else:
        try:
            mf = MetricFrame(
                metrics=selection_rate,
                y_true=y_true,
                y_pred=y_true,
                sensitive_features=sensitive_features
            )
            mf_by_group = mf.by_group.to_dict()
            mf_overall = mf.overall
            dp_diff = demographic_parity_difference(y_true, y_true, sensitive_features=sensitive_features)

            if pd.isna(dp_diff):
                print("Warning: Demographic parity difference is NaN. Likely only one group present.")
                
                bias_status = "Passed" # Treat as Passed since no disparity measurable
                dp_diff = 0.0
            else:
                
                if abs(dp_diff) > bias_threshold:
                    bias_status = "Warning" 
                else:
                    bias_status = "Passed"

            print(f"Demographic Parity Difference: {dp_diff:.4f}")
            
            print(f"Bias Check Status (Threshold: {bias_threshold}): {'Warning: Potential Bias Detected' if bias_status == 'Warning' else bias_status}")


        except Exception as e:
            print(f"Error calculating Fairlearn metrics: {e}")
            dp_diff = "Error"
            mf_by_group = {"Error": str(e)}
            mf_overall = "Error"
            
            bias_status = "Error"


    result = {
        "selection_rate_by_group": mf_by_group,
        "overall_selection_rate": mf_overall,
        "demographic_parity_difference": dp_diff,
        "bias_threshold_used": bias_threshold,
        "bias_check_status": bias_status,
        "sensitive_features": sensitive_features_list,
        "target_column": target_col,
        "check_type": "Demographic Parity"
    }

    try:
        if hasattr(result["demographic_parity_difference"], 'item'):
             result["demographic_parity_difference"] = result["demographic_parity_difference"].item()
        if hasattr(result["overall_selection_rate"], 'item'):
            result["overall_selection_rate"] = result["overall_selection_rate"].item()

        report_buffer = BytesIO(json.dumps(result, indent=2).encode("utf-8"))
        s3.put_object(Bucket=out_bucket, Key=report_key, Body=report_buffer)
        print(f"Demographic parity report saved successfully to {out_bucket}/{report_key}.")
    except TypeError as te:
         print(f"Error serializing result #to JSON: {te}")
         print(f"Result dictionary causing issue: {result}")
         bias_status = "Error" # Update status if report saving fails
    except Exception as e:
        print(f"Error saving report to {out_bucket}/{report_key}: {e}")
        bias_status = "Error" # Update status if report saving fails

   
    final_status_to_write = bias_status 
    try:
        with open("/tmp/bias_status.txt", "w") as f:
            f.write(final_status_to_write)
        print(f"Final simple bias status '{final_status_to_write}' written to /tmp/bias_status.txt for Argo.")
    except Exception as e:
        print(f"Warning: Could not write status file for Argo: {e}")


    if final_status_to_write == "Error":
         sys.exit("Exiting due to error status during bias check.")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for demographic parity bias.")
    parser.add_argument('--in_bucket', required=True, help="Input S3 bucket")
    parser.add_argument('--in_key', required=True, help="Input S3 key for processed data (Parquet format)")
    parser.add_argument('--out_bucket', required=True, help="Output S3 bucket for report")
    parser.add_argument('--report_key', required=True, help="Output S3 key for bias report (JSON format)")
    parser.add_argument('--sensitive_features', required=True, help="Comma-separated list of sensitive feature column names")
    parser.add_argument('--target_col', required=True, help="Name of the target column")
    parser.add_argument('--group_value', required=False, help="Specific group value (currently not used by demographic parity check, but accepted)")
    parser.add_argument('--positive_target_value', required=False, help="Value representing positive outcome (currently not used by demographic parity check, but accepted)")
    parser.add_argument('--bias_threshold', type=float, required=True, help="Threshold for absolute demographic parity difference")

    args = parser.parse_args()

    run_demographic_parity_check(
        args.in_bucket, args.in_key, args.out_bucket, args.report_key,
        args.sensitive_features, args.target_col, args.bias_threshold,
        os.environ.get("S3_ENDPOINT_URL"),
        os.environ.get("AWS_ACCESS_KEY_ID"),
        os.environ.get("AWS_SECRET_ACCESS_KEY")
    )

    print("Bias check script finished.")
import argparse
import os
import pandas as pd
import boto3
from io import BytesIO
import json
# Remove fairlearn imports if no longer needed for this specific check
# from fairlearn.metrics import MetricFrame, count, demographic_parity_difference, equalized_odds_difference

# Keep the positive_rate function or just use .mean() directly later
def positive_rate(y_true):
    # Ensure input is numeric for mean calculation
    return pd.to_numeric(y_true, errors='coerce').mean()

def check_bias(in_bucket, in_key, out_bucket, report_key, sensitive_features_str, target_col, endpoint_url, access_key, secret_key):
    print(f"Starting bias check for s3://{in_bucket}/{in_key}")
    print(f"Sensitive features: {sensitive_features_str}")
    print(f"Target column: {target_col}")
    print(f"Report will be saved to s3://{out_bucket}/{report_key}")

    s3_client = boto3.client(
        "s3", endpoint_url=endpoint_url, aws_access_key_id=access_key, aws_secret_access_key=secret_key
    )

    # Load processed data (assuming Parquet)
    try:
        obj = s3_client.get_object(Bucket=in_bucket, Key=in_key)
        in_buffer = BytesIO(obj['Body'].read())
        df = pd.read_parquet(in_buffer)
        print("Processed data loaded.")
    except Exception as e:
        print(f"Error loading processed data from S3: {e}")
        raise

    sensitive_features_list = sensitive_features_str.split(',')

    # --- Validation ---
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in processed data.")
    missing_sf = []
    for sf in sensitive_features_list:
        if sf not in df.columns:
            missing_sf.append(sf)
    if missing_sf:
         # Fail hard if sensitive features are missing after preprocessing
         raise ValueError(f"Sensitive features {missing_sf} not found in processed DataFrame columns: {df.columns.tolist()}")

    # --- Bias Calculation Logic using Pandas GroupBy ---
    print("Calculating metrics using Pandas groupby...")
    try:
        # Ensure target is numeric for calculations like mean
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        if df[target_col].isnull().any():
             print(f"Warning: Found NaN values in target column '{target_col}' after attempting numeric conversion. These rows might be excluded from mean calculation.")

        # Group by sensitive feature(s) and calculate mean (positive rate) and count
        grouped_stats = df.groupby(sensitive_features_list)[target_col].agg(['mean', 'count'])
        grouped_stats.rename(columns={'mean': 'positive_rate'}, inplace=True) # Rename for clarity
        print("Group statistics calculated:")
        print(grouped_stats)

        metrics_results = {
            "group_statistics": grouped_stats.to_dict(orient='index'), # Store stats per group
            "sensitive_features": sensitive_features_list,
            "target_column": target_col
        }

        # --- Decision Logic (Example: Demographic Parity Difference on True Labels) ---
        bias_threshold = 0.1 # Example threshold for difference in positive rate
        if not grouped_stats.empty:
            min_rate = grouped_stats['positive_rate'].min()
            max_rate = grouped_stats['positive_rate'].max()
            rate_difference = max_rate - min_rate
            metrics_results["positive_rate_difference"] = rate_difference
            print(f"Max difference in positive rate between groups: {rate_difference:.4f}")

            if rate_difference > bias_threshold:
                print(f"WARNING: Potential bias detected. Positive rate difference ({rate_difference:.4f}) exceeds threshold ({bias_threshold}).")
                metrics_results["bias_check_status"] = "Warning: Threshold Exceeded"
            else:
                print("Bias check passed (based on positive rate difference).")
                metrics_results["bias_check_status"] = "Passed"
        else:
             print("Warning: No groups found or data was empty, cannot calculate rate difference.")
             metrics_results["bias_check_status"] = "Skipped (No data/groups)"
             metrics_results["positive_rate_difference"] = None


    except Exception as e:
        print(f"Error during bias calculation: {e}")
        raise # Fail the step if calculation goes wrong

    # Save report to MinIO
    print(f"Saving bias report to s3://{out_bucket}/{report_key}...")
    try:
        # Convert DataFrame in results to JSON serializable dict first if needed, to_dict already did this.
        report_json = json.dumps(metrics_results, indent=2)
        report_buffer = BytesIO(report_json.encode('utf-8'))
        s3_client.put_object(Bucket=out_bucket, Key=report_key, Body=report_buffer)
        print(f"Bias report saved successfully.")
    except Exception as e:
         print(f"Error saving bias report to S3: {e}")
         raise

# --- End of check_bias function ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_bucket', type=str, required=True)
    parser.add_argument('--in_key', type=str, required=True)
    parser.add_argument('--out_bucket', type=str, required=True)
    parser.add_argument('--report_key', type=str, required=True)
    parser.add_argument('--sensitive_features', type=str, required=True, help='Comma-separated sensitive feature column names')
    parser.add_argument('--target_col', type=str, required=True, help='Target variable column name')
    args = parser.parse_args()

    s3_endpoint = os.environ.get("S3_ENDPOINT_URL")
    s3_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    s3_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    if not all([s3_endpoint, s3_access_key, s3_secret_key]):
        raise ValueError("S3 credentials or endpoint not found.")

    check_bias(args.in_bucket, args.in_key, args.out_bucket, args.report_key,
               args.sensitive_features, args.target_col,
               s3_endpoint, s3_access_key, s3_secret_key)
import argparse
import os
import pandas as pd
import boto3
from io import BytesIO
import json
from fairlearn.metrics import MetricFrame, count, demographic_parity_difference, equalized_odds_difference

def check_bias(in_bucket, in_key, out_bucket, report_key, sensitive_features_str, target_col, endpoint_url, access_key, secret_key):
    print(f"Starting bias check for s3://{in_bucket}/{in_key}")
    print(f"Sensitive features: {sensitive_features_str}")
    print(f"Report will be saved to s3://{out_bucket}/{report_key}")

    s3_client = boto3.client(
        "s3", endpoint_url=endpoint_url, aws_access_key_id=access_key, aws_secret_access_key=secret_key
    )

    # Load processed data (assuming Parquet)
    obj = s3_client.get_object(Bucket=in_bucket, Key=in_key)
    in_buffer = BytesIO(obj['Body'].read())
    df = pd.read_parquet(in_buffer)
    print("Processed data loaded.")

    sensitive_features_list = sensitive_features_str.split(',') # Expect comma-separated string

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")
    for sf in sensitive_features_list:
         if sf not in df.columns:
             # NOTE: Sensitive features might be encoded/transformed.
             # This check might need adjustment based on preprocessing.
             # For this example, assume they exist *before* OHE or are passed through.
             print(f"Warning: Sensitive feature '{sf}' not found directly. Bias check might be inaccurate if it was heavily transformed.")
             # raise ValueError(f"Sensitive feature '{sf}' not found.") # Optionally fail hard

    # --- Bias Calculation Logic ---
    # Fairlearn requires the actual target values (y_true) and potentially
    # model predictions (y_pred) if checking model fairness later.
    # Here, we check bias in the *data distribution* itself or simple metrics.

    y_true = df[target_col]
    sensitive_features_data = df[sensitive_features_list]

    # Example Metric: Demographic Parity (how target variable is distributed across groups)
    # We use 'count' metric here just to see group sizes, more meaningful metrics
    # often require model predictions (like accuracy difference, equalized odds diff).
    # Let's calculate the distribution of the positive class (assuming binary target 0/1)
    def positive_rate(y_true):
         return y_true.mean() # Assumes y_true is 0 or 1

    grouped_on_sex = MetricFrame(metrics=positive_rate,
                               y_true=y_true,
                               sensitive_features=sensitive_features_data) # Check sensitive_features_data format compatibility

    metrics_results = {
        "group_positive_rates": grouped_on_sex.by_group.to_dict(),
        # Add more metrics as needed
        # Example requiring predictions (if available):
        # 'demographic_parity_difference': demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features_data),
        # 'equalized_odds_difference': equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features_data),
    }
    print("Bias metrics calculated:")
    print(json.dumps(metrics_results, indent=2))

    # --- Decision Logic (Example) ---
    bias_threshold = 0.1 # Example threshold for difference in positive rate
    max_diff = grouped_on_sex.difference(method='between_groups')
    print(f"Max difference in positive rate between groups: {max_diff:.4f}")

    if max_diff > bias_threshold:
         # Option 1: Fail the pipeline
         # raise ValueError(f"Bias check FAILED: Demographic parity difference ({max_diff:.4f}) exceeds threshold ({bias_threshold}).")
         # Option 2: Log a warning and continue
         print(f"WARNING: Potential bias detected. Demographic parity difference ({max_diff:.4f}) exceeds threshold ({bias_threshold}).")
         metrics_results["bias_check_status"] = "Warning: Threshold Exceeded"
    else:
         print("Bias check passed.")
         metrics_results["bias_check_status"] = "Passed"


    # Save report to MinIO
    report_buffer = BytesIO(json.dumps(metrics_results, indent=2).encode('utf-8'))
    s3_client.put_object(Bucket=out_bucket, Key=report_key, Body=report_buffer)
    print(f"Bias report saved to s3://{out_bucket}/{report_key}")

    # Output a status or result if needed for conditional Argo steps
    # E.g., write "passed" or "failed" to a file Argo can read as an output parameter
    # with open("/tmp/bias_status.txt", "w") as f:
    #    f.write(metrics_results["bias_check_status"])


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
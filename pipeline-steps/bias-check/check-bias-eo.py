#---EQUALize ODDs---#
#Fairlearn
#https://fairlearn.org/v0.8.0/user_guide/tutorials/plot_equalized_odds.html

from fairlearn.metrics import equalized_odds_difference, MetricFrame
import argparse
import os
import pandas as pd
import boto3
from io import BytesIO
import json

def check_equalized_odds(in_bucket, in_key, out_bucket, report_key, sensitive_features_str, target_col, endpoint_url, access_key, secret_key):
    print(f"Starting Equalized Odds check for s3://{in_bucket}/{in_key}")
    print(f"Using sensitive features: {sensitive_features_str}")
    print(f"Target: {target_col}, Prediction: {pred_col}")

    s3_client = boto3.client(
        "s3", endpoint_url=endpoint_url, aws_access_key_id=access_key, aws_secret_access_key=secret_key
    )

    # Load data
    obj = s3_client.get_object(Bucket=in_bucket, Key=in_key)
    df = pd.read_parquet(BytesIO(obj["Body"].read()))
    print("Data loaded.")

    sensitive_features_list = sensitive_features_str.split(',')

    # --- Validation ---
    for col in [target_col, pred_col] + sensitive_features_list:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=[target_col, pred_col] + sensitive_features_list)

    # Create 1D sensitive feature index
    if len(sensitive_features_list) == 1:
        sensitive_features = df[sensitive_features_list[0]]
    else:
        sensitive_features = df[sensitive_features_list].apply(lambda row: tuple(row), axis=1)

    y_true = df[target_col]
    y_pred = df[pred_col]

    print("Computing Equalized Odds metrics...")
    eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)

    mf = MetricFrame(
        metrics=["true_positive_rate", "false_positive_rate"],
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )

    results = {
        "equalized_odds_difference": eo_diff,
        "true_positive_rate_by_group": mf.by_group["true_positive_rate"].to_dict(),
        "false_positive_rate_by_group": mf.by_group["false_positive_rate"].to_dict(),
        "sensitive_features": sensitive_features_list,
        "target_column": target_col,
        
        "check_type": "Equalized Odds"
    }

    threshold = 0.1
    if abs(eo_diff) > threshold:
        results["bias_check_status"] = f"Warning: EO diff {eo_diff:.4f} > threshold {threshold}"
    else:
        results["bias_check_status"] = "Passed"

    # Save result
    print("Uploading report to S3...")
    report_json = json.dumps(results, indent=2)
    s3_client.put_object(
        Bucket=out_bucket,
        Key=report_key,
        Body=BytesIO(report_json.encode("utf-8"))
    )
    print("Report uploaded successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_bucket", required=True)
    parser.add_argument("--in_key", required=True)
    parser.add_argument("--out_bucket", required=True)
    parser.add_argument("--report_key", required=True)
    parser.add_argument("--sensitive_features", required=True)
    parser.add_argument("--target_col", required=True)
    parser.add_argument("--pred_col", required=True)
    args = parser.parse_args()

    s3_endpoint = os.environ.get("S3_ENDPOINT_URL")
    s3_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    s3_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    check_equalized_odds(
        args.in_bucket, args.in_key,
        args.out_bucket, args.report_key,
        args.sensitive_features, args.target_col, 
        s3_endpoint, s3_access_key, s3_secret_key
    )

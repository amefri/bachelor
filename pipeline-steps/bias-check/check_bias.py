# bias_check_demographic_parity.py

from fairlearn.metrics import selection_rate, demographic_parity_difference, MetricFrame
import argparse
import os
import pandas as pd
import boto3
from io import BytesIO
import json

def run_demographic_parity_check(in_bucket, in_key, out_bucket, report_key,
                                  sensitive_features_str, target_col,
                                  endpoint_url, access_key, secret_key):
    print("Checking demographic parity...")

    s3 = boto3.client("s3", endpoint_url=endpoint_url,
                      aws_access_key_id=access_key, aws_secret_access_key=secret_key)

    obj = s3.get_object(Bucket=in_bucket, Key=in_key)
    df = pd.read_parquet(BytesIO(obj["Body"].read()))

    sensitive_features_list = sensitive_features_str.split(',')
    missing = [col for col in [target_col] + sensitive_features_list if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    df = df.dropna(subset=[target_col] + sensitive_features_list)

    if len(sensitive_features_list) == 1:
        sensitive_features = df[sensitive_features_list[0]]
    else:
        sensitive_features = df[sensitive_features_list].apply(tuple, axis=1)

    y_true = df[target_col]

    mf = MetricFrame(
        metrics=selection_rate,
        y_true=y_true,
        y_pred=y_true,
        sensitive_features=sensitive_features
    )

    dp_diff = demographic_parity_difference(y_true, y_true, sensitive_features=sensitive_features)

    result = {
        "selection_rate_by_group": mf.by_group.to_dict(),
        "overall_selection_rate": mf.overall,
        "demographic_parity_difference": dp_diff,
        "bias_check_status": "Warning" if abs(dp_diff) > 0.1 else "Passed",
        "sensitive_features": sensitive_features_list,
        "target_column": target_col,
        "check_type": "Demographic Parity"
    }

    report_buffer = BytesIO(json.dumps(result, indent=2).encode("utf-8"))
    s3.put_object(Bucket=out_bucket, Key=report_key, Body=report_buffer)
    print("Demographic parity report saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_bucket', required=True)
    parser.add_argument('--in_key', required=True)
    parser.add_argument('--out_bucket', required=True)
    parser.add_argument('--report_key', required=True)
    parser.add_argument('--sensitive_features', required=True)
    parser.add_argument('--target_col', required=True)
    args = parser.parse_args()

    run_demographic_parity_check(
        args.in_bucket, args.in_key, args.out_bucket, args.report_key,
        args.sensitive_features, args.target_col,
        os.environ.get("S3_ENDPOINT_URL"),
        os.environ.get("AWS_ACCESS_KEY_ID"),
        os.environ.get("AWS_SECRET_ACCESS_KEY")
    )

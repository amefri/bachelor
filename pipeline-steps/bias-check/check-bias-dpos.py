import argparse
import os
import pandas as pd
import boto3
from io import BytesIO
import json
import numpy as np 


def calculate_group_positive_rates(df, group_cols, target_col):
    """Calculates positive rate (mean of target) and counts per group."""
    try:
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        if df[target_col].isnull().any():
            print(f"Warning: NaN values found in target column '{target_col}' after conversion. Rows with NaN target will be excluded from rate calculation.")

        grouped_stats = df.groupby(group_cols)[target_col].agg(
            positive_rate=lambda x: x.mean(skipna=True),
            count='count'
        )
        # Ensure positive_rate column exists even if empty group
        if 'positive_rate' not in grouped_stats.columns:
             grouped_stats['positive_rate'] = np.nan

        # Fill NaN rates that might occur if a group has *only* NaN targets or is empty
        # (though count should be 0 then)
        grouped_stats['positive_rate'] = grouped_stats['positive_rate'].fillna(np.nan) # Keep NaN explicit here

        print("Group statistics (Positive Rate & Count):")
        print(grouped_stats)
        return grouped_stats

    except KeyError as e:
        print(f"ERROR: KeyError during grouping/aggregation: {e}. Check column names.")
        raise
    except Exception as e:
        print(f"ERROR: Unexpected error during group rate calculation: {type(e).__name__} - {e}")
        raise


def check_disparate_impact_data_distribution(in_bucket, in_key, out_bucket, report_key, sensitive_features_str, target_col, endpoint_url, access_key, secret_key):
   
    print(f"--- Starting Data Bias Check (Focus: Disparate Impact Min/Max Ratio on Data) ---")
    print(f"Input: s3://{in_bucket}/{in_key}")
    print(f"Sensitive features: {sensitive_features_str}")
    print(f"Target column (positive outcome): {target_col}")
    print(f"Report output: s3://{out_bucket}/{report_key}")

    s3_client = boto3.client(
        "s3", endpoint_url=endpoint_url, aws_access_key_id=access_key, aws_secret_access_key=secret_key
    )

    # --- Load Data ---
    try:
        print("Loading processed data...")
        obj = s3_client.get_object(Bucket=in_bucket, Key=in_key)
        in_buffer = BytesIO(obj['Body'].read())
        df = pd.read_parquet(in_buffer)
        print(f"Data loaded. Shape: {df.shape}, Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"ERROR: Failed loading processed data. {type(e).__name__}: {e}")
        raise

    sensitive_features_list = sensitive_features_str.split(',')

    # --- Validation ---
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in processed data.")
    missing_sf = [sf for sf in sensitive_features_list if sf not in df.columns]
    if missing_sf:
         raise ValueError(f"Sensitive features {missing_sf} not found in processed DataFrame columns: {df.columns.tolist()}")
    print("Required columns found.")

    # --- Calculate Positive Rates per Group ---
    try:
        print("Calculating positive rate per sensitive group...")
        group_stats = calculate_group_positive_rates(df, sensitive_features_list, target_col)

        # Prepare base results dictionary
        metrics_results = {
            "analysis_type": "Data Distribution Bias Check (Disparate Impact Min/Max Ratio)",
            "metric_calculated": "Ratio of Min/Max Positive Rate per Group",
            "sensitive_features": sensitive_features_list,
            "target_column": target_col,
            "group_statistics": group_stats.reset_index().to_dict(orient='records')
        }

        # --- Calculate Disparate Impact (Min/Max Ratio) ---
        di_ratio = None
        status = "Skipped"
        bias_threshold = 0.8 # Common threshold for DI 

        # Only calculate if we have rates for at least two groups and no NaNs
        valid_rates = group_stats['positive_rate'].dropna()
        if len(valid_rates) >= 2:
            min_rate = valid_rates.min()
            max_rate = valid_rates.max()
            metrics_results["min_positive_rate"] = min_rate
            metrics_results["max_positive_rate"] = max_rate
            print(f"Min positive rate: {min_rate:.4f}, Max positive rate: {max_rate:.4f}")

            # Handle division by zero if max_rate is 0
            if max_rate > 0:
                di_ratio = min_rate / max_rate
                metrics_results["disparate_impact_min_max_ratio"] = di_ratio
                print(f"Disparate Impact (Min/Max Ratio): {di_ratio:.4f}")

                if di_ratio < bias_threshold:
                    status = "Warning: Potential Adverse Impact Detected"
                    print(f"WARNING: {status}. DI ratio ({di_ratio:.4f}) < threshold ({bias_threshold}).")
                else:
                    status = "Passed"
                    print(f"Disparate Impact check passed (ratio >= {bias_threshold}).")
            elif min_rate == 0: # If max_rate is 0 and min_rate is 0, DI is effectively 1 (parity)
                 di_ratio = 1.0
                 metrics_results["disparate_impact_min_max_ratio"] = di_ratio
                 status = "Passed (All rates zero)"
                 print(f"Disparate Impact check passed (all group rates are zero). DI Ratio = {di_ratio:.1f}")
            else: # max_rate is 0 but min_rate is not (shouldn't happen, but defensively)
                 status = "Error (Max rate zero, Min rate non-zero)"
                 print(f"ERROR: Cannot calculate DI ratio meaningfully: {status}")
                 metrics_results["disparate_impact_min_max_ratio"] = None

        elif len(valid_rates) == 1:
            status = "Skipped (Only one group)"
            print(f"Warning: {status}. Cannot calculate DI ratio.")
            metrics_results["min_positive_rate"] = valid_rates.iloc[0]
            metrics_results["max_positive_rate"] = valid_rates.iloc[0]
            metrics_results["disparate_impact_min_max_ratio"] = 1.0 # Parity with only one group
        else: # No valid rates (empty data or all NaN targets)
             status = "Skipped (No valid rates)"
             print(f"Warning: {status}. Cannot calculate DI ratio.")
             metrics_results["min_positive_rate"] = None
             metrics_results["max_positive_rate"] = None
             metrics_results["disparate_impact_min_max_ratio"] = None

        metrics_results["bias_check_status"] = status
        metrics_results["disparate_impact_threshold"] = bias_threshold

    except Exception as e:
        print(f"ERROR: Unexpected error during bias calculation. {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

    # --- Save Report ---
    print(f"Saving bias report...")
    try:
        report_json = json.dumps(metrics_results, indent=2, default=str)
        report_buffer = BytesIO(report_json.encode('utf-8'))
        s3_client.put_object(Bucket=out_bucket, Key=report_key, Body=report_buffer)
        print(f"Bias report saved successfully to s3://{out_bucket}/{report_key}")
    except Exception as e:
         print(f"ERROR: Failed to save bias report. {type(e).__name__}: {e}")
         raise

    print(f"--- Bias Check Finished ---")


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

  
    check_disparate_impact_data_distribution(args.in_bucket, args.in_key, args.out_bucket, args.report_key,
                                              args.sensitive_features, args.target_col,
                                              s3_endpoint, s3_access_key, s3_secret_key)
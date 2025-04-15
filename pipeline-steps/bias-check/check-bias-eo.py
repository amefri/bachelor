import argparse
import os
import pandas as pd
import numpy as np # Import numpy
import boto3
from io import BytesIO
import json
from fairlearn.metrics import MetricFrame, mean_prediction # Fairlearn specific metrics/tools
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import sys
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Helper function to handle potentially non-serializable data
# (Keep the enhanced version from the previous correction)
def make_serializable(data):
    if isinstance(data, dict):
        return {k: make_serializable(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [make_serializable(item) for item in data]
    elif isinstance(data, (pd.Series, pd.DataFrame)):
        if isinstance(data, pd.Series):
            return data.astype(object).where(pd.notnull(data), None).to_dict()
        else: # DataFrame
            return data.astype(object).where(pd.notnull(data), None).to_dict(orient='records')
    elif hasattr(data, 'item'): # Handle numpy types (like np.float64, np.int64)
        # Check for NaN before item() as item() on NaN can raise error depending on numpy version
        if pd.isna(data):
            return None
        return data.item()
    elif pd.isna(data):
        return None
    elif isinstance(data, (int, float, str, bool)):
        return data
    else:
        logging.warning(f"Attempting direct conversion for potentially non-serializable type: {type(data)}")
        try:
            return str(data)
        except Exception:
            logging.error(f"Could not serialize type {type(data)}.")
            return f"Unserializable Type: {type(data)}"


# Main bias check function for REGRESSION
def run_s3_regression_bias_check(in_bucket, in_key, out_bucket, report_key,
                                 sensitive_features_str,
                                 target_col,
                                 pred_col,
                                 endpoint_url, access_key, secret_key):

    logging.info(f"--- Starting Regression Bias Check ---") # Renamed check type
    logging.info(f"Input Data: s3://{in_bucket}/{in_key} (Expected Format: Parquet)")
    logging.info(f"Output Report: s3://{out_bucket}/{report_key}")
    logging.info(f"Target Column (True Value): {target_col}")
    logging.info(f"Prediction Column: {pred_col}")
    logging.info(f"Sensitive Feature(s): {sensitive_features_str}")

    s3_client = boto3.client(
        "s3", endpoint_url=endpoint_url, aws_access_key_id=access_key, aws_secret_access_key=secret_key
    )

    # --- Load Data ---
    try:
        logging.info("Loading data from S3...")
        obj = s3_client.get_object(Bucket=in_bucket, Key=in_key)
        df = pd.read_parquet(BytesIO(obj["Body"].read()))
        logging.info(f"Data loaded. Shape: {df.shape}")
    except Exception as e:
        logging.error(f"Failed to load data from s3://{in_bucket}/{in_key}. {type(e).__name__}: {e}")
        raise

    # --- Prepare Data & Validate ---
    sensitive_features_list = [s.strip() for s in sensitive_features_str.strip().split(',') if s.strip()]
    if not sensitive_features_list:
         raise ValueError("Sensitive features list cannot be empty after parsing.")

    required_cols = [target_col, pred_col] + sensitive_features_list

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available: {df.columns.tolist()}")

    initial_rows = len(df)
    # Ensure target and prediction are numeric, drop rows if not (or handle conversion errors)
    try:
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df[pred_col] = pd.to_numeric(df[pred_col], errors='coerce')
    except Exception as e:
        logging.error(f"Failed to convert target/prediction columns to numeric: {e}")
        raise

    df_cleaned = df.dropna(subset=required_cols) # Drop rows with NA in any essential column
    rows_dropped = initial_rows - len(df_cleaned)
    if rows_dropped > 0:
        logging.warning(f"Dropped {rows_dropped} rows due to missing values in essential columns ({required_cols}) or non-numeric target/prediction.")

    if len(df_cleaned) == 0:
         raise ValueError("No valid data remaining after dropping NA/non-numeric values.")

    y_true = df_cleaned[target_col]
    y_pred = df_cleaned[pred_col]

    logging.info(f"--- Data distributions after cleaning ---")
    logging.info(f"Target '{target_col}' stats:\n{y_true.describe()}")
    logging.info(f"Prediction '{pred_col}' stats:\n{y_pred.describe()}")


    # Create sensitive features column/index
    if len(sensitive_features_list) == 1:
        sensitive_features_col_name = sensitive_features_list[0]
        sensitive = df_cleaned[sensitive_features_col_name]
        logging.info(f"Using single sensitive feature: '{sensitive_features_col_name}'")
    else:
        sensitive = df_cleaned[sensitive_features_list].apply(lambda row: tuple(row.astype(str)), axis=1)
        combined_name = "_".join(sensitive_features_list)
        sensitive.name = combined_name
        logging.info(f"Using combined sensitive features: {sensitive_features_list} as '{combined_name}'")

    logging.info(f"Unique sensitive feature values/combinations found: {sensitive.unique()}")

    # --- Calculate Fairness Metrics for REGRESSION ---
    logging.info("Calculating regression fairness metrics using Fairlearn...")
    results = {
        "check_type": "Regression Bias Check", # Updated check type
        "input_data": f"s3://{in_bucket}/{in_key}",
        "sensitive_features_used": sensitive_features_list,
        "target_column": target_col,
        "prediction_column": pred_col,
        "metrics_by_group": {}, # Will be populated
        "disparity_metrics": {}, # Store differences/ratios here
        "bias_check_status": "Calculation Error",
        "error_message": None
    }

    try:
        # Define the regression metrics to calculate
        metric_fns = {
            "mean_prediction": mean_prediction,
            "MAE": mean_absolute_error,
            "RMSE": root_mean_squared_error,
        }

        # Use MetricFrame with the regression metrics
        metric_frame = MetricFrame(
            metrics=metric_fns,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive
        )

        # Calculate overall metrics (optional but good context)
        overall_metrics = {name: func(y_true, y_pred) for name, func in metric_fns.items()}

        # Calculate differences between groups (max - min)
        # Note: metric_frame.difference() returns a Series with differences for all metrics
        differences = metric_frame.difference(method='between_groups')

        # --- Populate Results ---
        results["metrics_by_group"] = make_serializable(metric_frame.by_group)
        results["disparity_metrics"]["difference"] = make_serializable(differences)
        results["overall_metrics"] = make_serializable(overall_metrics)


        # --- Display Results ---
        logging.info("\n📊 Overall Regression Metrics:")
        for name, value in results["overall_metrics"].items():
             logging.info(f"  {name}: {value:.4f}")

        logging.info("\n📊 Regression Metrics by Group:")
        try:
            logging.info(metric_frame.by_group.to_string())
        except Exception as display_err:
            logging.warning(f"Could not format by_group metrics for display: {display_err}")
            logging.info(f"(Raw data): {results['metrics_by_group']}")


        logging.info("\n⚖️ Disparity Metrics (Difference between Max and Min Group):")
        for name, value in results["disparity_metrics"]["difference"].items():
            logging.info(f"  Difference in {name}: {value:.4f}")


        # --- Add optional status check based on thresholds ---
        # Define thresholds (THESE ARE EXAMPLES - adjust based on your domain knowledge)
        mae_diff_threshold = 50.0  # Allow MAE to differ by up to 50 units (e.g., days)
        rmse_diff_threshold = 75.0 # Allow RMSE to differ by up to 75 units
        mean_pred_diff_threshold = 100.0 # Allow mean prediction to differ by 100 units

        warnings = []
        if pd.isna(differences['MAE']) or pd.isna(differences['RMSE']) or pd.isna(differences['mean_prediction']):
             results["bias_check_status"] = "Inconclusive (Could not calculate disparity)"
        else:
            if abs(differences['MAE']) > mae_diff_threshold:
                warnings.append(f"MAE difference ({differences['MAE']:.2f}) > threshold ({mae_diff_threshold})")
            if abs(differences['RMSE']) > rmse_diff_threshold:
                 warnings.append(f"RMSE difference ({differences['RMSE']:.2f}) > threshold ({rmse_diff_threshold})")
            if abs(differences['mean_prediction']) > mean_pred_diff_threshold:
                 warnings.append(f"Mean Prediction difference ({differences['mean_prediction']:.2f}) > threshold ({mean_pred_diff_threshold})")

            if warnings:
                results["bias_check_status"] = "Warning: " + "; ".join(warnings)
            else:
                results["bias_check_status"] = "Passed (Disparities within thresholds)"

        logging.info(f"Bias Check Status: {results['bias_check_status']}")


    except Exception as e:
        error_msg = f"Failed during Fairlearn metric calculation. {type(e).__name__}: {e}"
        logging.error(error_msg, exc_info=True) # Log traceback for debugging
        results["error_message"] = error_msg
        results["bias_check_status"] = "Calculation Error"
        # Continue to save the report with the error


    # --- Save Report to S3 ---
    try:
        logging.info(f"Uploading report to s3://{out_bucket}/{report_key}...")
        # Use the enhanced make_serializable here
        report_json = json.dumps(results, indent=2, default=make_serializable, ensure_ascii=False)
        s3_client.put_object(
            Bucket=out_bucket,
            Key=report_key,
            Body=report_json.encode("utf-8"),
            ContentType='application/json'
        )
        logging.info("Report uploaded successfully.")
    except TypeError as json_err:
         logging.error(f"JSON Serialization Error: {json_err}. Check 'make_serializable' function.", exc_info=True)
         results["error_message"] = f"JSON Serialization Error: {json_err}. Report might be incomplete."
         results["bias_check_status"] = "Report Saving Error"
         # Try saving a simplified report
         try:
             minimal_report = {"error": results["error_message"], "status": results["bias_check_status"]}
             s3_client.put_object(Bucket=out_bucket, Key=report_key, Body=json.dumps(minimal_report).encode("utf-8"), ContentType='application/json')
             logging.warning("Uploaded a minimal error report instead.")
         except Exception as fallback_err:
             logging.error(f"Failed even to upload minimal error report: {fallback_err}", exc_info=True)
         raise RuntimeError(f"Failed to save complete report due to JSON error: {json_err}") from json_err

    except Exception as e:
        logging.error(f"Failed to upload report to S3. {type(e).__name__}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to save report to S3: {e}") from e

    logging.info(f"--- Regression Bias Check Finished ---")

    # Optional: Exit with non-zero status if calculation failed or report saving failed
    if "Error" in results["bias_check_status"]:
        sys.exit(f"Exiting due to status: {results['bias_check_status']}. Error: {results.get('error_message', 'Unknown error')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Fairlearn Regression bias check on S3 data.")

    # --- Arguments remain the same as before ---
    parser.add_argument('--in_bucket', type=str, required=True, help="S3 bucket for input Parquet data.")
    parser.add_argument('--in_key', type=str, required=True, help="S3 key for input Parquet data.")
    parser.add_argument('--out_bucket', type=str, required=True, help="S3 bucket to save the JSON bias report.")
    parser.add_argument('--report_key', type=str, required=True, help="S3 key for the output JSON report file.")
    parser.add_argument('--sensitive_features', type=str, required=True, help="Comma-separated list of sensitive feature column names.")
    parser.add_argument('--target_col', type=str, required=True, help="Name of the true target value column.")
    parser.add_argument('--pred_col', type=str, required=True, help="Name of the prediction column.")

    args = parser.parse_args()

    # --- Get S3 Credentials (from environment variables) ---
    s3_endpoint = os.environ.get("S3_ENDPOINT_URL")
    s3_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    s3_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    if not all([s3_endpoint, s3_access_key, s3_secret_key]):
        logging.warning("S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, or AWS_SECRET_ACCESS_KEY potentially missing.")

    # --- Run the Regression Bias Check ---
    run_s3_regression_bias_check(
        in_bucket=args.in_bucket,
        in_key=args.in_key,
        out_bucket=args.out_bucket,
        report_key=args.report_key,
        sensitive_features_str=args.sensitive_features,
        target_col=args.target_col,
        pred_col=args.pred_col,
        endpoint_url=s3_endpoint,
        access_key=s3_access_key,
        secret_key=s3_secret_key
    )
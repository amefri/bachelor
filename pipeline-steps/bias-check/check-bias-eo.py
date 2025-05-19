import argparse
import os
import pandas as pd
import numpy as np # Import numpy
import boto3
from io import BytesIO
import json

from fairlearn.metrics import MetricFrame, count # `count` ist immer nützlich
from fairlearn.metrics import selection_rate, true_positive_rate, false_positive_rate # Basismetriken für EO
from fairlearn.metrics import equalized_odds_difference # Direkte EO-Differenzmetrik
import sys
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Helper function to handle potentially non-serializable data (bleibt gleich)
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
    elif hasattr(data, 'item'):
        if pd.isna(data): return None
        return data.item()
    elif pd.isna(data): return None
    elif isinstance(data, (int, float, str, bool)): return data
    else:
        logging.warning(f"Attempting direct conversion for type: {type(data)}")
        try: return str(data)
        except Exception: logging.error(f"Could not serialize type {type(data)}."); return f"Unserializable: {type(data)}"

# Main bias check function for EO
def run_s3_classification_bias_check(in_bucket, in_key, out_bucket, report_key,
                                     sensitive_features_str,
                                     target_col,
                                     pred_col, # predictions (0 oder 1)
                                     endpoint_url, access_key, secret_key):

    logging.info(f"--- Starting Classification Bias Check (Equalized Odds) ---")
    logging.info(f"Input Data: s3://{in_bucket}/{in_key} (Expected Format: Parquet)")
    logging.info(f"Output Report: s3://{out_bucket}/{report_key}")
    logging.info(f"Target Column (True Label): {target_col}")
    logging.info(f"Prediction Column (Predicted Label): {pred_col}")
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
    # Ensure target and prediction are numeric (0 oder 1) and not NaN
    try:
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df[pred_col] = pd.to_numeric(df[pred_col], errors='coerce')
       
        if not df[target_col].isin([0, 1]).all():
            logging.warning(f"Target column '{target_col}' contains non-binary values after numeric conversion. This might affect classification metrics.")
        if not df[pred_col].isin([0, 1]).all():
            logging.warning(f"Prediction column '{pred_col}' contains non-binary values after numeric conversion. This might affect classification metrics.")

    except Exception as e:
        logging.error(f"Failed to convert target/prediction columns to numeric: {e}")
        raise

    df_cleaned = df.dropna(subset=required_cols)
    rows_dropped = initial_rows - len(df_cleaned)
    if rows_dropped > 0:
        logging.warning(f"Dropped {rows_dropped} rows due to missing values in essential columns ({required_cols}).")

    if len(df_cleaned) == 0:
         raise ValueError("No valid data remaining after dropping NA values.")

    y_true = df_cleaned[target_col].astype(int) # Sicherstellen, dass es int ist
    y_pred = df_cleaned[pred_col].astype(int) # hier auch

    logging.info(f"--- Data distributions after cleaning ---")
    logging.info(f"Target '{target_col}' value counts:\n{y_true.value_counts(normalize=True)}")
    logging.info(f"Prediction '{pred_col}' value counts:\n{y_pred.value_counts(normalize=True)}")

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

    logging.info(f"Unique sensitive feature values/combinations found: {sensitive.unique().tolist()}")

    # --- Calculate Fairness Metrics for EO ---
    logging.info("Calculating classification fairness metrics (for Equalized Odds) using Fairlearn...")
    results = {
        "check_type": "Classification Bias Check (Equalized Odds)",
        "input_data": f"s3://{in_bucket}/{in_key}",
        "sensitive_features_used": sensitive_features_list,
        "target_column": target_col,
        "prediction_column": pred_col,
        "metrics_by_group": {},
        "disparity_metrics": {},
        "bias_check_status": "Calculation Error",
        "error_message": None
    }

    try:
        # Define the classification metrics for EO
        metric_fns = {
            "count": count, # Anzahl der Samples pro Gruppe
            "selection_rate": selection_rate, # P(Y_pred=1) - Positivrate der Vorhersagen
            "true_positive_rate": true_positive_rate, 
            "false_positive_rate": false_positive_rate, 
            
        }

        metric_frame = MetricFrame(
            metrics=metric_fns,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive
        )

        overall_metrics = {
            "count": len(y_true),
            "selection_rate": selection_rate(y_true, y_pred),
            "true_positive_rate": true_positive_rate(y_true, y_pred),
            "false_positive_rate": false_positive_rate(y_true, y_pred),
            "equalized_odds_difference": equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive) # Gesamtdifferenz
        }

        
        differences = metric_frame.difference(method='between_groups')

        # --- Populate Results ---
        results["metrics_by_group"] = make_serializable(metric_frame.by_group)
        results["disparity_metrics"]["difference"] = make_serializable(differences)
        results["overall_metrics"] = make_serializable(overall_metrics)

        # --- Display Results ---
        logging.info("\nOverall Classification Metrics:")
        for name, value in results["overall_metrics"].items():
             logging.info(f"  {name}: {value:.4f}")

        logging.info("\n Classification Metrics by Group:")
        try:
            logging.info(metric_frame.by_group.to_string())
        except Exception as display_err:
            logging.warning(f"Could not format by_group metrics for display: {display_err}")
            logging.info(f"(Raw data): {results['metrics_by_group']}")

        logging.info("\nDisparity Metrics (Difference between Max and Min Group Value):")
        for name, value in results["disparity_metrics"]["difference"].items():
            logging.info(f"  Difference in {name}: {value:.4f}")

       
        eo_diff_threshold = 0.05 # choose fitting threshold for EO difference

        warnings = []
        # Check the direct EO difference metric
        if pd.isna(overall_metrics['equalized_odds_difference']):
            results["bias_check_status"] = "Inconclusive (Could not calculate Equalized Odds difference)"
        else:
            if abs(overall_metrics['equalized_odds_difference']) > eo_diff_threshold:
                warnings.append(f"Equalized Odds Difference ({overall_metrics['equalized_odds_difference']:.4f}) > threshold ({eo_diff_threshold})")

            if warnings:
                results["bias_check_status"] = "Warning: " + "; ".join(warnings)
            else:
                results["bias_check_status"] = "Passed (Equalized Odds disparity within threshold)"

        logging.info(f"Bias Check Status: {results['bias_check_status']}")

    except Exception as e:
        error_msg = f"Failed during Fairlearn metric calculation. {type(e).__name__}: {e}"
        logging.error(error_msg, exc_info=True)
        results["error_message"] = error_msg
        results["bias_check_status"] = "Calculation Error"

    # --- Save Report to S3 (bleibt gleich) ---
    try:
        logging.info(f"Uploading report to s3://{out_bucket}/{report_key}...")
        report_json = json.dumps(results, indent=2, default=make_serializable, ensure_ascii=False)
        s3_client.put_object(
            Bucket=out_bucket, Key=report_key, Body=report_json.encode("utf-8"), ContentType='application/json'
        )
        logging.info("Report uploaded successfully.")
    except Exception as e: 
        logging.error(f"Failed to save or upload report to S3. {type(e).__name__}: {e}", exc_info=True)
        results["error_message"] = f"Report Saving/Upload Error: {e}. Report might be incomplete or not saved."
        results["bias_check_status"] = "Report Saving/Upload Error"
        
        try:
             minimal_report = {"error": results["error_message"], "status": results["bias_check_status"]}
             s3_client.put_object(Bucket=out_bucket, Key=report_key, Body=json.dumps(minimal_report).encode("utf-8"), ContentType='application/json')
             logging.warning("Uploaded a minimal error report instead.")
        except Exception as fallback_err:
             logging.error(f"Failed even to upload minimal error report: {fallback_err}", exc_info=True)
        raise RuntimeError(f"Failed to save complete report and minimal report: {e}") from e


    logging.info(f"--- Classification Bias Check (Equalized Odds) Finished ---")
    if "Error" in results["bias_check_status"]:
        sys.exit(f"Exiting due to status: {results['bias_check_status']}. Error: {results.get('error_message', 'Unknown error')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Fairlearn Classification (Equalized Odds) bias check on S3 data.")
   
    parser.add_argument('--in_bucket', type=str, required=True)
    parser.add_argument('--in_key', type=str, required=True)
    parser.add_argument('--out_bucket', type=str, required=True)
    parser.add_argument('--report_key', type=str, required=True)
    parser.add_argument('--sensitive_features', type=str, required=True)
    parser.add_argument('--target_col', type=str, required=True)
    parser.add_argument('--pred_col', type=str, required=True) # Vorhersagespalte
    args = parser.parse_args()

    s3_endpoint = os.environ.get("S3_ENDPOINT_URL")
    s3_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    s3_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not all([s3_endpoint, s3_access_key, s3_secret_key]):
        logging.warning("S3 credentials might be missing.")

    run_s3_classification_bias_check(
        in_bucket=args.in_bucket, in_key=args.in_key, out_bucket=args.out_bucket, report_key=args.report_key,
        sensitive_features_str=args.sensitive_features, target_col=args.target_col, pred_col=args.pred_col,
        endpoint_url=s3_endpoint, access_key=s3_access_key, secret_key=s3_secret_key
    )
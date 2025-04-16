# train_predict.py
import argparse
import os
import pandas as pd
import boto3
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor

# --- S3 Data Load/Save Helper Functions --- #
def load_parquet_from_s3(bucket, key, s3_client):
    print(f"Loading data from s3://{bucket}/{key}")
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_parquet(BytesIO(obj['Body'].read()))
        print(f"Loaded data shape: {df.shape}")
        return df
    except Exception as e:
        print(f"ERROR loading data from s3://{bucket}/{key}: {e}")
        raise

def save_parquet_to_s3(df, bucket, key, s3_client):
    print(f"Saving data to s3://{bucket}/{key}")
    try:
        out_buffer = BytesIO()
        df.to_parquet(out_buffer, index=False)
        out_buffer.seek(0)
        s3_client.put_object(Bucket=bucket, Key=key, Body=out_buffer)
        print(f"Successfully saved s3://{bucket}/{key}")
    except Exception as e:
        print(f"ERROR saving data to s3://{bucket}/{key}: {e}")
        raise

# --- Train and Predict --- #
def train_and_predict(train_bucket, train_key, test_bucket, test_key, pred_bucket, pred_key, target_col, pred_col_name, s3_client):
    print("--- Starting Training and Prediction ---")
    train_df = load_parquet_from_s3(train_bucket, train_key, s3_client)
    test_df = load_parquet_from_s3(test_bucket, test_key, s3_client)

    # Separate features, target, and sensitive columns (if present - needed for some train steps)
    # Ensure sensitive columns are NOT used for training unless explicitly intended
    # For simplicity here, assume all columns except target are features
    sensitive_cols_in_train = [col for col in train_df.columns if col not in [target_col] and not pd.api.types.is_numeric_dtype(train_df[col])] # Rough guess
    print(f"Potential non-feature columns dropped for training (heuristics): {sensitive_cols_in_train}")


    # Prepare training data (handle potential sensitive columns if they exist)
    # Safer approach: Explicitly list feature columns if known, otherwise drop non-numeric + target
    feature_cols_train = [col for col in train_df.columns if col != target_col and pd.api.types.is_numeric_dtype(train_df[col])] # Simplistic: only numeric
    if not feature_cols_train:
         raise ValueError("No numeric feature columns found for training in train set.")
    print(f"Using training features: {feature_cols_train}")
    X_train = train_df[feature_cols_train]
    y_train = train_df[target_col]

    # Prepare test data - use the same features as training
    feature_cols_test = [col for col in feature_cols_train if col in test_df.columns]
    if len(feature_cols_test) != len(feature_cols_train):
         print(f"Warning: Not all training features found in test set. Using: {feature_cols_test}")
    if not feature_cols_test:
         raise ValueError("No matching training features found in test set.")

    X_test = test_df[feature_cols_test]
    # y_test is not needed for prediction but keep the full test_df

    print("🧠 Training RandomForestClassifier...")
    # --- Model Training ---
    # Consider adding hyperparameter tuning or using a more configured model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
 # Example: Add class_weight
    model.fit(X_train, y_train)
    print("✅ Training complete.")

    print("🔮 Making predictions on the test set...")
    predictions = model.predict(X_test)
    # predictions_proba = model.predict_proba(X_test)[:, 1] # Option: Get probabilities

    # Add predictions back to the original test dataframe
    # Important: Use the prediction column name passed as argument
    test_df[pred_col_name] = predictions
    print(f"✅ Predictions added as column '{pred_col_name}'.")

    print("💾 Saving test set with predictions...")
    save_parquet_to_s3(test_df, pred_bucket, pred_key, s3_client)
    print("--- Training and Prediction Finished ---")
    # No need to return df in this script context

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model and generate predictions on a test set.")

    # Input Data Locations
    parser.add_argument('--train_bucket', type=str, required=True, help="S3 bucket for training data.")
    parser.add_argument('--train_key', type=str, required=True, help="S3 key for training data (Parquet).")
    parser.add_argument('--test_bucket', type=str, required=True, help="S3 bucket for test data.")
    parser.add_argument('--test_key', type=str, required=True, help="S3 key for test data (Parquet).")

    # Output Data Location
    parser.add_argument('--pred_bucket', type=str, required=True, help="S3 bucket to save test data with predictions.")
    parser.add_argument('--pred_key', type=str, required=True, help="S3 key for the output Parquet file with predictions.")

    # Column Names
    parser.add_argument('--target_col', type=str, required=True, help="Name of the target variable column.")
    parser.add_argument('--pred_col_name', type=str, default='prediction', help="Name to give the new prediction column.")

    args = parser.parse_args()

    # Get S3 credentials from environment variables
    s3_endpoint = os.environ.get("S3_ENDPOINT_URL")
    s3_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    s3_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    if not all([s3_endpoint, s3_access_key, s3_secret_key]):
        # Depending on environment, boto3 might find credentials automatically
        print("Warning: S3 endpoint or credentials potentially missing from environment.")
        # raise ValueError("S3 endpoint and credentials must be set via environment variables.") # Uncomment if required

    s3_client = boto3.client(
        "s3", endpoint_url=s3_endpoint, aws_access_key_id=s3_access_key, aws_secret_access_key=s3_secret_key
    )

    train_and_predict(
        train_bucket=args.train_bucket,
        train_key=args.train_key,
        test_bucket=args.test_bucket,
        test_key=args.test_key,
        pred_bucket=args.pred_bucket,
        pred_key=args.pred_key,
        target_col=args.target_col,
        pred_col_name=args.pred_col_name, # Pass the desired prediction column name
        s3_client=s3_client
    )
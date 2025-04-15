import argparse
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import boto3
from io import StringIO, BytesIO
import sys # For version check

# --- Configuration Variables ---
# Define column names based on the INPUT data expected AFTER validation
TARGET_COLUMN = 'ABGANG'
SENSITIVE_COLUMN = 'Geschlecht' # Column to preserve for bias check

# List columns intended as features for the model
# These lists should NOT include TARGET_COLUMN or SENSITIVE_COLUMN
NUMERICAL_FEATURES = ['RGSCode', 'DS_VWD'] # Features to apply numerical transforms
CATEGORICAL_FEATURES = ['Verweildauer', 'Altersgruppe'] # Features to apply categorical transforms

# List columns to drop right after loading, BEFORE any processing
# Example: IDs, irrelevant text, dates if not used as features
DROP_COLS_AFTER_LOAD = ['Datum', 'RGSName']

# Preprocessing Parameters
IMPUTER_STRATEGY_NUM = 'median'
SCALE_NUMERICAL = False # Set to True to add StandardScaler to numerical pipeline
OHE_HANDLE_UNKNOWN = 'ignore' # 'error' or 'ignore'

# Input/Output Format
INPUT_FORMAT = 'csv' # Should match the *raw* data format read
INPUT_ENCODING = 'latin1' # Change if CSV has different encoding (e.g., 'latin-1')
OUTPUT_FORMAT = 'parquet' # 'parquet', 'csv', 'feather'
# --- End Configuration ---

def preprocess_data(in_bucket, in_key, out_bucket, out_key, endpoint_url, access_key, secret_key):

    print(f"--- Starting Preprocessing ---")
    print(f"Input: s3://{in_bucket}/{in_key} ({INPUT_FORMAT}, encoding={INPUT_ENCODING})")
    print(f"Output: s3://{out_bucket}/{out_key} (format={OUTPUT_FORMAT})")
    print(f"Target Column: {TARGET_COLUMN}")
    print(f"Sensitive Column: {SENSITIVE_COLUMN}")

    s3_client = boto3.client(
        "s3", endpoint_url=endpoint_url, aws_access_key_id=access_key, aws_secret_access_key=secret_key
    )

    # --- Load Data ---
    try:
        print("Loading data from S3...")
        obj = s3_client.get_object(Bucket=in_bucket, Key=in_key)
        raw_body = obj['Body'].read()
        if INPUT_FORMAT == 'csv':
            if INPUT_FORMAT == 'csv':
             delimiter=';' 
            df = pd.read_csv(BytesIO(raw_body), encoding=INPUT_ENCODING, delimiter=delimiter)
            if SENSITIVE_COLUMN in df.columns:
                geschlecht_map = {'Männer': 1, 'Frauen': 2}
                df[SENSITIVE_COLUMN] = df[SENSITIVE_COLUMN].map(geschlecht_map)
                print(f"Encoded '{SENSITIVE_COLUMN}' using map: {geschlecht_map}")
            # Add other options like delimiter if needed
          
        # Add elif for other INPUT_FORMAT if needed
        # elif INPUT_FORMAT == 'parquet':
        #     df = pd.read_parquet(BytesIO(raw_body))
        else:
             raise ValueError(f"Unsupported INPUT_FORMAT: {INPUT_FORMAT}")
        print(f"Input data loaded. Shape: {df.shape}")
        print(f"Input columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"ERROR: Failed to load data from S3. {type(e).__name__}: {e}")
        raise

    # --- Initial Column Handling ---
    try:
        # 1. Drop specified columns early
        if DROP_COLS_AFTER_LOAD:
            cols_present_to_drop = [col for col in DROP_COLS_AFTER_LOAD if col in df.columns]
            if cols_present_to_drop:
                 df = df.drop(columns=cols_present_to_drop)
                 print(f"Dropped columns: {cols_present_to_drop}")

        # 2. Validate presence of essential columns AFTER drops
        essential_cols = [TARGET_COLUMN, SENSITIVE_COLUMN] + NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        missing_essential = [col for col in essential_cols if col not in df.columns]
        if missing_essential:
            raise ValueError(f"Missing essential columns after initial drop: {missing_essential}. Available columns: {df.columns.tolist()}")

        # 3. Separate Target, Sensitive Attribute, and Features (X)
        y = df[TARGET_COLUMN]
        sensitive_data = df[[SENSITIVE_COLUMN]] # Keep as DataFrame

        feature_cols = [col for col in df.columns if col not in [TARGET_COLUMN, SENSITIVE_COLUMN]]
        X = df[feature_cols]
        print(f"Features (X) shape: {X.shape}")
        print(f"Target (y) shape: {y.shape}")
        print(f"Sensitive data shape: {sensitive_data.shape}")

    except Exception as e:
         print(f"ERROR: Failed during initial column handling. {type(e).__name__}: {e}")
         raise

    # --- Define and Apply Preprocessing Steps ---
    try:
        print("Defining preprocessing pipelines...")
        # Use only the feature columns actually present in X for defining transformers
        active_numerical_cols = [col for col in NUMERICAL_FEATURES if col in X.columns]
        active_categorical_cols = [col for col in CATEGORICAL_FEATURES if col in X.columns]
        print(f" Active Numerical Features for Transform: {active_numerical_cols}")
        print(f" Active Categorical Features for Transform: {active_categorical_cols}")

        # Numerical Pipeline
        numerical_steps = [('imputer_num', SimpleImputer(strategy=IMPUTER_STRATEGY_NUM))]
        if SCALE_NUMERICAL:
            numerical_steps.append(('scaler', StandardScaler()))
        numerical_transformer = Pipeline(steps=numerical_steps)

        # Categorical Pipeline
        categorical_transformer = Pipeline(steps=[
            # Optional: Add categorical imputer if needed
            # ('imputer_cat', SimpleImputer(strategy=IMPUTER_STRATEGY_CAT)),
            ('onehot', OneHotEncoder(handle_unknown=OHE_HANDLE_UNKNOWN, sparse_output=False))
        ])

        # Column Transformer - apply pipelines only to the selected feature columns in X
        # Use remainder='passthrough' to keep any other columns that might be in X
        # but weren't explicitly listed as numerical/categorical features (e.g., other numeric types)
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, active_numerical_cols),
                ('cat', categorical_transformer, active_categorical_cols)
            ],
            remainder='passthrough', # Keep other columns from X
            verbose_feature_names_out=False # Cleaner output names
        )

        # Attempt to set pandas output for easier handling (requires sklearn >= 1.2)
        try:
            preprocessor.set_output(transform="pandas")
            print("Set ColumnTransformer output to pandas DataFrame.")
            use_pandas_output = True
        except AttributeError:
            print("Warning: scikit-learn version may not support set_output(transform='pandas'). Will construct DataFrame manually.")
            use_pandas_output = False

        print("Applying preprocessing transformations to X...")
        X_processed = preprocessor.fit_transform(X)

        # Construct DataFrame with correct column names
        if use_pandas_output:
            X_processed_df = X_processed # Already a DataFrame
        else:
             # Manual DataFrame construction if set_output failed
            feature_names = preprocessor.get_feature_names_out()
            print(f"Manually creating DataFrame with columns: {feature_names}")
            # Need to handle potential sparse matrix output from older ColumnTransformer versions
            try:
                 X_processed_df = pd.DataFrame(X_processed.toarray(), columns=feature_names, index=X.index)
            except AttributeError: # If output wasn't sparse
                 X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)

        print(f"Processed features (X_processed_df) shape: {X_processed_df.shape}")
        print(f"Processed feature columns: {X_processed_df.columns.tolist()}")

    except Exception as e:
        print(f"ERROR: Failed during preprocessing transformation. {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for unexpected errors
        raise

    # --- Combine Final DataFrame ---
    try:
        print("Combining processed features, sensitive attribute, and target...")
        # Ensure indices align before concatenating - reindex just in case
        processed_df = pd.concat([
            X_processed_df,
            sensitive_data.reindex(X_processed_df.index), # Add original sensitive column
            y.reindex(X_processed_df.index)              # Add original target column
        ], axis=1)
        print(f"Final processed DataFrame shape: {processed_df.shape}")
        print(f"Final columns: {processed_df.columns.tolist()}") # Verify SENSITIVE_COLUMN and TARGET_COLUMN are present!
    except Exception as e:
        print(f"ERROR: Failed combining final DataFrame. {type(e).__name__}: {e}")
        raise

    # --- Save Processed Data ---
    print(f"Saving processed data as {OUTPUT_FORMAT}...")
    try:
        out_buffer = BytesIO()
        if OUTPUT_FORMAT == 'parquet':
            processed_df.to_parquet(out_buffer, index=False)
        elif OUTPUT_FORMAT == 'csv':
            processed_df.to_csv(out_buffer, index=False)
        elif OUTPUT_FORMAT == 'feather':
            processed_df.to_feather(out_buffer) # Requires pyarrow
        else:
            raise ValueError(f"Unsupported OUTPUT_FORMAT: {OUTPUT_FORMAT}")

        out_buffer.seek(0)
        s3_client.put_object(Bucket=out_bucket, Key=out_key, Body=out_buffer)
        print(f"Processed data saved successfully to s3://{out_bucket}/{out_key}")
    except Exception as e:
        print(f"ERROR: Failed to save processed data to S3. {type(e).__name__}: {e}")
        raise

    print(f"--- Preprocessing Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Arguments remain the same - only paths
    parser.add_argument('--in_bucket', type=str, required=True)
    parser.add_argument('--in_key', type=str, required=True)
    parser.add_argument('--out_bucket', type=str, required=True)
    parser.add_argument('--out_key', type=str, required=True)
    args = parser.parse_args()

    # Get S3 credentials from env vars
    s3_endpoint = os.environ.get("S3_ENDPOINT_URL")
    s3_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    s3_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not all([s3_endpoint, s3_access_key, s3_secret_key]):
        raise ValueError("S3 credentials or endpoint not found.")

    preprocess_data(args.in_bucket, args.in_key, args.out_bucket, args.out_key,
                    s3_endpoint, s3_access_key, s3_secret_key)
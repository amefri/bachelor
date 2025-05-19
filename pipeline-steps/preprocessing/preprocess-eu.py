import argparse
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import boto3
from io import StringIO, BytesIO
import sys
import traceback

TARGET_COLUMN = 'OBS_VALUE'
SENSITIVE_COLUMN = 'sex'
NUMERICAL_FEATURES = ['TIME_PERIOD']
CATEGORICAL_FEATURES = ['freq', 'age', 'unit', 'geo']
DROP_COLS_AFTER_LOAD = ['DATAFLOW', 'LAST UPDATE', 'OBS_FLAG', 'CONF_STATUS']
IMPUTER_STRATEGY_NUM = 'median'
SCALE_NUMERICAL = False
OHE_HANDLE_UNKNOWN = 'ignore'
INPUT_FORMAT = 'csv'
INPUT_ENCODING = 'utf-8'
INPUT_DELIMITER = ','
OUTPUT_FORMAT = 'parquet'

def preprocess_data(in_bucket, in_key, out_bucket, out_key, endpoint_url, access_key, secret_key):
    print(f"--- Starting Preprocessing ---")
    print(f"Input: s3://{in_bucket}/{in_key} ({INPUT_FORMAT}, encoding={INPUT_ENCODING}, delimiter='{INPUT_DELIMITER}')")
    print(f"Output: s3://{out_bucket}/{out_key} (format={OUTPUT_FORMAT})")
    print(f"Target Column: {TARGET_COLUMN}")
    print(f"Sensitive Column: {SENSITIVE_COLUMN}")
    print(f"Numerical Features: {NUMERICAL_FEATURES}")
    print(f"Categorical Features: {CATEGORICAL_FEATURES}")
    print(f"Drop Columns After Load: {DROP_COLS_AFTER_LOAD}")

    s3_client = boto3.client(
        "s3", endpoint_url=endpoint_url, aws_access_key_id=access_key, aws_secret_access_key=secret_key
    )

    try:
        print("Loading data from S3...")
        obj = s3_client.get_object(Bucket=in_bucket, Key=in_key)
        raw_body = obj['Body'].read()

        if INPUT_FORMAT == 'csv':
            df = pd.read_csv(BytesIO(raw_body),
                             encoding=INPUT_ENCODING,
                             delimiter=INPUT_DELIMITER)
        else:
            raise ValueError(f"Unsupported INPUT_FORMAT: {INPUT_FORMAT}")

        print(f"Input data loaded. Shape: {df.shape}")
        print(f"Input columns: {df.columns.tolist()}")

    except Exception as e:
        print(f"ERROR: Failed to load data from S3. {type(e).__name__}: {e}")
        traceback.print_exc()
        raise

    try:
        if DROP_COLS_AFTER_LOAD:
            cols_present_to_drop = [col for col in DROP_COLS_AFTER_LOAD if col in df.columns]
            if cols_present_to_drop:
                df = df.drop(columns=cols_present_to_drop)
                print(f"Dropped columns: {cols_present_to_drop}")
            else:
                print(f"No columns from DROP_COLS_AFTER_LOAD found to drop.")

        all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        essential_cols = [TARGET_COLUMN] + ([SENSITIVE_COLUMN] if SENSITIVE_COLUMN else []) + all_features
        missing_essential = [col for col in essential_cols if col not in df.columns]
        if missing_essential:
            raise ValueError(f"Missing essential columns after initial drop: {missing_essential}. Available columns: {df.columns.tolist()}")
        print("All essential columns present.")

        y = df[TARGET_COLUMN]
        sensitive_data = pd.DataFrame(index=df.index)
        if SENSITIVE_COLUMN and SENSITIVE_COLUMN in df.columns:
             sensitive_data = df[[SENSITIVE_COLUMN]]
             print(f"Separated sensitive column: {SENSITIVE_COLUMN}")
        elif SENSITIVE_COLUMN:
             print(f"Warning: SENSITIVE_COLUMN '{SENSITIVE_COLUMN}' defined but not found in data after drops.")
        else:
             print("No SENSITIVE_COLUMN defined.")

        cols_to_exclude = [TARGET_COLUMN] + ([SENSITIVE_COLUMN] if SENSITIVE_COLUMN else [])
        feature_cols = [col for col in df.columns if col not in cols_to_exclude]
        active_feature_cols = [col for col in feature_cols if col in all_features]
        X = df[active_feature_cols]

        print(f"Features (X) shape: {X.shape}. Columns: {X.columns.tolist()}")
        print(f"Target (y) shape: {y.shape}. Name: {y.name}")
        if SENSITIVE_COLUMN and not sensitive_data.empty:
            print(f"Sensitive data shape: {sensitive_data.shape}. Column: {sensitive_data.columns.tolist()}")

    except Exception as e:
        print(f"ERROR: Failed during initial column handling. {type(e).__name__}: {e}")
        traceback.print_exc()
        raise

    try:
        print("Defining preprocessing pipelines...")
        active_numerical_cols = [col for col in NUMERICAL_FEATURES if col in X.columns]
        active_categorical_cols = [col for col in CATEGORICAL_FEATURES if col in X.columns]
        print(f" Active Numerical Features for Transform: {active_numerical_cols}")
        print(f" Active Categorical Features for Transform: {active_categorical_cols}")

        numerical_steps = [('imputer_num', SimpleImputer(strategy=IMPUTER_STRATEGY_NUM))]
        if SCALE_NUMERICAL:
            numerical_steps.append(('scaler', StandardScaler()))
        numerical_transformer = Pipeline(steps=numerical_steps)

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown=OHE_HANDLE_UNKNOWN, sparse_output=False))
        ])

        transformers_list = []
        if active_numerical_cols:
             transformers_list.append(('num', numerical_transformer, active_numerical_cols))
        if active_categorical_cols:
             transformers_list.append(('cat', categorical_transformer, active_categorical_cols))

        if not transformers_list:
             print("Warning: No numerical or categorical features found/specified for transformation.")
             preprocessor = ColumnTransformer(transformers=[], remainder='passthrough')
        else:
            preprocessor = ColumnTransformer(
                transformers=transformers_list,
                remainder='drop',
                verbose_feature_names_out=False
            )

        try:
            preprocessor.set_output(transform="pandas")
            print("Set ColumnTransformer output to pandas DataFrame.")
            use_pandas_output = True
        except AttributeError:
            print("Warning: scikit-learn version < 1.2. Cannot set output to pandas. Will construct DataFrame manually.")
            use_pandas_output = False

        print("Applying preprocessing transformations to X...")
        if X.empty and not transformers_list:
             print("Input X is empty and no transformations defined. Creating empty processed DataFrame.")
             X_processed_df = pd.DataFrame(index=X.index)
        elif X.empty and transformers_list:
             print("Warning: Input X is empty but transformations were defined. Result will be empty.")
             try:
                 dummy_X = pd.DataFrame(columns=X.columns, data=[[pd.NA]*len(X.columns)])
                 preprocessor.fit(dummy_X)
                 feature_names = preprocessor.get_feature_names_out()
                 X_processed_df = pd.DataFrame(columns=feature_names, index=X.index)
             except Exception as fit_err:
                  print(f"Could not determine output columns on empty X. Error: {fit_err}")
                  X_processed_df = pd.DataFrame(index=X.index)
        else:
            X_processed = preprocessor.fit_transform(X)
            if use_pandas_output:
                X_processed_df = X_processed
            else:
                feature_names = preprocessor.get_feature_names_out()
                print(f"Manually creating DataFrame with columns: {feature_names}")
                try:
                    X_array = X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed
                except AttributeError:
                    X_array = X_processed
                X_processed_df = pd.DataFrame(X_array, columns=feature_names, index=X.index)

        print(f"Processed features (X_processed_df) shape: {X_processed_df.shape}")
        if not X_processed_df.empty:
            print(f"Processed feature columns: {X_processed_df.columns.tolist()}")
        else:
            print("Processed feature DataFrame is empty.")

    except Exception as e:
        print(f"ERROR: Failed during preprocessing transformation. {type(e).__name__}: {e}")
        traceback.print_exc()
        raise

    try:
        print("Combining processed features, sensitive attribute, and target...")
        final_df_parts = [X_processed_df]
        if SENSITIVE_COLUMN and not sensitive_data.empty:
             final_df_parts.append(sensitive_data.reindex(X_processed_df.index))
        final_df_parts.append(y.reindex(X_processed_df.index))
        processed_df = pd.concat(final_df_parts, axis=1)
        print(f"Final processed DataFrame shape: {processed_df.shape}")
        print(f"Final columns: {processed_df.columns.tolist()}")
        if TARGET_COLUMN not in processed_df.columns:
            print(f"WARNING: Target column '{TARGET_COLUMN}' is missing from the final DataFrame!")
        if SENSITIVE_COLUMN and SENSITIVE_COLUMN not in processed_df.columns:
             print(f"WARNING: Sensitive column '{SENSITIVE_COLUMN}' is missing from the final DataFrame!")

    except Exception as e:
        print(f"ERROR: Failed combining final DataFrame. {type(e).__name__}: {e}")
        traceback.print_exc()
        raise

    print(f"Saving processed data as {OUTPUT_FORMAT}...")
    try:
        out_buffer = BytesIO()
        if OUTPUT_FORMAT == 'parquet':
            processed_df.to_parquet(out_buffer, index=False)
        elif OUTPUT_FORMAT == 'csv':
            processed_df.to_csv(out_buffer, index=False)
        elif OUTPUT_FORMAT == 'feather':
            processed_df.to_feather(out_buffer)
        else:
            raise ValueError(f"Unsupported OUTPUT_FORMAT: {OUTPUT_FORMAT}")

        out_buffer.seek(0)
        s3_client.put_object(Bucket=out_bucket, Key=out_key, Body=out_buffer)
        print(f"Processed data saved successfully to s3://{out_bucket}/{out_key}")
    except Exception as e:
        print(f"ERROR: Failed to save processed data to S3. {type(e).__name__}: {e}")
        traceback.print_exc()
        raise

    print(f"--- Preprocessing Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_bucket', type=str, required=True)
    parser.add_argument('--in_key', type=str, required=True)
    parser.add_argument('--out_bucket', type=str, required=True)
    parser.add_argument('--out_key', type=str, required=True)
    args = parser.parse_args()

    s3_endpoint = os.environ.get("S3_ENDPOINT_URL")
    s3_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    s3_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not all([s3_endpoint, s3_access_key, s3_secret_key]):
        raise ValueError("S3 credentials or endpoint not found in environment variables.")

    preprocess_data(args.in_bucket, args.in_key, args.out_bucket, args.out_key,
                    s3_endpoint, s3_access_key, s3_secret_key)

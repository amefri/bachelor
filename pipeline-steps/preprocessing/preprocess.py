import argparse
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import boto3
from io import StringIO, BytesIO

# --- Configuration Variables ---
# Column Names & Types 

TARGET_COLUMN = 'DS_VWD'
# Explicitly list columns
NUMERICAL_COLS = ['Datum', 'RGSCode']
CATEGORICAL_COLS = ['Geschlecht']


# Preprocessing Parameters
IMPUTER_STRATEGY_NUM = 'median'
IMPUTER_STRATEGY_CAT = 'most_frequent' # Used if categorical imputation is needed
SCALE_NUMERICAL = False # Set to True to add StandardScaler
OHE_HANDLE_UNKNOWN = 'ignore' # 'error' or 'ignore'
DROP_COLS_BEFORE_PROCESSING = [] # List columns to drop early, e.g., IDs

# Input/Output Format
INPUT_FORMAT = 'csv' # Should match the *raw* data format read
OUTPUT_FORMAT = 'parquet' # 'parquet', 'csv', 'feather'
# --- End Configuration ---

def preprocess_data(in_bucket, in_key, out_bucket, out_key, endpoint_url, access_key, secret_key):
  
    print(f"Starting preprocessing for s3://{in_bucket}/{in_key}")
    print(f"Outputting to s3://{out_bucket}/{out_key} in format {OUTPUT_FORMAT}")
    s3_client = boto3.client(
        "s3", endpoint_url=endpoint_url, aws_access_key_id=access_key, aws_secret_access_key=secret_key
    )

    # --- Load Data ---
    obj = s3_client.get_object(Bucket=in_bucket, Key=in_key)
    if INPUT_FORMAT == 'csv':
        df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
   
    else:
         raise ValueError(f"Unsupported INPUT_FORMAT: {INPUT_FORMAT}")
    print("Input data loaded.")

    # --- Preprocessing Logic using Configuration ---
  
    if DROP_COLS_BEFORE_PROCESSING:
        df = df.drop(columns=DROP_COLS_BEFORE_PROCESSING, errors='ignore')
        print(f"Dropped columns: {DROP_COLS_BEFORE_PROCESSING}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in input data for preprocessing.")

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    # Use explicit column lists defined above
    active_numerical_cols = [col for col in NUMERICAL_COLS if col in X.columns]
    active_categorical_cols = [col for col in CATEGORICAL_COLS if col in X.columns]
    print(f"Processing Numerical: {active_numerical_cols}")
    print(f"Processing Categorical: {active_categorical_cols}")

    # Create preprocessing pipelines
    numerical_steps = [('imputer', SimpleImputer(strategy=IMPUTER_STRATEGY_NUM))]
    if SCALE_NUMERICAL:
        numerical_steps.append(('scaler', StandardScaler()))
    numerical_transformer = Pipeline(steps=numerical_steps)

    categorical_transformer = Pipeline(steps=[
       
        ('onehot', OneHotEncoder(handle_unknown=OHE_HANDLE_UNKNOWN))
    ])

    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, active_numerical_cols),
            ('cat', categorical_transformer, active_categorical_cols)
        ],
        remainder='passthrough' # Keep columns not listed in numerical/categorical
    )

    # Apply transformations
    X_processed = preprocessor.fit_transform(X)
    print("Preprocessing transformations applied.")

    # Get feature names
    try:
        ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(active_categorical_cols)
    except AttributeError: 
         ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(active_categorical_cols)

    # Handle remainder='passthrough' columns
    passthrough_features = []
    if hasattr(preprocessor, 'get_feature_names_out'): # Newer scikit-learn
        all_feature_names = preprocessor.get_feature_names_out()
    else: # Manual construction for older versions or more control
        all_processed_cols = list(active_numerical_cols) + list(ohe_feature_names)
        if preprocessor.remainder == 'passthrough':
             # Figure out which columns were passed through
             processed_indices = set()
             for name, trans, cols in preprocessor.transformers_:
                 if name != 'remainder':
                     try:
                         processed_indices.update(preprocessor._feature_indices[name])
                     except AttributeError: # Older sklearn might need index lookup differently
                          indices = [X.columns.get_loc(c) for c in cols]
                          processed_indices.update(indices)

             passthrough_indices = [i for i in range(len(X.columns)) if i not in processed_indices]
             passthrough_features = X.columns[passthrough_indices].tolist()
        all_feature_names = all_processed_cols + passthrough_features

    # Convert back to DataFrame
    try:
        X_processed_df = pd.DataFrame(X_processed.toarray(), columns=all_feature_names)
    except AttributeError:
        X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names)

    # Combine processed features with target
    processed_df = pd.concat([X_processed_df, y.reset_index(drop=True)], axis=1)
    print("Preprocessing complete.")
    print(f"Processed data shape: {processed_df.shape}")
    print(f"Processed columns: {processed_df.columns.tolist()}")


    # --- Save Processed Data using Configuration ---
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
    print(f"Processed data saved to s3://{out_bucket}/{out_key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Pass only paths, let internal config handle columns/formats
    parser.add_argument('--in_bucket', type=str, required=True)
    parser.add_argument('--in_key', type=str, required=True)
    parser.add_argument('--out_bucket', type=str, required=True)
    parser.add_argument('--out_key', type=str, required=True)
    args = parser.parse_args()

    # ... (Get S3 credentials from env vars) ...
    s3_endpoint = os.environ.get("S3_ENDPOINT_URL")
    s3_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    s3_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not all([s3_endpoint, s3_access_key, s3_secret_key]):
        raise ValueError("S3 credentials or endpoint not found.")

    preprocess_data(args.in_bucket, args.in_key, args.out_bucket, args.out_key,
                    s3_endpoint, s3_access_key, s3_secret_key)
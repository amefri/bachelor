import argparse
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Keep StandardScaler if you might add it
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import boto3
from io import StringIO, BytesIO

def preprocess_data(in_bucket, in_key, out_bucket, out_key, endpoint_url, access_key, secret_key):
    print(f"Starting preprocessing for s3://{in_bucket}/{in_key}")
    print(f"Outputting to s3://{out_bucket}/{out_key}")
    s3_client = boto3.client(
        "s3", endpoint_url=endpoint_url, aws_access_key_id=access_key, aws_secret_access_key=secret_key
    )

    try:
        obj = s3_client.get_object(Bucket=in_bucket, Key=in_key)
        df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
        print("Input data loaded.")
    except Exception as e:
        print(f"Error loading data from S3: {e}")
        raise # Re-raise the exception to fail the step

    # --- Preprocessing Logic ---
    target_col = 'target' # Define target column name
    sensitive_col = 'sensitive_attr' # Define sensitive column name

    # Check if essential columns exist *before* processing
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in input data. Columns: {df.columns.tolist()}")
    if sensitive_col not in df.columns:
        raise ValueError(f"Sensitive column '{sensitive_col}' not found in input data. Columns: {df.columns.tolist()}")

    # Separate target and sensitive columns FIRST
    y = df[target_col]
    sensitive_data = df[[sensitive_col]] # Keep sensitive column separate (as a DataFrame)

    # X contains only the features to be potentially transformed
    X = df.drop([target_col, sensitive_col], axis=1)

    # Identify column types *within X*
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    # EXCLUDE sensitive column - it's already removed from X
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    print(f"Numerical columns to process in X: {numerical_cols.tolist()}")
    print(f"Categorical columns to process in X: {categorical_cols.tolist()}")

  
    numerical_transformer = SimpleImputer(strategy='median') # Simpler if just imputing

    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # sparse_output=False often easier

    # Create a column transformer *only for features in X*
    # Use remainder='passthrough' to keep any columns in X not explicitly handled
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough', # Keeps columns in X that are neither num nor cat
        verbose_feature_names_out=False # Often gives cleaner column names
    )
    # Use set_output(transform="pandas") if scikit-learn >= 1.2 for direct DataFrame output
    try:
        preprocessor.set_output(transform="pandas")
    except AttributeError:
        print("Warning: set_output not available. Will construct DataFrame manually.")
        pass # Continue without direct pandas output


    print("Fitting and transforming feature set X...")
    # Apply transformations *only to X*
    try:
        X_processed_intermediate = preprocessor.fit_transform(X)

        # Check if output is already a DataFrame (due to set_output)
        if isinstance(X_processed_intermediate, pd.DataFrame):
            X_processed_df = X_processed_intermediate
        else:
            # Construct DataFrame manually if set_output wasn't used/available
            print("Constructing DataFrame manually from transformer output.")
            feature_names = preprocessor.get_feature_names_out()
            X_processed_df = pd.DataFrame(X_processed_intermediate, columns=feature_names, index=X.index) # Ensure index is preserved

        print("Transformation successful.")
        print(f"Columns after transformation of X: {X_processed_df.columns.tolist()}")

    except Exception as e:
        print(f"Error during preprocessing transformation: {e}")
        # Print shapes for debugging
        print(f"Shape of X: {X.shape}")
        if 'X_processed_intermediate' in locals():
             print(f"Output shape of transformer: {X_processed_intermediate.shape}")
        raise

    # --- Combine processed features, original sensitive attr, and target ---
    # Use original index to ensure alignment.
    print("Combining processed features, sensitive attribute, and target...")
    try:
        # Ensure indices match before concatenating, reindex if needed (should match if preserved above)
        processed_df = pd.concat([X_processed_df, sensitive_data.reindex(X_processed_df.index), y.reindex(X_processed_df.index)], axis=1)
    except Exception as e:
        print(f"Error concatenating final DataFrame: {e}")
        # Print shapes for debugging
        print(f"Shape of X_processed_df: {X_processed_df.shape}")
        print(f"Shape of sensitive_data: {sensitive_data.shape}")
        print(f"Shape of y: {y.shape}")
        # Check indices if error occurs
        print(f"Index of X_processed_df (first 5): {X_processed_df.index[:5]}")
        print(f"Index of sensitive_data (first 5): {sensitive_data.index[:5]}")
        print(f"Index of y (first 5): {y.index[:5]}")
        raise

    print("Preprocessing complete.")
    print(f"Final processed data shape: {processed_df.shape}")
    print(f"Final processed columns: {processed_df.columns.tolist()}") # Verify sensitive_attr is present

    # Save processed data to MinIO (using Parquet for efficiency)
    print(f"Saving processed data to s3://{out_bucket}/{out_key}...")
    try:
        out_buffer = BytesIO()
        processed_df.to_parquet(out_buffer, index=False)
        out_buffer.seek(0)
        s3_client.put_object(Bucket=out_bucket, Key=out_key, Body=out_buffer)
        print(f"Processed data saved successfully.")
    except Exception as e:
        print(f"Error saving processed data to S3: {e}")
        raise

# --- End of preprocess_data function ---

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

    # Call the preprocessing function
    preprocess_data(args.in_bucket, args.in_key, args.out_bucket, args.out_key,
                    s3_endpoint, s3_access_key, s3_secret_key)
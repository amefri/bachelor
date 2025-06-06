import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import boto3
from io import BytesIO

def split_data(in_bucket, in_key, out_bucket, train_key, test_key, val_key, target_col, test_size, val_size, random_state, endpoint_url, access_key, secret_key):
    print(f"Starting data splitting for s3://{in_bucket}/{in_key}")
    print(f"Output train: s3://{out_bucket}/{train_key}")
    print(f"Output test: s3://{out_bucket}/{test_key}")
    print(f"Output val: s3://{out_bucket}/{val_key}")

    s3_client = boto3.client(
        "s3", endpoint_url=endpoint_url, aws_access_key_id=access_key, aws_secret_access_key=secret_key
    )

    obj = s3_client.get_object(Bucket=in_bucket, Key=in_key)
    in_buffer = BytesIO(obj['Body'].read())
    df = pd.read_parquet(in_buffer)
    print("Processed data loaded.")

    # Split into features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # First split: train and temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=random_state
    )

    # Second split: val and test
    if X_temp.shape[0] > 1:
        relative_test_size = test_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=relative_test_size, random_state=random_state
        )
    elif X_temp.shape[0] == 1:
        print("Only 1 sample in temp split. Assigning to validation set.")
        X_val, X_test = X_temp, pd.DataFrame(columns=X_temp.columns)
        y_val, y_test = y_temp, pd.Series(dtype=y_temp.dtype)
    else:
        print("No samples in temp split.")
        X_val, X_test = pd.DataFrame(columns=X.columns), pd.DataFrame(columns=X.columns)
        y_val, y_test = pd.Series(dtype=y.dtype), pd.Series(dtype=y.dtype)

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    # Combine features and target
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Optional: sample down large splits
    MAX_TRAIN_ROWS = 5000
    MAX_VAL_ROWS = 2000
    MAX_TEST_ROWS = 2000

    train_df = train_df.sample(n=min(MAX_TRAIN_ROWS, len(train_df)), random_state=random_state)
    val_df = val_df.sample(n=min(MAX_VAL_ROWS, len(val_df)), random_state=random_state)
    test_df = test_df.sample(n=min(MAX_TEST_ROWS, len(test_df)), random_state=random_state)

    print(f"Sampled: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Save each split
    for df_split, key in zip([train_df, val_df, test_df], [train_key, val_key, test_key]):
        if not df_split.empty:
            out_buffer = BytesIO()
            df_split.to_parquet(out_buffer, index=False)
            out_buffer.seek(0)
            s3_client.put_object(Bucket=out_bucket, Key=key, Body=out_buffer)
            print(f"Saved s3://{out_bucket}/{key}")
        else:
            print(f"Skipped empty split for s3://{out_bucket}/{key}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_bucket', type=str, required=True)
    parser.add_argument('--in_key', type=str, required=True)
    parser.add_argument('--out_bucket', type=str, required=True)
    parser.add_argument('--train_key', type=str, required=True)
    parser.add_argument('--test_key', type=str, required=True)
    parser.add_argument('--val_key', type=str, required=True)
    parser.add_argument('--target_col', type=str, required=True)
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--val_size', type=float, default=0.15)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    s3_endpoint = os.environ.get("S3_ENDPOINT_URL")
    s3_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    s3_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    if not all([s3_endpoint, s3_access_key, s3_secret_key]):
        raise ValueError("S3 credentials or endpoint not found.")

    split_data(args.in_bucket, args.in_key, args.out_bucket,
               args.train_key, args.test_key, args.val_key,
               args.target_col, args.test_size, args.val_size, args.random_state,
               s3_endpoint, s3_access_key, s3_secret_key)

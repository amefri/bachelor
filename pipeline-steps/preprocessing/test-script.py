import os
import boto3
import pandas as pd
from io import StringIO
from preprocess import preprocess_data

def setup_mock_s3():
    # Mock S3 credentials (for local MinIO or test S3 environment)
    os.environ["S3_ENDPOINT_URL"] = "http://localhost:9000"  # Adjust as needed
    os.environ["AWS_ACCESS_KEY_ID"] = "user"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "password"

    s3_client = boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
    )
    
    return s3_client

def upload_mock_data(s3_client, bucket, key):
    # Create a test dataset
    data = """feature1,feature2,target\n1,A,0\n2,B,1\n3,A,0\n4,C,1"""
    
    # Ensure bucket exists
    try:
        s3_client.create_bucket(Bucket=bucket)
    except:
        pass  # Bucket might already exist
    
    s3_client.put_object(Bucket=bucket, Key=key, Body=data)
    print(f"Uploaded mock data to s3://{bucket}/{key}")

def test_preprocess():
    s3_client = setup_mock_s3()
    in_bucket = "test-bucket"
    in_key = "input/test_data.csv"
    out_bucket = "test-bucket"
    out_key = "output/processed.parquet"

    upload_mock_data(s3_client, in_bucket, in_key)
    
    preprocess_data(in_bucket, in_key, out_bucket, out_key,
                    os.environ["S3_ENDPOINT_URL"],
                    os.environ["AWS_ACCESS_KEY_ID"],
                    os.environ["AWS_SECRET_ACCESS_KEY"])
    
    # Check if processed file exists
    response = s3_client.list_objects_v2(Bucket=out_bucket, Prefix=out_key)
    assert "Contents" in response, "Processed file was not created."
    print("Test passed: Processed file exists.")

if __name__ == "__main__":
    test_preprocess()

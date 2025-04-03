import boto3
import pandas as pd
from io import StringIO
import botocore.exceptions
from validate import validate_data  

# MinIO Configuration
MINIO_ENDPOINT = "http://localhost:9000"
MINIO_ACCESS_KEY = "user"
MINIO_SECRET_KEY = "password"
TEST_BUCKET = "test-bucket"
TEST_KEY = "test-data.csv"
TEST_DF_URL = "/home/amelie/k8s-ml-pipeline/sample_data/Abgang_AL_Geschlecht_Altersgruppen_VWD_RGS.csv"

# Create an S3 client for MinIO
s3_client = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY
)

# Ensure the test bucket exists
try:
    s3_client.create_bucket(Bucket=TEST_BUCKET)
    print(f"Bucket '{TEST_BUCKET}' created successfully.")
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
        print(f"Bucket '{TEST_BUCKET}' already exists. Skipping creation.")
    else:
        raise 

# Upload  test CSV file
csv_content = pd.read_csv(TEST_DF_URL, encoding = "latin1", sep=";" )
print(f"pdf gelesen")


# Convert DataFrame to CSV format
csv_buffer = StringIO()
csv_content.to_csv(csv_buffer, index=False)  # Convert DataFrame to CSV string
csv_binary = csv_buffer.getvalue().encode("utf-8")  # Encode as bytes
print(f"Encoded")
# Upload to MinIO
s3_client.put_object(Bucket=TEST_BUCKET, Key=TEST_KEY, Body=csv_binary)
print(f"Uploaded test file '{TEST_KEY}' to bucket '{TEST_BUCKET}'.")

# run funkiton
validate_data(TEST_BUCKET, TEST_KEY, MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY)

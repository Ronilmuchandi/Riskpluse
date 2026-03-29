import boto3
import os
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

bucket = os.getenv('AWS_BUCKET_NAME')

files_to_upload = [
    ('data/raw/creditcard.csv', 'data/raw/creditcard.csv'),
    ('data/processed/scaler.pkl', 'data/processed/scaler.pkl'),
]

for local_path, s3_path in files_to_upload:
    print(f"Uploading {local_path}...")
    s3.upload_file(local_path, bucket, s3_path)
    print(f"Done — s3://{bucket}/{s3_path}")

print("\nAll files uploaded to S3.")
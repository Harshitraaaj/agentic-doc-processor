from __future__ import annotations

from typing import Optional

import boto3

from services.storage_service import StorageService
from utils.config import settings


class S3StorageAdapter(StorageService):
    """S3-backed storage adapter for cloud profile."""

    def __init__(self, bucket_name: Optional[str] = None, region: Optional[str] = None):
        self.bucket_name = bucket_name or settings.AWS_S3_BUCKET
        if not self.bucket_name:
            raise ValueError("S3 bucket not configured. Set [aws].s3_bucket in config.ini")
        self.region = region or settings.AWS_REGION
        self.client = boto3.client("s3", region_name=self.region)

    def put_file(self, key: str, data: bytes, content_type: Optional[str] = None) -> str:
        extra_args = {"ContentType": content_type} if content_type else {}
        self.client.put_object(Bucket=self.bucket_name, Key=key, Body=data, **extra_args)
        return f"s3://{self.bucket_name}/{key}"

    def get_file(self, key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket_name, Key=key)
        return response["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except Exception:
            return False

    def get_signed_url(self, key: str, expires_seconds: int = 900) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket_name, "Key": key},
            ExpiresIn=expires_seconds,
        )

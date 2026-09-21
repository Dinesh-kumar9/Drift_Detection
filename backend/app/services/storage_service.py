"""
MinIO / S3 storage abstraction.
In local dev, S3_ENDPOINT_URL points to MinIO.
In production, unset S3_ENDPOINT_URL to use real AWS S3.
"""

import io
import logging
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from app.core.config import settings

logger = logging.getLogger(__name__)


def _get_client():
    kwargs = dict(
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
        region_name=settings.S3_REGION,
    )
    if settings.S3_ENDPOINT_URL:
        kwargs["endpoint_url"] = settings.S3_ENDPOINT_URL
    return boto3.client("s3", **kwargs)


def upload_bytes(bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    """Upload raw bytes and return the S3 URI."""
    client = _get_client()
    _ensure_bucket(client, bucket)
    client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    logger.info("Uploaded s3://%s/%s (%d bytes)", bucket, key, len(data))
    return f"s3://{bucket}/{key}"


def upload_file(bucket: str, key: str, file_path: str) -> str:
    """Upload a file from disk."""
    client = _get_client()
    _ensure_bucket(client, bucket)
    client.upload_file(file_path, bucket, key)
    return f"s3://{bucket}/{key}"


def download_bytes(bucket: str, key: str) -> bytes:
    """Download object as bytes."""
    client = _get_client()
    response = client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


def download_to_buffer(bucket: str, key: str) -> io.BytesIO:
    """Download object into an in-memory BytesIO buffer."""
    data = download_bytes(bucket, key)
    return io.BytesIO(data)


def object_exists(bucket: str, key: str) -> bool:
    client = _get_client()
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def _ensure_bucket(client, bucket: str) -> None:
    """Create bucket if it doesn't exist (idempotent, local MinIO only)."""
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError:
        try:
            client.create_bucket(Bucket=bucket)
            logger.info("Created bucket: %s", bucket)
        except ClientError as e:
            if e.response["Error"]["Code"] not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                raise

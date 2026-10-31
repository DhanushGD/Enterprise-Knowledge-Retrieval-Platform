"""S3-compatible object storage. Same code runs on MinIO, AWS S3, R2 or B2."""
from __future__ import annotations

from core.settings import settings


def get_s3_client():
    import boto3
    from botocore.client import Config

    endpoint = settings.s3_endpoint_url or None
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
        # MinIO needs path-style URLs (host/bucket/key). Real AWS does not.
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path" if endpoint else "auto"},
        ),
    )


def ensure_bucket(client=None, bucket: str | None = None) -> bool:
    """Create the bucket if missing. Returns True if it had to be created.

    Replaces the old `minio-init` container, so it works with any MinIO image
    and any S3 provider.
    """
    from botocore.exceptions import ClientError

    client = client or get_s3_client()
    bucket = bucket or settings.s3_bucket
    try:
        client.head_bucket(Bucket=bucket)
        return False
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))
        if code not in ("404", "NoSuchBucket", "NotFound"):
            raise  # 403 etc: wrong credentials or someone else's bucket
    params = {"Bucket": bucket}
    if settings.s3_region != "us-east-1" and not settings.s3_endpoint_url:
        params["CreateBucketConfiguration"] = {"LocationConstraint": settings.s3_region}
    client.create_bucket(**params)
    return True


class RawStore:
    """Where original documents land, untouched, under raw/<source>/<path>."""

    def __init__(self, client=None, bucket: str | None = None):
        self.client = client or get_s3_client()
        self.bucket = bucket or settings.s3_bucket

    def ensure(self) -> None:
        ensure_bucket(self.client, self.bucket)

    def put(self, key: str, data: bytes, content_type: str | None = None) -> None:
        extra = {"ContentType": content_type} if content_type else {}
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, **extra)

    def get(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def delete(self, keys: list[str]) -> None:
        """Delete objects (missing keys are fine). S3 accepts at most 1000 per request."""
        for i in range(0, len(keys), 1000):
            self.client.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": [{"Key": k} for k in keys[i:i + 1000]], "Quiet": True},
            )

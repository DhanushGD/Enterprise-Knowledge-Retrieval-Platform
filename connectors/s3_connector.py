"""Reads documents from any S3-compatible bucket (MinIO, AWS S3, R2, B2).

This is the connector you point at "company specific data": upload a company's
files under a prefix, set `source.type: s3` in config/company.yaml, done.
"""
from __future__ import annotations

from datetime import timezone
from typing import Iterable, Iterator

from connectors.base import SourceConnector, SourceDocument
from connectors.local_folder import SUPPORTED_EXTENSIONS


class S3Connector(SourceConnector):
    def __init__(
        self,
        name: str,
        bucket: str,
        prefix: str = "",
        client=None,
        default_department: str | None = None,
        max_documents: int | None = None,
        extensions: Iterable[str] | None = None,
    ):
        if client is None:
            from core.storage import get_s3_client  # lazy: keeps boto3 optional for tests

            client = get_s3_client()
        self.name = name
        self.bucket = bucket
        self.prefix = prefix
        self.client = client
        self.default_department = default_department
        self.max_documents = max_documents
        self.extensions = {e.lower() for e in (extensions or SUPPORTED_EXTENSIONS)}

    def list_documents(self) -> Iterator[SourceDocument]:
        paginator = self.client.get_paginator("list_objects_v2")
        count = 0
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                if not any(key.lower().endswith(ext) for ext in self.extensions):
                    continue
                last_modified = obj["LastModified"]
                if last_modified.tzinfo is None:
                    last_modified = last_modified.replace(tzinfo=timezone.utc)
                yield SourceDocument(
                    source_name=self.name,
                    path=key[len(self.prefix):].lstrip("/"),
                    size=obj["Size"],
                    last_modified=last_modified,
                    etag=obj.get("ETag", "").strip('"') or None,
                    default_department=self.default_department,
                )
                count += 1
                if self.max_documents and count >= self.max_documents:
                    return

    def fetch(self, path: str) -> bytes:
        key = f"{self.prefix.rstrip('/')}/{path}" if self.prefix else path
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

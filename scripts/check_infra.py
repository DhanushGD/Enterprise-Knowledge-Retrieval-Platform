"""Step 2 verification: confirms every service is reachable.

Run:  python scripts/check_infra.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import boto3
import redis
import requests
from botocore.client import Config
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text

from core.settings import company, settings
from core.storage import ensure_bucket

GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"
results = []


def check(name, fn):
    try:
        detail = fn()
        print(f"{GREEN}OK  {RESET} {name}" + (f"  ({detail})" if detail else ""))
        results.append(True)
    except Exception as e:  # noqa: BLE001
        print(f"{RED}FAIL{RESET} {name}  -> {type(e).__name__}: {e}")
        results.append(False)


def postgres():
    engine = create_engine(settings.postgres_dsn)
    with engine.connect() as c:
        return c.execute(text("select version()")).scalar().split(",")[0]


def qdrant():
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    return f"{len(client.get_collections().collections)} collections"


def redis_check():
    r = redis.from_url(settings.redis_url)
    r.set("healthcheck", "1", ex=10)
    return "ping=" + str(r.ping())


def minio():
    s3 = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
        config=Config(signature_version="s3v4"),
    )
    created = ensure_bucket(s3, settings.s3_bucket)  # no init container needed
    s3.put_object(Bucket=settings.s3_bucket, Key="healthcheck.txt", Body=b"ok")
    return f"bucket '{settings.s3_bucket}' {'created, ' if created else ''}writable"


def prometheus():
    r = requests.get("http://localhost:9090/-/healthy", timeout=5)
    r.raise_for_status()


def grafana():
    r = requests.get("http://localhost:3000/api/health", timeout=5)
    r.raise_for_status()


print(f"\nCompany: {company['company']['name']}\n")
check("Postgres", postgres)
check("Qdrant", qdrant)
check("Redis", redis_check)
check("MinIO (S3)", minio)
check("Prometheus", prometheus)
check("Grafana", grafana)

print()
if all(results):
    print(f"{GREEN}All services healthy. Step 2 done.{RESET}")
else:
    print(f"{RED}Some services failed. Run `docker compose ps` and `docker compose logs <service>`.{RESET}")
    sys.exit(1)

# Airflow + a separate virtualenv for the project's own dependencies.
#
# Why a separate venv: Airflow pins its own versions of many libraries (SQLAlchemy, pydantic,
# ...). The ETL code needs different ones (fastembed, qdrant-client, SQLAlchemy 2). Installing
# both into one environment is a dependency fight; two environments never conflict.
#
# Build context is the project root:  docker compose -f docker-compose.airflow.yml build
FROM apache/airflow:3.1.8

USER root
# The git connector shells out to `git`.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements-etl.txt /tmp/requirements-etl.txt
RUN python -m venv /opt/airflow/ekrp-venv \
    && /opt/airflow/ekrp-venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/airflow/ekrp-venv/bin/pip install --no-cache-dir -r /tmp/requirements-etl.txt

# Created here (not by Docker at mount time) so the airflow user owns it: git clones and the
# embedding-model cache live in this folder, which compose mounts as a named volume.
RUN mkdir -p /opt/airflow/ekrp-data

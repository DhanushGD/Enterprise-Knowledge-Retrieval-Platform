"""Builds connectors from the `source:` section of config/company.yaml."""
from __future__ import annotations

from pathlib import Path

from connectors.base import SourceConnector
from connectors.local_folder import LocalFolderConnector


def build_connectors(
    source_cfg: dict, project_root: Path, clone_root: Path
) -> list[SourceConnector]:
    kind = source_cfg["type"]
    max_docs = source_cfg.get("max_documents")

    if kind == "local":
        cfg = source_cfg["local"]
        return [
            LocalFolderConnector(
                name=cfg["name"],
                root=project_root / cfg["path"],
                default_department=cfg.get("default_department"),
                max_documents=max_docs,
            )
        ]

    if kind in ("git", "github"):
        from connectors.git_repo import GitRepoConnector

        return [
            GitRepoConnector(
                name=repo["name"],
                url=repo["url"],
                clone_root=clone_root,
                subpath=repo.get("subpath", ""),
                default_department=repo.get("default_department"),
                max_documents=max_docs,
            )
            for repo in source_cfg["repos"]
        ]

    if kind == "s3":
        from connectors.s3_connector import S3Connector

        cfg = source_cfg["s3"]
        return [
            S3Connector(
                name=cfg["name"],
                bucket=cfg["bucket"],
                prefix=cfg.get("prefix", ""),
                default_department=cfg.get("default_department"),
                max_documents=max_docs,
            )
        ]

    raise ValueError(f"Unknown source.type '{kind}'. Use local, git or s3.")

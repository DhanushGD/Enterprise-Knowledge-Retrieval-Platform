"""Synthetic metadata: stands in for SharePoint permissions and properties.

Real SharePoint gives every file a department/site, sensitivity label and ACL.
Open-source docs have none, so we derive them deterministically from the path.
Deterministic matters: re-running ingestion must never reshuffle access levels.
"""
from __future__ import annotations

import hashlib
import re

# Path tokens that hint at a value. Matched on whole words, so "hr" never fires on "three".
DEPARTMENT_HINTS = {
    "support": "support", "product": "product",
    "hr": "hr", "people": "hr", "benefits": "hr", "recruiting": "hr",
    "engineering": "engineering", "infrastructure": "engineering", "sre": "engineering",
}
PRODUCT_HINTS = {
    "billing": "billing", "invoice": "billing", "invoices": "billing",
    "payment": "billing", "refund": "billing", "refunds": "billing",
    "integration": "integrations", "integrations": "integrations",
    "api": "integrations", "webhook": "integrations", "webhooks": "integrations",
    "security": "security", "auth": "security", "authentication": "security",
    "password": "security", "permissions": "security", "sso": "security",
}
ACCESS_HINTS = {
    "public": "public", "faq": "public",
    "internal": "internal", "private": "internal",
    "confidential": "restricted", "restricted": "restricted", "secret": "restricted",
}
# Fallback mix when no hint matches: mostly public, some internal, few restricted.
ACCESS_WEIGHTS = [("public", 60), ("internal", 30), ("restricted", 10)]


def _tokens(path: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", path.lower()) if t]


def _bucket(source_name: str, path: str, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}:{source_name}:{path}".encode()).hexdigest()
    return int(digest[:8], 16)


def _first_hint(tokens: list[str], hints: dict[str, str], allowed: list[str]) -> str | None:
    for token in tokens:
        value = hints.get(token)
        if value in allowed:
            return value
    return None


def assign_metadata(
    source_name: str, path: str, default_department: str | None, taxonomy: dict
) -> dict:
    tokens = _tokens(path)
    departments, levels, products = (
        taxonomy["department"], taxonomy["access_level"], taxonomy["product"]
    )

    department = (
        _first_hint(tokens, DEPARTMENT_HINTS, departments)
        or (default_department if default_department in departments else departments[0])
    )

    product = _first_hint(tokens, PRODUCT_HINTS, products) or products[
        _bucket(source_name, path, "product") % len(products)
    ]

    access = _first_hint(tokens, ACCESS_HINTS, levels)
    if access is None and department == "hr" and "restricted" in levels:
        access = "restricted"   # HR material is always locked down
    if access is None:
        roll = _bucket(source_name, path, "access") % 100
        cumulative = 0
        access = levels[0]
        for level, weight in ACCESS_WEIGHTS:
            cumulative += weight
            if roll < cumulative and level in levels:
                access = level
                break

    return {"department": department, "access_level": access, "product": product}

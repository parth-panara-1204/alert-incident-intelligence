"""HTTP client to push synthetic alerts to the backend ingest API."""

from __future__ import annotations

from typing import Callable, Mapping

import requests


def ingest_records(
    source: str,
    records: list[Mapping[str, object]],
    *,
    ingest_url: str | None,
    db_url: str | None,
    table: str,
    schema: str,
    timeout: float = 10.0,
    continue_on_error: bool = False,
    payload_encoder: Callable[[Mapping[str, object]], object] | None = None,
) -> None:
    """Send records to the backend ingest endpoint one by one."""
    if not ingest_url:
        return

    encoder = payload_encoder or (lambda rec: rec)
    session = requests.Session()

    for rec in records:
        body = {
            "source": source,
            "payload": encoder(rec),
            "db_url": db_url,
            "table": table,
            "target_schema": schema,
        }
        try:
            resp = session.post(ingest_url, json=body, timeout=timeout)
            resp.raise_for_status()
        except Exception as exc:  # pragma: no cover - network dependent
            msg = f"Ingest failed for {source} record: {exc}"
            if continue_on_error:
                print(msg)
                continue
            raise RuntimeError(msg) from exc

    print(f"Ingested {len(records)} {source} records to backend at {ingest_url}")

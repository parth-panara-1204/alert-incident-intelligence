from __future__ import annotations

import os
import re
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, text

REQUIRED_COLUMNS = ["source", "organization", "device", "alert_type", "severity", "timestamp"]
VALID_SOURCES = {"meraki", "auvik", "ncentral"}
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export "):].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


def normalize_postgres_url(db_url: str) -> str:
    if db_url.startswith("postgresql://"):
        return db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    if db_url.startswith("postgres://"):
        return db_url.replace("postgres://", "postgresql+psycopg://", 1)
    return db_url


def build_db_url(cli_db_url: str | None = None) -> str:
    """Return a SQLAlchemy URL.

    Priority: explicit cli -> DATABASE_URL -> PG* variables -> local sqlite fallback.
    The sqlite fallback keeps the app usable for demos when Postgres creds are missing.
    """

    if cli_db_url:
        return normalize_postgres_url(cli_db_url)

    env_db_url = os.getenv("DATABASE_URL")
    if env_db_url:
        return normalize_postgres_url(env_db_url)

    host = os.getenv("PGHOST")
    port = os.getenv("PGPORT", "5432")
    database = os.getenv("PGDATABASE")
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")

    required = {
        "PGHOST": host,
        "PGDATABASE": database,
        "PGUSER": user,
        "PGPASSWORD": password,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        fallback_path = Path(__file__).resolve().parent.parent / "local_alerts.db"
        return f"sqlite:///{fallback_path}"

    safe_password = quote_plus(password)
    return f"postgresql+psycopg://{user}:{safe_password}@{host}:{port}/{database}"


def _coerce_json_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError("JSON payload list must contain objects only.")
        return payload
    raise ValueError("JSON payload must be an object or list of objects.")


def parse_payload(source: str, payload: Any) -> list[dict[str, Any]]:
    source_key = source.strip().lower()
    if source_key not in VALID_SOURCES:
        raise ValueError(f"Unsupported source '{source}'. Expected one of: {sorted(VALID_SOURCES)}")

    alerts: list[dict[str, Any]] = []

    if source_key == "meraki":
        for alert in _coerce_json_records(payload):
            alerts.append(
                {
                    "source": "meraki",
                    "organization": alert.get("organizationName"),
                    "device": alert.get("deviceName"),
                    "alert_type": alert.get("alertType"),
                    "severity": alert.get("alertLevel"),
                    "timestamp": alert.get("occurredAt"),
                }
            )

    elif source_key == "auvik":
        for alert in _coerce_json_records(payload):
            alerts.append(
                {
                    "source": "auvik",
                    "organization": alert.get("companyName"),
                    "device": alert.get("entityName"),
                    "alert_type": alert.get("alertName"),
                    "severity": alert.get("alertSeverityString"),
                    "timestamp": alert.get("date"),
                }
            )

    elif source_key == "ncentral":
        if isinstance(payload, list):
            if not all(isinstance(item, str) for item in payload):
                raise ValueError("N-Central payload list must contain XML strings only.")
            payload = "\n".join(payload)

        if not isinstance(payload, str):
            raise ValueError("N-Central payload must be XML text or list of XML strings.")

        xml_data = re.sub(r"<\?xml.*?\?>", "", payload)
        xml_data = "<root>" + xml_data + "</root>"

        root = ET.fromstring(xml_data)
        for n in root.findall("notification"):
            alerts.append(
                {
                    "source": "ncentral",
                    "organization": n.findtext("CustomerName"),
                    "device": n.findtext("DeviceName"),
                    "alert_type": n.findtext("AffectedService"),
                    "severity": n.findtext("QualitativeNewState"),
                    "timestamp": n.findtext("TimeOfStateChange"),
                }
            )

    return alerts


def dedupe_alerts(alerts: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(alerts)

    if df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].astype(str).str.strip()

    return df[REQUIRED_COLUMNS].drop_duplicates(subset=REQUIRED_COLUMNS, keep="first")


def _validate_identifier(identifier: str, field_name: str) -> str:
    if not IDENTIFIER_RE.match(identifier):
        raise ValueError(f"Invalid {field_name} '{identifier}'. Use letters, numbers, and underscore only.")
    return identifier


def append_deduped_to_postgres(
    df: pd.DataFrame,
    db_url: str,
    table: str = "stitched_alerts_dedup",
    schema: str = "public",
) -> int:
    table = _validate_identifier(table, "table")
    schema = _validate_identifier(schema, "schema")

    if df.empty:
        return 0

    engine = create_engine(db_url)
    is_sqlite = db_url.startswith("sqlite")
    table_ref = f'"{table}"' if is_sqlite else f'"{schema}"."{table}"'
    unique_idx = f"{table}_uniq_alert"

    # SQLite has no schemas, so skip schema creation when using the demo fallback DB.
    create_schema_sql = None if is_sqlite else text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    create_table_sql = text(
        f"""
        CREATE TABLE IF NOT EXISTS {table_ref} (
            source TEXT NOT NULL,
            organization TEXT NOT NULL,
            device TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
        """
    )
    create_idx_sql = text(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS "{unique_idx}"
        ON {table_ref} (source, organization, device, alert_type, severity, timestamp)
        """
    )

    insert_sql = text(
        f"""
        INSERT INTO {table_ref} (source, organization, device, alert_type, severity, timestamp)
        VALUES (:source, :organization, :device, :alert_type, :severity, :timestamp)
        ON CONFLICT (source, organization, device, alert_type, severity, timestamp) DO NOTHING
        """
    )

    records = df.to_dict(orient="records")

    with engine.begin() as conn:
        if create_schema_sql is not None:
            conn.execute(create_schema_sql)
        conn.execute(create_table_sql)
        conn.execute(create_idx_sql)
        result = conn.execute(insert_sql, records)

    return int(result.rowcount or 0)


def _incident_id_for_row(row: pd.Series, ts: pd.Timestamp | None, window_minutes: int) -> str:
    if ts is not None and pd.notna(ts):
        bucket = int(ts.timestamp()) // (window_minutes * 60)
    else:
        bucket = f"raw:{row.get('timestamp', '')}"
    key = f"{row.get('source', '')}|{row.get('organization', '')}|{row.get('alert_type', '')}|{bucket}"
    return "inc_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:24]


def _highest_severity(values: pd.Series) -> str:
    rank = {
        "emergency": 5,
        "critical": 4,
        "failed": 3,
        "warning": 2,
        "normal": 1,
    }
    best = "unknown"
    best_score = -1
    for value in values.fillna("").astype(str):
        key = value.strip().lower()
        score = rank.get(key, 0)
        if score > best_score:
            best = value if value else "unknown"
            best_score = score
    return best


def append_incident_tables(
    df: pd.DataFrame,
    db_url: str,
    schema: str = "public",
    correlation_window_minutes: int = 10,
) -> tuple[int, int]:
    schema = _validate_identifier(schema, "schema")

    if df.empty:
        return 0, 0

    work = df.copy()
    work["ts"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce", format="mixed")
    work["incident_id"] = work.apply(
        lambda row: _incident_id_for_row(row, row["ts"], correlation_window_minutes),
        axis=1,
    )

    is_sqlite = db_url.startswith("sqlite")
    alerts_ref = f'"alerts_with_incident"' if is_sqlite else f'"{schema}"."alerts_with_incident"'
    incidents_ref = f'"incidents"' if is_sqlite else f'"{schema}"."incidents"'

    alerts_records = work[
        ["source", "organization", "device", "alert_type", "severity", "timestamp", "incident_id"]
    ].to_dict(orient="records")

    incident_rows: list[dict[str, Any]] = []
    for incident_id, group in work.groupby("incident_id", sort=False):
        ts_vals = group["ts"].dropna()
        if not ts_vals.empty:
            start_time = ts_vals.min().strftime("%Y-%m-%dT%H:%M:%SZ")
            end_time = ts_vals.max().strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            start_time = str(group["timestamp"].iloc[0])
            end_time = str(group["timestamp"].iloc[-1])

        devices = sorted({str(x) for x in group["device"].fillna("unknown")})
        device_value = devices[0] if len(devices) == 1 else f"MULTIPLE ({len(devices)} devices)"
        incident_type = str(group["alert_type"].mode().iloc[0]) if not group["alert_type"].mode().empty else "unknown"

        incident_rows.append(
            {
                "incident_id": str(incident_id),
                "source": str(group["source"].iloc[0]),
                "organization": str(group["organization"].iloc[0]),
                "device": device_value,
                "incident_type": incident_type,
                "start_time": start_time,
                "end_time": end_time,
                "alert_count": int(len(group)),
                "highest_severity": _highest_severity(group["severity"]),
                "status": "open",
            }
        )

    engine = create_engine(db_url)
    create_schema_sql = None if is_sqlite else text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    create_alerts_table_sql = text(
        f"""
        CREATE TABLE IF NOT EXISTS {alerts_ref} (
            source TEXT NOT NULL,
            organization TEXT NOT NULL,
            device TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            incident_id TEXT
        )
        """
    )
    create_alerts_idx_sql = text(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS "alerts_with_incident_uniq_alert"
        ON {alerts_ref} (source, organization, device, alert_type, severity, timestamp)
        """
    )
    create_incidents_table_sql = text(
        f"""
        CREATE TABLE IF NOT EXISTS {incidents_ref} (
            incident_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            organization TEXT NOT NULL,
            device TEXT NOT NULL,
            incident_type TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            alert_count BIGINT NOT NULL,
            highest_severity TEXT NOT NULL,
            status TEXT NOT NULL
        )
        """
    )

    upsert_alerts_sql = text(
        f"""
        INSERT INTO {alerts_ref} (source, organization, device, alert_type, severity, timestamp, incident_id)
        VALUES (:source, :organization, :device, :alert_type, :severity, :timestamp, :incident_id)
        ON CONFLICT (source, organization, device, alert_type, severity, timestamp)
        DO UPDATE SET incident_id = EXCLUDED.incident_id
        """
    )

    upsert_incidents_sql = text(
        f"""
        INSERT INTO {incidents_ref}
        (incident_id, source, organization, device, incident_type, start_time, end_time, alert_count, highest_severity, status)
        VALUES
        (:incident_id, :source, :organization, :device, :incident_type, :start_time, :end_time, :alert_count, :highest_severity, :status)
        ON CONFLICT (incident_id)
        DO UPDATE SET
            source = EXCLUDED.source,
            organization = EXCLUDED.organization,
            device = EXCLUDED.device,
            incident_type = EXCLUDED.incident_type,
            start_time = EXCLUDED.start_time,
            end_time = EXCLUDED.end_time,
            alert_count = EXCLUDED.alert_count,
            highest_severity = EXCLUDED.highest_severity,
            status = EXCLUDED.status
        """
    )

    with engine.begin() as conn:
        if create_schema_sql is not None:
            conn.execute(create_schema_sql)
        conn.execute(create_alerts_table_sql)
        conn.execute(create_alerts_idx_sql)
        conn.execute(create_incidents_table_sql)
        r1 = conn.execute(upsert_alerts_sql, alerts_records)
        r2 = conn.execute(upsert_incidents_sql, incident_rows)

    return int(r1.rowcount or 0), int(r2.rowcount or 0)

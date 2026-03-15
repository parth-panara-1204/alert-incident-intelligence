from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, validator
from sqlalchemy import create_engine, text

from parser.pipeline_service import build_db_url


DATA_DIR = Path(__file__).resolve().parent.parent / "synthesis"
DB_LOOKBACK_LIMIT = 750


class ChatRequest(BaseModel):
	question: str = Field(..., min_length=3, description="Analyst question to ask the incident copilot")
	top_k: int = Field(default=4, ge=1, le=15, description="How many matches to return for context")

	@validator("question")
	def _strip_question(cls, value: str) -> str:  # noqa: N805
		cleaned = value.strip()
		if not cleaned:
			raise ValueError("question cannot be empty")
		return cleaned


@dataclass
class Match:
	source: str
	title: str
	organization: str
	device: str
	severity: str
	timestamp: str
	status: str
	summary: str
	score: float

	def as_dict(self) -> dict[str, Any]:
		return {
			"source": self.source,
			"title": self.title,
			"organization": self.organization,
			"device": self.device,
			"severity": self.severity,
			"timestamp": self.timestamp,
			"status": self.status,
			"summary": self.summary,
			"score": round(self.score, 3),
		}


def chat_with_synthesis(*, question: str, top_k: int = 4) -> dict[str, Any]:
	if top_k < 1:
		raise ValueError("top_k must be at least 1")

	# Prefer fresh alerts from the database; fall back to the static synthetic corpus.
	records = _search_records(question, limit=top_k)
	if not records:
		return {
			"answer": "I could not find any synthetic alerts related to that question. Try mentioning a device, vendor, or severity.",
			"matches": [],
		}

	headline = _build_headline(records)
	return {
		"answer": headline,
		"matches": [record.as_dict() for record in records],
	}


# Internal helpers


def _build_headline(records: list[Match]) -> str:
	if not records:
		return "No matching alerts found."

	parts = []
	for match in records:
		parts.append(
			f"{match.source} | {match.organization or 'unknown org'} | {match.device or 'unknown device'} | {match.title} | severity: {match.severity or 'unknown'} | status: {match.status or 'n/a'}"
		)

	return "Here are the closest synthetic alerts I found:\n- " + "\n- ".join(parts)


def _search_records(question: str, *, limit: int) -> list[Match]:
	query_tokens = _tokenize(question)
	candidates: list[Match] = []

	for record in _combined_corpus():
		overlap = len(query_tokens & record["tokens"])
		severity_weight = _severity_weight(record.get("severity", ""))
		recency_weight = _recency_weight(record.get("timestamp"))
		score = (overlap * 1.8) + severity_weight + recency_weight
		candidates.append(
			Match(
				source=record.get("source", "unknown"),
				title=record.get("title", "Unknown alert"),
				organization=record.get("organization", ""),
				device=record.get("device", ""),
				severity=record.get("severity", ""),
				timestamp=record.get("timestamp", ""),
				status=record.get("status", ""),
				summary=record.get("summary", ""),
				score=score,
			)
		)

	ordered = sorted(candidates, key=lambda m: m.score, reverse=True)
	return ordered[:limit]


@lru_cache(maxsize=1)
def _load_corpus() -> tuple[dict[str, Any], ...]:
	records: list[dict[str, Any]] = []
	records.extend(_load_json_file(DATA_DIR / "synthetic_auvik.json", source="auvik"))
	records.extend(_load_json_file(DATA_DIR / "synthetic_meraki.json", source="meraki"))
	records.extend(_load_ncentral_xml(DATA_DIR / "synthetic_ncentral.xml"))
	return tuple(records)


def _combined_corpus() -> list[dict[str, Any]]:
	"""Merge fresh DB alerts (if available) with the static synthetic corpus."""
	records = _load_db_alerts(limit=DB_LOOKBACK_LIMIT)
	if not records:
		return list(_load_corpus())

	# Add synthetic corpus as backfill so the bot can still answer edge cases.
	records.extend(_load_corpus())
	return records


def _load_json_file(path: Path, *, source: str) -> list[dict[str, Any]]:
	if not path.exists():
		return []
	with path.open("r", encoding="utf-8") as handle:
		payload = json.load(handle)

	records: list[dict[str, Any]] = []
	for item in payload:
		organization = item.get("companyName") or item.get("organization") or item.get("customerName") or ""
		device = item.get("entityName") or item.get("deviceName") or item.get("device") or ""
		severity = item.get("alertSeverityString") or item.get("severity") or item.get("alertStatusString") or ""
		status = item.get("alertStatusString") or item.get("status") or ""
		timestamp = item.get("date") or item.get("occurredAt") or ""
		title = item.get("alertName") or item.get("title") or item.get("subject") or "Unknown alert"
		summary = item.get("alertDescription") or item.get("description") or title

		text_blob = " ".join(str(v) for v in [title, organization, device, severity, status, summary] if v)
		records.append(
			{
				"source": source,
				"organization": organization,
				"device": device,
				"severity": severity,
				"status": status,
				"timestamp": timestamp,
				"title": title,
				"summary": summary,
				"tokens": _tokenize(text_blob),
			}
		)

	return records


def _load_ncentral_xml(path: Path) -> list[dict[str, Any]]:
	if not path.exists():
		return []

	tree = ET.parse(path)
	root = tree.getroot()
	records: list[dict[str, Any]] = []

	for node in root.findall("notification"):
		organization = _get_text(node, "CustomerName")
		device = _get_text(node, "DeviceName") or _get_text(node, "DeviceURI")
		severity = _get_text(node, "QualitativeNewState")
		status = _get_text(node, "QualitativeNewState")
		timestamp = _get_text(node, "TimeOfStateChange")
		title = _get_text(node, "AffectedService") or "N-central alert"
		quantitative = _get_text(node, "QuantitativeNewState")
		summary = f"{title}: {quantitative}" if quantitative else title

		text_blob = " ".join(str(v) for v in [title, organization, device, severity, status, summary] if v)
		records.append(
			{
				"source": "ncentral",
				"organization": organization,
				"device": device,
				"severity": severity,
				"status": status,
				"timestamp": timestamp,
				"title": title,
				"summary": summary,
				"tokens": _tokenize(text_blob),
			}
		)

	return records


def _tokenize(text: str) -> set[str]:
	words = re.findall(r"[a-z0-9]+", str(text).lower())
	return set(words)


def _severity_weight(severity: str) -> float:
	normalized = severity.lower()
	if "crit" in normalized or "emerg" in normalized:
		return 2.0
	if "warn" in normalized:
		return 1.2
	if "fail" in normalized:
		return 1.5
	return 0.6 if normalized else 0.0


def _recency_weight(timestamp: str | None) -> float:
	if not timestamp:
		return 0.0
	try:
		parsed = _parse_timestamp(timestamp)
	except ValueError:
		return 0.0

	now = datetime.now(timezone.utc)
	delta = now - parsed.astimezone(timezone.utc)
	days = max(delta.total_seconds() / 86400, 0.1)
	return max(0.0, 2.5 / (1 + days))


def _parse_timestamp(value: str) -> datetime:
	cleaned = value.strip()
	if cleaned.endswith("Z"):
		cleaned = cleaned.replace("Z", "+00:00")
	parsed = datetime.fromisoformat(cleaned)
	if parsed.tzinfo is None:
		parsed = parsed.replace(tzinfo=timezone.utc)
	return parsed


def _get_text(node: ET.Element, tag: str) -> str:
	child = node.find(tag)
	return child.text.strip() if child is not None and child.text else ""


def _load_db_alerts(*, limit: int) -> list[dict[str, Any]]:
	"""Load recent alerts from the database; return empty list on failure or no data."""
	try:
		db_url = build_db_url(None)
		schema_prefix = "" if db_url.startswith("sqlite") else '"public".'
		stitched_ref = f'"stitched_alerts_dedup"' if not schema_prefix else f'{schema_prefix}"stitched_alerts_dedup"'
		alerts_ref = f'"alerts_with_incident"' if not schema_prefix else f'{schema_prefix}"alerts_with_incident"'
		incidents_ref = f'"incidents"' if not schema_prefix else f'{schema_prefix}"incidents"'

		query = text(
			f"""
			SELECT
			  s.source,
			  s.organization,
			  s.device,
			  s.alert_type,
			  s.severity,
			  s.timestamp,
			  a.incident_id,
			  i.incident_type,
			  i.status
			FROM {stitched_ref} s
			LEFT JOIN {alerts_ref} a
			  ON s.source = a.source
			 AND s.organization = a.organization
			 AND s.device = a.device
			 AND s.alert_type = a.alert_type
			 AND s.severity = a.severity
			 AND s.timestamp = a.timestamp
			LEFT JOIN {incidents_ref} i
			  ON a.incident_id = i.incident_id
			ORDER BY s.timestamp DESC
			LIMIT :limit
			"""
		)

		engine = create_engine(db_url)
		with engine.connect() as conn:
			rows = conn.execute(query, {"limit": limit}).mappings().all()

		records: list[dict[str, Any]] = []
		for row in rows:
			title = row.get("alert_type") or row.get("incident_type") or "Alert"
			summary = " ".join(
				str(v)
				for v in [row.get("incident_type"), row.get("status"), row.get("device"), row.get("alert_type")]
				if v
			)
			records.append(
				{
					"source": row.get("source", "unknown"),
					"organization": row.get("organization", ""),
					"device": row.get("device", ""),
					"severity": row.get("severity", ""),
					"status": row.get("status", ""),
					"timestamp": row.get("timestamp", ""),
					"title": title,
					"summary": summary or title,
					"tokens": _tokenize(" ".join([title, summary])),
				}
			)

		return records
	except Exception:
		# If the DB is unavailable or tables are empty, fall back to synthetic corpus.
		return []

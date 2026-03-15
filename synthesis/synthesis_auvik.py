"""Generate synthetic Auvik-style alerts via CTGAN (no normalization path)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from ctgan import CTGAN
from scipy.stats import ks_2samp

from ingest_client import ingest_records


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "data" / "auvik.json"
DEFAULT_SYNTHETIC = Path(__file__).with_name("synthetic_auvik.json")


def load_raw(input_path: Path = DEFAULT_INPUT) -> list[Mapping[str, object]]:
	with input_path.open("r", encoding="utf-8") as f:
		return json.load(f)


DEFAULT_EPOCHS = 200
DEFAULT_SAMPLES = 1000
DEFAULT_JITTER_SECONDS = 900  # +/- 15 minutes
DEFAULT_NEAR_DUP_WINDOW = 300  # seconds window to drop near-duplicates vs source
DISCRETE_COLUMNS = [
	"entityId",
	"subject",
	"alertStatusString",
	"alertId",
	"alertName",
	"entityName",
	"companyName",
	"entityType",
	"link",
	"alertStatus",
	"correlationId",
	"alertDescription",
	"alertSeverityString",
	"alertSeverity",
	"companyId",
]


# -----------------------------
# CTGAN synthesis (native schema)
# -----------------------------
def _to_epoch(series: pd.Series) -> pd.Series:
	return pd.to_datetime(series, utc=True, errors="coerce").astype("int64") // 10**9


def _fill_ts(series: pd.Series, ref: pd.Series) -> pd.Series:
	fallback = pd.to_datetime(ref, utc=True, errors="coerce").median()
	ts = pd.to_datetime(series, unit="s", utc=True, errors="coerce")
	if pd.notna(fallback):
		ts = ts.fillna(fallback)
	return ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def synthesize(
	raw_records: list[Mapping[str, object]],
	*,
	samples: int,
	epochs: int,
	seed: int | None,
	jitter_seconds: int,
	near_dup_window: int,
) -> list[Mapping[str, object]]:
	df = pd.DataFrame(raw_records)
	df["date_epoch"] = _to_epoch(df["date"])
	df = df.drop(columns=["date"], errors="ignore")
	raw_df = pd.DataFrame(raw_records).copy()
	raw_df["date_epoch"] = _to_epoch(raw_df["date"])

	if seed is not None:
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)

	model = CTGAN(epochs=epochs)
	model.fit(df, discrete_columns=DISCRETE_COLUMNS)
	synthetic = model.sample(samples)

	if jitter_seconds > 0 and "date_epoch" in synthetic:
		synthetic["date_epoch"] = pd.to_numeric(synthetic["date_epoch"], errors="coerce")
		jitter = np.random.randint(-jitter_seconds, jitter_seconds + 1, size=len(synthetic))
		synthetic["date_epoch"] = (synthetic["date_epoch"] + jitter).clip(lower=0)

	if near_dup_window > 0:
		before = len(synthetic)
		synthetic = _drop_near_duplicates(raw_df, synthetic, near_dup_window)
		after = len(synthetic)
		removed = before - after
		if removed > 0:
			print(f"Removed {removed} near-duplicate rows against source")

	synthetic["date"] = _fill_ts(synthetic.get("date_epoch"), pd.Series([r.get("date") for r in raw_records]))
	synthetic = synthetic.drop(columns=["date_epoch"], errors="ignore")
	return synthetic.to_dict(orient="records")


def _rebalance_alert_type(records: list[Mapping[str, object]]) -> list[Mapping[str, object]]:
	"""Spread alertName/subject for more variety in dashboards."""
	if not records:
		return records

	choices = [
		("Interface Down", 0.2),
		("BGP Session Down", 0.15),
		("VPN Tunnel Degraded", 0.15),
		("High CPU Utilization", 0.15),
		("High Memory Utilization", 0.1),
		("Packet Loss High", 0.15),
		("Temperature High", 0.1),
	]
	labels, weights = zip(*choices)
	for rec in records:
		chosen = random.choices(labels, weights=weights, k=1)[0]
		rec["alertName"] = chosen
		rec["subject"] = rec.get("subject") or chosen
		rec["alertDescription"] = rec.get("alertDescription") or chosen
	return records


def _report_similarity(
	raw_records: list[Mapping[str, object]],
	synthetic_records: list[Mapping[str, object]],
) -> None:
	raw_df = pd.DataFrame(raw_records).copy()
	synth_df = pd.DataFrame(synthetic_records).copy()

	raw_df["date_epoch"] = pd.to_datetime(raw_df["date"], utc=True, errors="coerce").astype("int64") // 10**9
	synth_df["date_epoch"] = pd.to_datetime(synth_df["date"], utc=True, errors="coerce").astype("int64") // 10**9

	print("\n--- Similarity report (raw vs synthetic) ---")
	print(f"Samples: raw={len(raw_df)} synthetic={len(synth_df)}")

	ks_stat, ks_p = ks_2samp(raw_df["date_epoch"].dropna(), synth_df["date_epoch"].dropna())
	print(
		"date_epoch: "
		f"mean raw {raw_df['date_epoch'].mean():.2f} synth {synth_df['date_epoch'].mean():.2f} | "
		f"std raw {raw_df['date_epoch'].std():.2f} synth {synth_df['date_epoch'].std():.2f} | "
		f"ks_stat {ks_stat:.4f} p {ks_p:.4f}"
	)

	def _top_counts(df: pd.DataFrame, col: str, n: int = 3) -> pd.Series:
		return df[col].value_counts(normalize=True).head(n)

	print("\nTop category frequency (proportion)")
	for col in DISCRETE_COLUMNS:
		if col not in raw_df or col not in synth_df:
			continue
		real_top = _top_counts(raw_df, col)
		synth_top = _top_counts(synth_df, col)
		print(f"\n{col}:")
		print("  raw:\n", real_top.to_string())
		print("  synth:\n", synth_top.to_string())


def _drop_near_duplicates(raw_df: pd.DataFrame, synth_df: pd.DataFrame, window_seconds: int) -> pd.DataFrame:
	if synth_df.empty or raw_df.empty or "date_epoch" not in synth_df or "date_epoch" not in raw_df:
		return synth_df

	raw_map: dict[tuple, list] = {}
	for _, row in raw_df.iterrows():
		key = tuple(row.get(col) for col in DISCRETE_COLUMNS)
		de = row.get("date_epoch")
		if pd.isna(de):
			continue
		raw_map.setdefault(key, []).append(de)

	keep = []
	for _, row in synth_df.iterrows():
		key = tuple(row.get(col) for col in DISCRETE_COLUMNS)
		de = row.get("date_epoch")
		if pd.isna(de):
			keep.append(row)
			continue
		times = raw_map.get(key)
		if not times:
			keep.append(row)
			continue
		# drop if categories match and timestamp is close, but keep exact matches (equal timestamp)
		if any(abs(de - t) <= window_seconds and de != t for t in times):
			continue
		keep.append(row)

	if not keep:
		return synth_df.iloc[0:0]
	return pd.DataFrame(keep, columns=synth_df.columns)


def save_synthetic(records: list[Mapping[str, object]], output_path: Path = DEFAULT_SYNTHETIC) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def cli() -> None:
	parser = argparse.ArgumentParser(description="Auvik CTGAN synthesizer")
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
	parser.add_argument("--output", type=Path, default=DEFAULT_SYNTHETIC)
	parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="Synthetic rows")
	parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="CTGAN epochs")
	parser.add_argument("--seed", type=int, default=None, help="Random seed")
	parser.add_argument("--jitter-seconds", type=int, default=DEFAULT_JITTER_SECONDS, help="Max absolute seconds to jitter timestamps")
	parser.add_argument("--near-dup-window", type=int, default=DEFAULT_NEAR_DUP_WINDOW, help="Drop rows matching source categories within this many seconds (keeps exact matches)")
	parser.add_argument("--ingest-url", type=str, default=None, help="Backend ingest endpoint (POST), e.g., http://localhost:8000/ingest")
	parser.add_argument("--ingest-db-url", type=str, default=None, help="Optional DB URL override passed to backend")
	parser.add_argument("--ingest-table", type=str, default="stitched_alerts_dedup", help="Backend destination table")
	parser.add_argument("--ingest-schema", type=str, default="public", help="Backend destination schema")
	parser.add_argument("--ingest-timeout", type=float, default=10.0, help="Ingest request timeout in seconds")
	parser.add_argument("--continue-on-ingest-error", action="store_true", help="Log ingest errors and continue")
	parser.add_argument("--report", action="store_true", help="Print similarity report vs raw data")
	args = parser.parse_args()

	raw = load_raw(args.input)
	records = synthesize(
		raw,
		samples=args.samples,
		epochs=args.epochs,
		seed=args.seed,
		jitter_seconds=args.jitter_seconds,
		near_dup_window=args.near_dup_window,
	)
	records = _rebalance_alert_type(records)
	save_synthetic(records, args.output)
	print(f"Synthetic Auvik alerts ({len(records)}) -> {args.output}")

	ingest_records(
		source="auvik",
		records=records,
		ingest_url=args.ingest_url,
		db_url=args.ingest_db_url,
		table=args.ingest_table,
		schema=args.ingest_schema,
		timeout=args.ingest_timeout,
		continue_on_error=args.continue_on_ingest_error,
	)

	if args.report:
		_report_similarity(raw, records)


if __name__ == "__main__":
	cli()

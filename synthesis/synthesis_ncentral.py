"""Generate synthetic N-Central-style alerts via CTGAN (no normalization path)."""

from __future__ import annotations

import argparse
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from ctgan import CTGAN
from scipy.stats import ks_2samp

from ingest_client import ingest_records


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "data" / "ncentral.xml"
DEFAULT_SYNTHETIC = Path(__file__).with_name("synthetic_ncentral.xml")
DEFAULT_EPOCHS = 200
DEFAULT_SAMPLES = 1000
DEFAULT_JITTER_SECONDS = 900  # +/- 15 minutes
DEFAULT_NEAR_DUP_WINDOW = 300  # seconds window to drop near-duplicates vs source


def load_root(input_path: Path = DEFAULT_INPUT) -> ET.Element:
	raw_xml = input_path.read_text(encoding="utf-8")

	# ncentral export repeats XML headers; strip and wrap so ElementTree can parse.
	sanitized = re.sub(r"<\?xml.*?\?>", "", raw_xml)
	wrapped = f"<root>{sanitized}</root>"
	return ET.fromstring(wrapped)


# -----------------------------
# CTGAN synthesis (native schema)
# -----------------------------
NC_FIELDS = [
	"ActiveNotificationTriggerID",
	"CustomerName",
	"DeviceURI",
	"DeviceName",
	"ExternalCustomerID",
	"AffectedService",
	"TaskIdent",
	"NcentralURI",
	"QualitativeOldState",
	"QualitativeNewState",
	"TimeOfStateChange",
	"ProbeURI",
	"QuantitativeNewState",
	"ServiceOrganizationName",
	"RemoteControlLink",
	"AcknowledgementTime",
	"AcknowledgementUser",
	"ActiveProfile",
]


def _record_to_xml(rec: Mapping[str, object]) -> str:
	n = ET.Element("notification")
	for field in NC_FIELDS:
		value = rec.get(field)
		if value is None:
			continue
		child = ET.SubElement(n, field)
		child.text = str(value)
	return ET.tostring(n, encoding="unicode")


def _to_epoch(series: pd.Series) -> pd.Series:
	return pd.to_datetime(series, utc=True, errors="coerce").astype("int64") // 10**9


def _fill_ts(series: pd.Series, ref: pd.Series) -> pd.Series:
	fallback = pd.to_datetime(ref, utc=True, errors="coerce").median()
	ts = pd.to_datetime(series, unit="s", utc=True, errors="coerce")
	if pd.notna(fallback):
		ts = ts.fillna(fallback)
	return ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_dataframe(input_path: Path) -> pd.DataFrame:
	root = load_root(input_path)
	rows = []
	for n in root.findall("notification"):
		rows.append({field: n.findtext(field) for field in NC_FIELDS})
	return pd.DataFrame(rows)


def synthesize(
	input_path: Path = DEFAULT_INPUT,
	*,
	samples: int,
	epochs: int,
	seed: int | None,
	jitter_seconds: int,
	near_dup_window: int,
) -> list[Mapping[str, object]]:
	df = _load_dataframe(input_path)
	ref_time = df["TimeOfStateChange"].copy()
	df["TimeOfStateChange_epoch"] = _to_epoch(df["TimeOfStateChange"])
	df = df.drop(columns=["TimeOfStateChange"], errors="ignore")
	raw_df = _load_dataframe(input_path)
	raw_df["TimeOfStateChange_epoch"] = _to_epoch(raw_df["TimeOfStateChange"])

	if seed is not None:
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)

	discrete = [col for col in df.columns if col != "TimeOfStateChange_epoch"]

	model = CTGAN(epochs=epochs)
	model.fit(df, discrete_columns=discrete)
	synthetic = model.sample(samples)

	if jitter_seconds > 0 and "TimeOfStateChange_epoch" in synthetic:
		synthetic["TimeOfStateChange_epoch"] = pd.to_numeric(synthetic["TimeOfStateChange_epoch"], errors="coerce")
		jitter = np.random.randint(-jitter_seconds, jitter_seconds + 1, size=len(synthetic))
		synthetic["TimeOfStateChange_epoch"] = (synthetic["TimeOfStateChange_epoch"] + jitter).clip(lower=0)

	if near_dup_window > 0:
		before = len(synthetic)
		synthetic = _drop_near_duplicates(raw_df, synthetic, near_dup_window)
		after = len(synthetic)
		removed = before - after
		if removed > 0:
			print(f"Removed {removed} near-duplicate rows against source")

	synthetic["TimeOfStateChange"] = _fill_ts(
		synthetic["TimeOfStateChange_epoch"], ref_time
	)
	synthetic = synthetic.drop(columns=["TimeOfStateChange_epoch"], errors="ignore")
	return synthetic.reindex(columns=NC_FIELDS).to_dict(orient="records")


def _rebalance_alert_type(records: list[Mapping[str, object]]) -> list[Mapping[str, object]]:
	"""Spread AffectedService values for N-Central synthetic alerts."""
	if not records:
		return records

	choices = [
		("Firewall WAN Link", 0.2),
		("VPN Tunnel", 0.15),
		("Disk Usage", 0.15),
		("CPU Load", 0.1),
		("Memory Utilization", 0.1),
		("Power Supply", 0.1),
		("Switch Port", 0.1),
		("Wireless AP Down", 0.1),
	]
	labels, weights = zip(*choices)
	for rec in records:
		rec["AffectedService"] = random.choices(labels, weights=weights, k=1)[0]
	return records


def _report_similarity(
	raw_records: list[Mapping[str, object]],
	synthetic_records: list[Mapping[str, object]],
) -> None:
	raw_df = pd.DataFrame(raw_records).copy()
	synth_df = pd.DataFrame(synthetic_records).copy()

	raw_df["TimeOfStateChange_epoch"] = _to_epoch(raw_df["TimeOfStateChange"])
	synth_df["TimeOfStateChange_epoch"] = _to_epoch(synth_df["TimeOfStateChange"])

	print("\n--- Similarity report (raw vs synthetic) ---")
	print(f"Samples: raw={len(raw_df)} synthetic={len(synth_df)}")

	ks_stat, ks_p = ks_2samp(raw_df["TimeOfStateChange_epoch"].dropna(), synth_df["TimeOfStateChange_epoch"].dropna())
	print(
		"TimeOfStateChange_epoch: "
		f"mean raw {raw_df['TimeOfStateChange_epoch'].mean():.2f} synth {synth_df['TimeOfStateChange_epoch'].mean():.2f} | "
		f"std raw {raw_df['TimeOfStateChange_epoch'].std():.2f} synth {synth_df['TimeOfStateChange_epoch'].std():.2f} | "
		f"ks_stat {ks_stat:.4f} p {ks_p:.4f}"
	)

	def _top_counts(df: pd.DataFrame, col: str, n: int = 3) -> pd.Series:
		return df[col].value_counts(normalize=True).head(n)

	print("\nTop category frequency (proportion)")
	for col in NC_FIELDS:
		if col == "TimeOfStateChange":
			continue
		if col not in raw_df or col not in synth_df:
			continue
		real_top = _top_counts(raw_df, col)
		synth_top = _top_counts(synth_df, col)
		print(f"\n{col}:")
		print("  raw:\n", real_top.to_string())
		print("  synth:\n", synth_top.to_string())


def _drop_near_duplicates(raw_df: pd.DataFrame, synth_df: pd.DataFrame, window_seconds: int) -> pd.DataFrame:
	if synth_df.empty or raw_df.empty or "TimeOfStateChange_epoch" not in synth_df or "TimeOfStateChange_epoch" not in raw_df:
		return synth_df

	discrete_cols = [col for col in NC_FIELDS if col != "TimeOfStateChange"]
	raw_map: dict[tuple, list] = {}
	for _, row in raw_df.iterrows():
		key = tuple(row.get(col) for col in discrete_cols)
		de = row.get("TimeOfStateChange_epoch")
		if pd.isna(de):
			continue
		raw_map.setdefault(key, []).append(de)

	keep = []
	for _, row in synth_df.iterrows():
		key = tuple(row.get(col) for col in discrete_cols)
		de = row.get("TimeOfStateChange_epoch")
		if pd.isna(de):
			keep.append(row)
			continue
		times = raw_map.get(key)
		if not times:
			keep.append(row)
			continue
		if any(abs(de - t) <= window_seconds and de != t for t in times):
			continue
		keep.append(row)

	if not keep:
		return synth_df.iloc[0:0]
	return pd.DataFrame(keep, columns=synth_df.columns)


def save_synthetic(records: list[Mapping[str, object]], output_path: Path = DEFAULT_SYNTHETIC) -> None:
	root = ET.Element("root")
	for rec in records:
		n = ET.SubElement(root, "notification")
		for field in NC_FIELDS:
			value = rec.get(field)
			if value is None:
				continue
			child = ET.SubElement(n, field)
			child.text = str(value)

	xml_str = ET.tostring(root, encoding="unicode")
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(xml_str, encoding="utf-8")


def cli() -> None:
	parser = argparse.ArgumentParser(description="N-Central CTGAN synthesizer")
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

	records = synthesize(
		args.input,
		samples=args.samples,
		epochs=args.epochs,
		seed=args.seed,
		jitter_seconds=args.jitter_seconds,
		near_dup_window=args.near_dup_window,
	)
	records = _rebalance_alert_type(records)
	save_synthetic(records, args.output)
	print(f"Synthetic N-Central alerts ({len(records)}) -> {args.output}")

	ingest_records(
		source="ncentral",
		records=records,
		ingest_url=args.ingest_url,
		db_url=args.ingest_db_url,
		table=args.ingest_table,
		schema=args.ingest_schema,
		timeout=args.ingest_timeout,
		continue_on_error=args.continue_on_ingest_error,
		payload_encoder=_record_to_xml,
	)

	if args.report:
		_report_similarity(_load_dataframe(args.input).to_dict(orient="records"), records)


if __name__ == "__main__":
	cli()

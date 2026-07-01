#!/usr/bin/env python3
"""
Summarize parsed target-attention shape / pack-stat JSON records.

Inputs are JSON files produced by:
  - parse_ta_shape_log.py
  - parse_ta_pack_stats_log.py

The script prints compact distribution stats and highlights extreme batches.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_FIELDS = [
    ("valid_user", "valid_user/n_seg"),
    ("packed_q", "packed_q/sum_q"),
    ("packed_k", "packed_k/sum_k"),
    ("q_min", "q_min"),
    ("q_max", "q_max"),
    ("k_min", "k_min"),
    ("k_max", "k_max"),
    ("q_drop_ratio_by_q_max", "q_drop_by_qmax"),
    ("k_drop_ratio_by_k_max", "k_drop_by_kmax"),
    ("invalid_user_ratio", "invalid_user_ratio"),
    ("query_drop_ratio", "query_drop_ratio"),
    ("k_drop_ratio_global", "k_drop_global"),
    ("k_drop_ratio_valid_user", "k_drop_valid_user"),
]


def _load_records(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload, {"source": str(path), "record_count": len(payload)}
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object or list")
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError("expected payload['records'] to be a list")
    return records, payload


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _values(records: List[Dict[str, Any]], key: str) -> List[float]:
    values = []
    for record in records:
        value = _as_float(record.get(key))
        if value is not None:
            values.append(value)
    return values


def _percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("empty values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def _stats(values: List[float]) -> Dict[str, float]:
    values = sorted(values)
    total = sum(values)
    return {
        "min": values[0],
        "mean": total / len(values),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
        "max": values[-1],
    }


def _fmt(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}"


def _print_field_stats(records: List[Dict[str, Any]]) -> None:
    print("\nfield distributions")
    for key, label in DEFAULT_FIELDS:
        values = _values(records, key)
        if not values:
            continue
        stats = _stats(values)
        print(
            f"  {label:24s} "
            f"min={_fmt(stats['min']):>10s} "
            f"mean={_fmt(stats['mean']):>10s} "
            f"p50={_fmt(stats['p50']):>10s} "
            f"p90={_fmt(stats['p90']):>10s} "
            f"p99={_fmt(stats['p99']):>10s} "
            f"max={_fmt(stats['max']):>10s}"
        )


def _record_label(index: int, record: Dict[str, Any]) -> str:
    parts = [
        f"idx={index}",
        f"tag={record.get('tag', 'NA')}",
        f"source={record.get('source_kind', 'NA')}",
    ]
    for key in [
        "valid_user",
        "packed_q",
        "packed_k",
        "q_max",
        "k_max",
        "q_drop_ratio_by_q_max",
        "k_drop_ratio_by_k_max",
        "invalid_user_ratio",
        "k_drop_ratio_global",
    ]:
        value = _as_float(record.get(key))
        if value is not None:
            parts.append(f"{key}={_fmt(value)}")
    return " ".join(parts)


def _top_records(records: List[Dict[str, Any]], key: str, top: int) -> List[Tuple[int, Dict[str, Any]]]:
    ranked = []
    for i, record in enumerate(records):
        value = _as_float(record.get(key))
        if value is not None:
            ranked.append((value, i, record))
    ranked.sort(reverse=True, key=lambda item: item[0])
    return [(i, record) for _, i, record in ranked[:top]]


def _print_top(records: List[Dict[str, Any]], key: str, label: str, top: int) -> None:
    rows = _top_records(records, key, top)
    if not rows:
        return
    print(f"\ntop {min(top, len(rows))} by {label}")
    for i, record in rows:
        print(f"  {_record_label(i, record)}")


def _write_csv(records: List[Dict[str, Any]], path: Path) -> None:
    keys = [
        "tag",
        "source_kind",
        "valid_user",
        "total_user",
        "invalid_user_ratio",
        "packed_q",
        "packed_k",
        "sum_q",
        "sum_k",
        "q_min",
        "q_max",
        "k_min",
        "k_max",
        "q_dense_slots_by_q_max",
        "q_drop_ratio_by_q_max",
        "k_dense_slots_by_k_max",
        "k_drop_ratio_by_k_max",
        "query_drop_ratio",
        "k_drop_ratio_global",
        "k_drop_ratio_valid_user",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index"] + keys)
        writer.writeheader()
        for i, record in enumerate(records):
            row = {"index": i}
            for key in keys:
                row[key] = record.get(key, "")
            writer.writerow(row)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_file", type=Path)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--csv", type=Path, default=None, help="write per-record CSV")
    args = parser.parse_args(argv)

    records, payload = _load_records(args.json_file)
    if not records:
        print("no records found")
        return 1

    source_kinds = {}
    for record in records:
        key = str(record.get("source_kind", "unknown"))
        source_kinds[key] = source_kinds.get(key, 0) + 1

    print(f"source: {payload.get('source', args.json_file)}")
    print(f"records: {len(records)}")
    print(
        "source_kinds: "
        + ", ".join(f"{key}={value}" for key, value in sorted(source_kinds.items()))
    )

    if not _values(records, "invalid_user_ratio"):
        print(
            "invalid_user_ratio: NA "
            "(sum(q_seq)=packed_q, len(q_seq)=valid_user; total_user is not in shape_summary)"
        )

    _print_field_stats(records)
    _print_top(records, "packed_k", "packed_k", args.top)
    _print_top(records, "valid_user", "valid_user", args.top)
    _print_top(records, "k_drop_ratio_by_k_max", "k_drop_ratio_by_k_max", args.top)
    _print_top(records, "q_drop_ratio_by_q_max", "q_drop_ratio_by_q_max", args.top)
    _print_top(records, "invalid_user_ratio", "invalid_user_ratio", args.top)

    if args.csv is not None:
        _write_csv(records, args.csv)
        print(f"\nwrote csv: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

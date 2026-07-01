#!/usr/bin/env python3
"""
Summarize q_lens / k_lens distributions from parsed TA_SHAPE JSON.

Input should be produced by:
  python3 parse_ta_shape_log.py stderr.log --tag long_term5 -o real_shape_all.json

This script answers questions like:
  - Are most K sequences exactly 2048?
  - What fraction of K tokens come from max-length sequences?
  - What are the q_len / k_len percentiles and buckets?
"""

import argparse
import collections
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_K_BUCKETS = [1, 16, 32, 64, 128, 256, 512, 1024, 1536, 2047, 2048]
DEFAULT_Q_BUCKETS = [1, 2, 4, 8, 12, 16, 32, 64, 128]


def _load_records(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("records", [])
    else:
        raise ValueError("expected JSON object or list")
    if not isinstance(records, list):
        raise ValueError("expected records to be a list")
    return records


def _int_list(value: Any) -> List[int]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _collect_lens(records: List[Dict[str, Any]], key: str) -> Tuple[List[int], List[int]]:
    all_lens = []
    per_record_counts = []
    for record in records:
        lens = _int_list(record.get(key))
        if lens:
            all_lens.extend(lens)
            per_record_counts.append(len(lens))
    return all_lens, per_record_counts


def _percentile(sorted_values: List[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def _stats(values: Iterable[int]) -> Dict[str, float]:
    xs = sorted(float(v) for v in values)
    if not xs:
        return {}
    return {
        "min": xs[0],
        "mean": sum(xs) / len(xs),
        "p50": _percentile(xs, 0.50),
        "p90": _percentile(xs, 0.90),
        "p95": _percentile(xs, 0.95),
        "p99": _percentile(xs, 0.99),
        "max": xs[-1],
    }


def _fmt(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}"


def _print_stats(label: str, values: List[int]) -> None:
    stats = _stats(values)
    if not stats:
        print(f"{label}: no values")
        return
    print(
        f"{label}: "
        f"n={len(values)} total={sum(values)} "
        f"min={_fmt(stats['min'])} mean={_fmt(stats['mean'])} "
        f"p50={_fmt(stats['p50'])} p90={_fmt(stats['p90'])} "
        f"p95={_fmt(stats['p95'])} p99={_fmt(stats['p99'])} "
        f"max={_fmt(stats['max'])}"
    )


def _bucket_label(prev: Optional[int], upper: int) -> str:
    if prev is None:
        return f"<= {upper}"
    return f"{prev + 1}..{upper}"


def _bucket_counts(values: List[int], buckets: List[int]) -> List[Tuple[str, int, int]]:
    buckets = sorted(set(int(v) for v in buckets))
    rows = []
    prev = None  # type: Optional[int]
    for upper in buckets:
        if prev is None:
            picked = [v for v in values if v <= upper]
        else:
            picked = [v for v in values if prev < v <= upper]
        rows.append((_bucket_label(prev, upper), len(picked), sum(picked)))
        prev = upper
    picked = [v for v in values if v > buckets[-1]]
    if picked:
        rows.append((f"> {buckets[-1]}", len(picked), sum(picked)))
    return rows


def _print_buckets(label: str, values: List[int], buckets: List[int]) -> None:
    if not values:
        return
    total_count = len(values)
    total_tokens = sum(values)
    print(f"\n{label} buckets")
    for bucket, count, tokens in _bucket_counts(values, buckets):
        count_ratio = count / float(total_count)
        token_ratio = tokens / float(max(total_tokens, 1))
        print(
            f"  {bucket:12s} "
            f"segments={count:8d} ({count_ratio:8.4%}) "
            f"tokens={tokens:12d} ({token_ratio:8.4%})"
        )


def _print_top_exact(label: str, values: List[int], top: int) -> None:
    if not values:
        return
    total_count = len(values)
    total_tokens = sum(values)
    counter = collections.Counter(values)
    rows = counter.most_common(top)
    print(f"\n{label} most common exact lengths")
    for length, count in rows:
        tokens = length * count
        print(
            f"  len={length:6d} "
            f"segments={count:8d} ({count / float(total_count):8.4%}) "
            f"tokens={tokens:12d} ({tokens / float(max(total_tokens, 1)):8.4%})"
        )


def _print_exact_value(label: str, values: List[int], target: int) -> None:
    if not values:
        return
    count = sum(1 for value in values if value == target)
    tokens = count * target
    print(
        f"\n{label} == {target}: "
        f"segments={count}/{len(values)} ({count / float(len(values)):8.4%}) "
        f"tokens={tokens}/{sum(values)} ({tokens / float(max(sum(values), 1)):8.4%})"
    )


def _per_record_extremes(records: List[Dict[str, Any]], target_k: int, top: int) -> None:
    rows = []
    for i, record in enumerate(records):
        k_lens = _int_list(record.get("k_lens"))
        q_lens = _int_list(record.get("q_lens"))
        if not k_lens:
            continue
        target_count = sum(1 for value in k_lens if value == target_k)
        rows.append(
            {
                "index": i,
                "tag": record.get("tag", "NA"),
                "n_seg": len(k_lens),
                "sum_q": sum(q_lens),
                "sum_k": sum(k_lens),
                "k_target_count": target_count,
                "k_target_ratio": target_count / float(len(k_lens)),
                "k_target_token_ratio": (target_count * target_k) / float(max(sum(k_lens), 1)),
                "k_min": min(k_lens),
                "k_max": max(k_lens),
            }
        )

    if not rows:
        return

    for key, title in [
        ("sum_k", "largest packed K"),
        ("k_target_ratio", f"highest K=={target_k} segment ratio"),
        ("k_target_token_ratio", f"highest K=={target_k} token ratio"),
    ]:
        print(f"\ntop {top} records by {title}")
        for row in sorted(rows, key=lambda item: item[key], reverse=True)[:top]:
            print(
                f"  idx={row['index']} tag={row['tag']} n_seg={row['n_seg']} "
                f"sum_q={row['sum_q']} sum_k={row['sum_k']} "
                f"k_eq_target={row['k_target_count']} "
                f"k_eq_target_ratio={row['k_target_ratio']:.6f} "
                f"k_eq_target_token_ratio={row['k_target_token_ratio']:.6f} "
                f"k_min={row['k_min']} k_max={row['k_max']}"
            )


def _write_hist_csv(path: Path, q_lens: List[int], k_lens: List[int]) -> None:
    rows = []
    for name, values in [("q_len", q_lens), ("k_len", k_lens)]:
        total_count = len(values)
        total_tokens = sum(values)
        for length, count in sorted(collections.Counter(values).items()):
            tokens = length * count
            rows.append(
                {
                    "kind": name,
                    "length": length,
                    "segments": count,
                    "segment_ratio": count / float(max(total_count, 1)),
                    "tokens": tokens,
                    "token_ratio": tokens / float(max(total_tokens, 1)),
                }
            )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "kind",
                "length",
                "segments",
                "segment_ratio",
                "tokens",
                "token_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_file", type=Path)
    parser.add_argument("--target-k", type=int, default=2048)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--hist-csv", type=Path, default=None)
    args = parser.parse_args(argv)

    records = _load_records(args.json_file)
    q_lens, q_record_counts = _collect_lens(records, "q_lens")
    k_lens, k_record_counts = _collect_lens(records, "k_lens")

    if not q_lens and not k_lens:
        print("no q_lens/k_lens found; use parse_ta_shape_log.py output, not shape_summary-only JSON")
        return 1

    print(f"records with q_lens: {len(q_record_counts)}")
    print(f"records with k_lens: {len(k_record_counts)}")
    _print_stats("q_lens", q_lens)
    _print_stats("k_lens", k_lens)

    _print_exact_value("k_lens", k_lens, args.target_k)
    _print_top_exact("q_lens", q_lens, args.top)
    _print_top_exact("k_lens", k_lens, args.top)
    _print_buckets("q_lens", q_lens, DEFAULT_Q_BUCKETS)
    _print_buckets("k_lens", k_lens, DEFAULT_K_BUCKETS)
    _per_record_extremes(records, args.target_k, args.top)

    if args.hist_csv is not None:
        _write_hist_csv(args.hist_csv, q_lens, k_lens)
        print(f"\nwrote histogram csv: {args.hist_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

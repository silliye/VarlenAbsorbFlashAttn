#!/usr/bin/env python3
"""
Extract target-attention packing statistics from TensorFlow tf.Print logs.

Expected log lines are produced by a tf.Print prefix like:
  [TA_SHAPE long_term5] pack_stats total_user valid_user invalid_user ...

It also accepts the shape summary line:
  [TA_SHAPE long_term5] raw_q raw_k q_swiglu k_swiglu ...

The script emits JSON records with count and ratio validation warnings.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TAG_RE = re.compile(r"\[TA_SHAPE (?P<tag>[^\]]+)\]\s*(?P<body>.*)")
BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")
NUMBER_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)


PACK_STATS_PREFIX = (
    "pack_stats "
    "total_user valid_user invalid_user invalid_user_ratio "
    "total_query valid_query dropped_query query_drop_ratio "
    "total_k_slot packed_k dropped_k_global k_drop_ratio_global "
    "valid_user_k_slot dropped_k_valid_user k_drop_ratio_valid_user:"
)

SHAPE_SUMMARY_PREFIX = (
    "raw_q raw_k q_swiglu k_swiglu "
    "n_seg sum_q sum_k q_min q_max k_min k_max:"
)


FIELDS = [
    ("total_user", int),
    ("valid_user", int),
    ("invalid_user", int),
    ("invalid_user_ratio", float),
    ("total_query", int),
    ("valid_query", int),
    ("dropped_query", int),
    ("query_drop_ratio", float),
    ("total_k_slot", int),
    ("packed_k", int),
    ("dropped_k_global", int),
    ("k_drop_ratio_global", float),
    ("valid_user_k_slot", int),
    ("dropped_k_valid_user", int),
    ("k_drop_ratio_valid_user", float),
]


def _payload_after_colon(body: str, fallback_prefix: str) -> str:
    if ":" in body:
        return body.split(":", 1)[1]
    return body[len(fallback_prefix):]


def _parse_number(text: str) -> float:
    if "..." in text:
        raise ValueError(
            "found an ellipsis in a printed tensor; increase tf.Print summarize"
        )
    match = NUMBER_RE.search(text)
    if match is None:
        raise ValueError(f"missing numeric value in {text!r}")
    return float(match.group(0))


def _parse_pack_groups(body: str) -> List[float]:
    groups = BRACKET_RE.findall(body)
    if len(groups) == len(FIELDS):
        return [_parse_number(group) for group in groups]

    # Fallback for logs that print scalar tensors without brackets.
    values = [_parse_number(match.group(0)) for match in NUMBER_RE.finditer(body)]
    if len(values) != len(FIELDS):
        raise ValueError(
            f"expected {len(FIELDS)} values, got {len(values)} "
            f"({len(groups)} bracket groups)"
        )
    return values


def _parse_int_list(text: str) -> List[int]:
    if "..." in text:
        raise ValueError(
            "found an ellipsis in a printed tensor; increase tf.Print summarize"
        )
    return [int(float(match.group(0))) for match in NUMBER_RE.finditer(text)]


def _parse_shape_groups(body: str) -> List[List[int]]:
    return [_parse_int_list(group) for group in BRACKET_RE.findall(body)]


def _ratio(num: int, den: int) -> float:
    return float(num) / float(max(den, 1))


def _close(a: float, b: float, *, atol: float = 1e-4) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=atol)


def _parse_pack_stats(tag: str, body: str) -> Dict[str, Any]:
    values = _parse_pack_groups(body)
    record = {"tag": tag, "source_kind": "pack_stats"}  # type: Dict[str, Any]
    for (name, cast), value in zip(FIELDS, values):
        if cast is int:
            record[name] = int(value)
        else:
            record[name] = float(value)
    record["warnings"] = _validate(record)
    record["valid"] = not record["warnings"]
    return record


def _parse_shape_summary(tag: str, body: str) -> Dict[str, Any]:
    groups = _parse_shape_groups(body)
    if len(groups) != 11:
        raise ValueError(f"expected 11 bracket groups in shape summary, got {len(groups)}")

    scalars = []
    for idx, group in enumerate(groups[4:], start=4):
        if len(group) != 1:
            raise ValueError(f"shape summary group {idx} should be scalar, got {group}")
        scalars.append(group[0])

    n_seg, sum_q, sum_k, q_min, q_max, k_min, k_max = scalars
    q_dense_by_q_max = n_seg * q_max
    k_dense_by_k_max = n_seg * k_max
    dropped_q_by_q_max = q_dense_by_q_max - sum_q
    dropped_k_by_k_max = k_dense_by_k_max - sum_k

    warnings = [
        "derived from raw_q/raw_k shape summary; total_user and invalid_user_ratio are not available",
        "q/k drop ratios use observed q_max/k_max as per-valid-user dense slots; use explicit pack_stats for exact global ratios",
    ]
    if q_dense_by_q_max < sum_q:
        warnings.append("n_seg*q_max is smaller than sum_q; q_max field may be inconsistent")
    if k_dense_by_k_max < sum_k:
        warnings.append("n_seg*k_max is smaller than sum_k; k_max field may be inconsistent")

    return {
        "tag": tag,
        "source_kind": "shape_summary",
        "raw_q_shape": groups[0],
        "raw_k_shape": groups[1],
        "q_swiglu_shape": groups[2],
        "k_swiglu_shape": groups[3],
        "valid_user": n_seg,
        "n_seg": n_seg,
        "packed_q": sum_q,
        "packed_k": sum_k,
        "sum_q": sum_q,
        "sum_k": sum_k,
        "q_min": q_min,
        "q_max": q_max,
        "k_min": k_min,
        "k_max": k_max,
        "q_dense_slots_by_q_max": q_dense_by_q_max,
        "dropped_q_by_q_max": dropped_q_by_q_max,
        "q_drop_ratio_by_q_max": _ratio(dropped_q_by_q_max, q_dense_by_q_max),
        "k_dense_slots_by_k_max": k_dense_by_k_max,
        "dropped_k_by_k_max": dropped_k_by_k_max,
        "k_drop_ratio_by_k_max": _ratio(dropped_k_by_k_max, k_dense_by_k_max),
        "warnings": warnings,
        "valid": True,
    }


def _validate(record: Dict[str, Any]) -> List[str]:
    warnings = []

    count_checks = [
        ("invalid_user", record["total_user"] - record["valid_user"]),
        ("dropped_query", record["total_query"] - record["valid_query"]),
        ("dropped_k_global", record["total_k_slot"] - record["packed_k"]),
        ("dropped_k_valid_user", record["valid_user_k_slot"] - record["packed_k"]),
    ]
    for key, expected in count_checks:
        if record[key] != expected:
            warnings.append(f"{key}={record[key]} != expected {expected}")

    ratio_checks = [
        ("invalid_user_ratio", record["invalid_user"], record["total_user"]),
        ("query_drop_ratio", record["dropped_query"], record["total_query"]),
        ("k_drop_ratio_global", record["dropped_k_global"], record["total_k_slot"]),
        (
            "k_drop_ratio_valid_user",
            record["dropped_k_valid_user"],
            record["valid_user_k_slot"],
        ),
    ]
    for key, num, den in ratio_checks:
        expected = _ratio(num, den)
        if not _close(record[key], expected):
            warnings.append(f"{key}={record[key]:.8g} != expected {expected:.8g}")

    if record["valid_user"] > record["total_user"]:
        warnings.append("valid_user is greater than total_user")
    if record["valid_query"] > record["total_query"]:
        warnings.append("valid_query is greater than total_query")
    if record["packed_k"] > record["total_k_slot"]:
        warnings.append("packed_k is greater than total_k_slot")
    if record["packed_k"] > record["valid_user_k_slot"]:
        warnings.append("packed_k is greater than valid_user_k_slot")

    return warnings


def parse_log(path: Path, tag_filter: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    records = []  # type: List[Dict[str, Any]]
    warnings = []  # type: List[str]
    seen_tag_lines = 0
    seen_pack_stats_lines = 0
    seen_shape_summary_lines = 0

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            match = TAG_RE.search(line)
            if match is None:
                continue

            tag = match.group("tag")
            if tag_filter is not None and tag != tag_filter:
                continue
            seen_tag_lines += 1

            body = match.group("body").strip()
            try:
                if body.startswith(PACK_STATS_PREFIX) or body.startswith("pack_stats"):
                    seen_pack_stats_lines += 1
                    record = _parse_pack_stats(
                        tag,
                        _payload_after_colon(body, "pack_stats"),
                    )
                elif body.startswith(SHAPE_SUMMARY_PREFIX) or body.startswith("raw_q"):
                    seen_shape_summary_lines += 1
                    record = _parse_shape_summary(
                        tag,
                        _payload_after_colon(body, SHAPE_SUMMARY_PREFIX),
                    )
                else:
                    continue
            except ValueError as exc:
                warnings.append(f"line {line_no}: {exc}")
                continue
            records.append(record)

    if not records:
        tag_desc = tag_filter if tag_filter is not None else "any tag"
        if seen_tag_lines == 0:
            warnings.append(
                f"no [TA_SHAPE {tag_desc}] lines found; check the log file, tag, stdout/stderr capture, or whether the tf.Print op executed"
            )
        elif seen_pack_stats_lines == 0 and seen_shape_summary_lines == 0:
            warnings.append(
                f"saw {seen_tag_lines} [TA_SHAPE] line(s), but none were pack_stats or raw_q shape summary lines; check the message text"
            )

    return records, warnings


def _summary_lines(records: List[Dict[str, Any]]) -> str:
    if not records:
        return "no pack_stats records found"

    lines = []
    for i, record in enumerate(records):
        lines.append(
            _summary_line(i, record)
        )
    return "\n".join(lines)


def _summary_line(index: int, record: Dict[str, Any]) -> str:
    if record.get("source_kind") == "shape_summary":
        return " ".join(
            [
                f"record={index}",
                f"tag={record['tag']}",
                "source=shape_summary",
                f"valid_user={record['valid_user']}",
                f"packed_q={record['packed_q']}/{record['q_dense_slots_by_q_max']}",
                f"q_drop_ratio_by_q_max={record['q_drop_ratio_by_q_max']:.6f}",
                f"packed_k={record['packed_k']}/{record['k_dense_slots_by_k_max']}",
                f"k_drop_ratio_by_k_max={record['k_drop_ratio_by_k_max']:.6f}",
                "invalid_user_ratio=NA",
            ]
        )

    return " ".join(
        [
            f"record={index}",
            f"tag={record['tag']}",
            "source=pack_stats",
            f"valid_user={record['valid_user']}/{record['total_user']}",
            f"invalid_user_ratio={record['invalid_user_ratio']:.6f}",
            f"valid_query={record['valid_query']}/{record['total_query']}",
            f"query_drop_ratio={record['query_drop_ratio']:.6f}",
            f"packed_k={record['packed_k']}/{record['total_k_slot']}",
            f"k_drop_ratio_global={record['k_drop_ratio_global']:.6f}",
            f"k_drop_ratio_valid_user={record['k_drop_ratio_valid_user']:.6f}",
        ]
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file", type=Path)
    parser.add_argument("--tag", default=None, help="only parse one TA_SHAPE tag, e.g. long_term5")
    parser.add_argument("--last", action="store_true", help="emit only the last complete record")
    parser.add_argument("--summary", action="store_true", help="emit one compact text line per record")
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args(argv)

    records, warnings = parse_log(args.log_file, tag_filter=args.tag)
    if args.last and records:
        records = [records[-1]]

    if args.summary:
        text = _summary_lines(records)
    else:
        payload = {
            "source": str(args.log_file),
            "record_count": len(records),
            "records": records,
            "parse_warnings": warnings,
        }
        text = json.dumps(payload, indent=2, ensure_ascii=False)

    if args.output is None:
        print(text)
    else:
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}")

    if warnings:
        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())

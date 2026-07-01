#!/usr/bin/env python3
"""
Extract real ragged target-attention shapes from TensorFlow tf.Print logs.

Expected log lines are produced by the three tf.Print calls with prefixes:
  [TA_SHAPE long_term5] raw_q raw_k q_swiglu k_swiglu ...
  [TA_SHAPE long_term5] q_lens:
  [TA_SHAPE long_term5] k_lens:

The script emits JSON records that can be used later as benchmark fixtures.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TAG_RE = re.compile(r"\[TA_SHAPE (?P<tag>[^\]]+)\]\s*(?P<body>.*)")
BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")
INT_RE = re.compile(r"-?\d+")


SUMMARY_PREFIX = (
    "raw_q raw_k q_swiglu k_swiglu "
    "n_seg sum_q sum_k q_min q_max k_min k_max:"
)
Q_LENS_PREFIX = "q_lens:"
K_LENS_PREFIX = "k_lens:"


def _payload_after_colon(body: str, fallback_prefix: str) -> str:
    if ":" in body:
        return body.split(":", 1)[1]
    return body[len(fallback_prefix):]


def _parse_ints(text: str) -> List[int]:
    if "..." in text:
        raise ValueError(
            "found an ellipsis in a printed tensor; increase tf.Print summarize"
        )
    return [int(match.group(0)) for match in INT_RE.finditer(text)]


def _parse_groups(body: str) -> List[List[int]]:
    return [_parse_ints(group) for group in BRACKET_RE.findall(body)]


def _empty_record(tag: str) -> Dict[str, Any]:
    return {"tag": tag}


def _parse_summary(tag: str, body: str) -> Dict[str, Any]:
    groups = _parse_groups(body)
    if len(groups) != 11:
        raise ValueError(f"expected 11 bracket groups in summary, got {len(groups)}")

    scalars = []
    for idx, group in enumerate(groups[4:], start=4):
        if len(group) != 1:
            raise ValueError(f"summary group {idx} should be scalar, got {group}")
        scalars.append(group[0])

    n_seg, sum_q, sum_k, q_min, q_max, k_min, k_max = scalars
    return {
        "tag": tag,
        "raw_q_shape": groups[0],
        "raw_k_shape": groups[1],
        "q_swiglu_shape": groups[2],
        "k_swiglu_shape": groups[3],
        "n_seg": n_seg,
        "sum_q": sum_q,
        "sum_k": sum_k,
        "q_min": q_min,
        "q_max": q_max,
        "k_min": k_min,
        "k_max": k_max,
    }


def _attach_lens(record: Dict[str, Any], key: str, body: str) -> None:
    groups = _parse_groups(body)
    if len(groups) != 1:
        raise ValueError(f"expected one bracket group for {key}, got {len(groups)}")
    record[key] = groups[0]


def _is_complete(record: Dict[str, Any]) -> bool:
    return all(key in record for key in ("raw_q_shape", "q_lens", "k_lens"))


def _validate(record: Dict[str, Any]) -> List[str]:
    warnings = []
    q_lens = record.get("q_lens")
    k_lens = record.get("k_lens")
    if not isinstance(q_lens, list) or not isinstance(k_lens, list):
        return ["record is incomplete"]

    n_seg = record.get("n_seg")
    sum_q = record.get("sum_q")
    sum_k = record.get("sum_k")
    if n_seg is not None and len(q_lens) != n_seg:
        warnings.append(f"len(q_lens)={len(q_lens)} != n_seg={n_seg}")
    if n_seg is not None and len(k_lens) != n_seg:
        warnings.append(f"len(k_lens)={len(k_lens)} != n_seg={n_seg}")
    if sum_q is not None and sum(q_lens) != sum_q:
        warnings.append(f"sum(q_lens)={sum(q_lens)} != sum_q={sum_q}")
    if sum_k is not None and sum(k_lens) != sum_k:
        warnings.append(f"sum(k_lens)={sum(k_lens)} != sum_k={sum_k}")

    if q_lens:
        if record.get("q_min") is not None and min(q_lens) != record["q_min"]:
            warnings.append(f"min(q_lens)={min(q_lens)} != q_min={record['q_min']}")
        if record.get("q_max") is not None and max(q_lens) != record["q_max"]:
            warnings.append(f"max(q_lens)={max(q_lens)} != q_max={record['q_max']}")
    if k_lens:
        if record.get("k_min") is not None and min(k_lens) != record["k_min"]:
            warnings.append(f"min(k_lens)={min(k_lens)} != k_min={record['k_min']}")
        if record.get("k_max") is not None and max(k_lens) != record["k_max"]:
            warnings.append(f"max(k_lens)={max(k_lens)} != k_max={record['k_max']}")

    q_shape = record.get("q_swiglu_shape")
    k_shape = record.get("k_swiglu_shape")
    if isinstance(q_shape, list) and len(q_shape) >= 2 and sum_q is not None:
        if q_shape[1] != sum_q:
            warnings.append(f"q_swiglu_shape[1]={q_shape[1]} != sum_q={sum_q}")
    if isinstance(k_shape, list) and len(k_shape) >= 2 and sum_k is not None:
        if k_shape[1] != sum_k:
            warnings.append(f"k_swiglu_shape[1]={k_shape[1]} != sum_k={sum_k}")

    return warnings


def parse_log(path: Path, tag_filter: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    records = []  # type: List[Dict[str, Any]]
    warnings = []  # type: List[str]
    current_by_tag = {}  # type: Dict[str, Dict[str, Any]]
    seen_tag_lines = 0
    seen_shape_lines = 0

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
                if body.startswith(SUMMARY_PREFIX) or body.startswith("raw_q"):
                    seen_shape_lines += 1
                    current = _parse_summary(tag, _payload_after_colon(body, SUMMARY_PREFIX))
                    old = current_by_tag.get(tag)
                    if old is not None and not _is_complete(old):
                        warnings.append(
                            f"line {line_no}: dropping incomplete record for tag {tag!r}"
                        )
                    current_by_tag[tag] = current
                elif body.startswith(Q_LENS_PREFIX) or body.startswith("q_lens"):
                    seen_shape_lines += 1
                    current = current_by_tag.setdefault(tag, _empty_record(tag))
                    _attach_lens(current, "q_lens", _payload_after_colon(body, Q_LENS_PREFIX))
                elif body.startswith(K_LENS_PREFIX) or body.startswith("k_lens"):
                    seen_shape_lines += 1
                    current = current_by_tag.setdefault(tag, _empty_record(tag))
                    _attach_lens(current, "k_lens", _payload_after_colon(body, K_LENS_PREFIX))
                else:
                    continue
            except ValueError as exc:
                warnings.append(f"line {line_no}: {exc}")
                continue

            current = current_by_tag.get(tag)
            if current is not None and _is_complete(current):
                current["warnings"] = _validate(current)
                current["valid"] = not current["warnings"]
                records.append(current)
                del current_by_tag[tag]

    for tag, current in current_by_tag.items():
        if current:
            warnings.append(f"end of file: incomplete record for tag {tag!r}")

    if not records:
        tag_desc = tag_filter if tag_filter is not None else "any tag"
        if seen_tag_lines == 0:
            warnings.append(
                f"no [TA_SHAPE {tag_desc}] lines found; check the log file, tag, stdout/stderr capture, or whether the tf.Print op executed"
            )
        elif seen_shape_lines == 0:
            warnings.append(
                f"saw {seen_tag_lines} [TA_SHAPE] line(s), but none were shape/q_lens/k_lens lines; check the message text or use parse_ta_pack_stats_log.py"
            )

    return records, warnings


def _python_lists(record: Dict[str, Any]) -> str:
    q_lens = record["q_lens"]
    k_lens = record["k_lens"]
    d_q = record.get("q_swiglu_shape", [None, None, None])[-1]
    d_kv = record.get("k_swiglu_shape", [None, None, None])[-1]
    return "\n".join(
        [
            f"q_lens = {q_lens}",
            f"k_lens = {k_lens}",
            f"d_q = {d_q}",
            f"d_kv = {d_kv}",
        ]
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file", type=Path)
    parser.add_argument("--tag", default=None, help="only parse one TA_SHAPE tag, e.g. long_term5")
    parser.add_argument("--last", action="store_true", help="emit only the last complete record")
    parser.add_argument("--python-lists", action="store_true", help="emit q_lens/k_lens Python assignments")
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args(argv)

    records, warnings = parse_log(args.log_file, tag_filter=args.tag)
    if args.last and records:
        records = [records[-1]]

    if args.python_lists:
        if not records:
            print("no complete TA_SHAPE records found", file=sys.stderr)
            return 1
        text = "\n\n".join(_python_lists(record) for record in records)
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust filter:
  • Keeps only items that are either seed ASINs from a CSV or their related ASINs (provided they exist in the dataset).
  • Compatible with multiple item formats (JSON array / JSONL / Python repr with single quotes).
  • **Fix**: Adapts to CSV delimiters (comma/tab/semicolon/pipe/whitespace) to avoid errors
    like "Found: ['asin\\titemID']".

Run main() directly, or import and call filter_items(...).
"""

import csv
import gzip
import io
import json
import sys
import ast
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple, Union


# ---------- IO helpers ----------

def open_maybe_gzip(path: Union[str, Path], mode: str = "rt", encoding: str = "utf-8"):
    """Opens a file, transparently handling .gz compression."""
    path = str(path)
    if path.endswith(".gz"):
        if "b" in mode:
            return gzip.open(path, mode)
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding=encoding, newline="")
    return open(path, mode, encoding=encoding, newline="")


# ---------- CSV (seeds) ----------

def _detect_delimiter(header_line: str) -> str:
    """
    Try to infer the delimiter from the header line. Supports: comma, tab, semicolon, pipe, whitespace.
    """
    # Normalize BOM
    header_line = header_line.lstrip("\ufeff").rstrip("\n\r")

    # If it contains obvious delimiter characters, pick the one with the max count
    candidates = [",", "\t", ";", "|"]
    counts = {d: header_line.count(d) for d in candidates}
    best = max(counts, key=counts.get)
    if counts[best] > 0:
        return best

    # Fallback: treat as whitespace separated
    return "whitespace"


def _split_row(line: str, delimiter: str) -> List[str]:
    """Splits a row string based on the detected delimiter."""
    line = line.rstrip("\n\r")
    if delimiter == "whitespace":
        # Split on any run of whitespace
        return re.split(r"\s+", line.strip())
    else:
        return line.split(delimiter)


def read_seeds_csv(csv_path: Union[str, Path]) -> Set[str]:
    """
    Robust CSV reader that:
      - Automatically detects the delimiter (, / \t / ; / | / whitespace).
      - Supports files with a BOM.
      - Handles case-insensitive and stripped column names.
      - Requires an 'asin' column to be present.
    """
    seeds: Set[str] = set()
    with open_maybe_gzip(csv_path, "rt", encoding="utf-8") as f:
        # Peek at the header to determine the delimiter
        pos = f.tell()
        header_line = f.readline()
        if not header_line:
            raise ValueError(f"CSV '{csv_path}' is empty.")
        delimiter = _detect_delimiter(header_line)

        # Parse header into a list of column names
        header_cols = [c.strip().lower() for c in _split_row(header_line.lstrip("\ufeff"), delimiter)]
        # Find the index of the 'asin' column
        try:
            asin_idx = header_cols.index("asin")
        except ValueError:
            # Provide a more helpful error with the raw header
            raise ValueError(
                f"CSV '{csv_path}' must include a column named 'asin'. "
                f"Found header: {header_line.strip()}"
            )

        # Read the remaining lines
        for i, line in enumerate(f, start=2):
            if not line.strip():
                continue
            parts = _split_row(line, delimiter)
            if asin_idx >= len(parts):
                # Skip malformed lines
                continue
            asin = parts[asin_idx].strip().strip('"').strip("'")
            if asin:
                seeds.add(asin)
    return seeds


# ---------- Items dataset ----------

def parse_obj_strict_or_repr(s: str) -> dict:
    """Parses a string as either strict JSON or a Python literal, expecting a dict."""
    s = s.strip()
    if not s:
        raise ValueError("Empty string.")
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    obj = ast.literal_eval(s)
    if isinstance(obj, dict):
        return obj
    raise ValueError("Parsed object is not a dict")


def parse_array_strict_or_repr(text: str) -> List[dict]:
    """Parses a string as either a strict JSON array or a Python list literal."""
    text = text.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    except Exception:
        pass
    data = ast.literal_eval(text)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def read_items_dataset(items_path: Union[str, Path]) -> Dict[str, dict]:
    """
    Reads an items dataset, supporting multiple formats:
      - A single JSON array of objects.
      - JSON Lines (one JSON object per line).
      - A Python list repr string of dicts.
      - One Python dict repr (with single quotes) per line.
    """
    asin_to_item: Dict[str, dict] = {}
    with open_maybe_gzip(items_path, "rt", encoding="utf-8") as f:
        head = f.read(4096)
        f.seek(0)
        stripped = head.lstrip()
        first_char = stripped[:1]

        if first_char == "[":  # Likely a JSON array
            try:
                arr = json.load(f)
            except Exception:
                f.seek(0)
                arr = parse_array_strict_or_repr(f.read())
            for obj in arr:
                asin = str(obj.get("asin", "")).strip()
                if asin:
                    asin_to_item[asin] = obj
            return asin_to_item

        if first_char == "{":  # Likely JSON Lines
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = parse_obj_strict_or_repr(line)
                asin = str(obj.get("asin", "")).strip()
                if asin:
                    asin_to_item[asin] = obj
            return asin_to_item

        # Fallback: treat the entire file as a Python list repr
        arr = parse_array_strict_or_repr(f.read())
        for obj in arr:
            asin = str(obj.get("asin", "")).strip()
            if asin:
                asin_to_item[asin] = obj
        return asin_to_item


# ---------- Core logic ----------

def extract_related_asins(item: dict) -> Set[str]:
    """Extracts all related ASINs from an item's 'related' field."""
    out: Set[str] = set()
    rel = item.get("related")
    if not isinstance(rel, dict):
        return out
    for value in rel.values():
        if isinstance(value, list):
            for x in value:
                if isinstance(x, str) and x.strip():
                    out.add(x.strip())
        elif isinstance(value, str):
            if value.strip():
                out.add(value.strip())
    return out


def build_keep_set(seeds: Set[str], asin_to_item: Dict[str, dict]) -> Set[str]:
    """Builds a set of ASINs to keep: seeds + their related items that exist in the dataset."""
    keep: Set[str] = set()
    missing_seeds = 0
    for s in seeds:
        if s in asin_to_item:
            keep.add(s)
            for r in extract_related_asins(asin_to_item[s]):
                if r in asin_to_item: # only keep related ASINs if they are in the dataset
                    keep.add(r)
        else:
            missing_seeds += 1
    if missing_seeds:
        print(f"[WARN] {missing_seeds} seed ASIN(s) not found in the items dataset.", file=sys.stderr)
    return keep


def write_output(out_path: Union[str, Path], kept_items: Iterable[dict], json_array: bool = False):
    """Writes the kept items to the output path, either as JSONL or a JSON array."""
    if json_array:
        data = list(kept_items)
        with open_maybe_gzip(out_path, "wt", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")
    else: # JSONL
        with open_maybe_gzip(out_path, "wt", encoding="utf-8") as f:
            for obj in kept_items:
                f.write(json.dumps(obj, ensure_ascii=False))
                f.write("\n")


def filter_items(seeds_csv: str, items_path: str, out_path: str, json_array: bool = False):
    """Main filtering function."""
    asin_to_item = read_items_dataset(items_path)
    seeds = read_seeds_csv(seeds_csv)

    keep_set = build_keep_set(seeds, asin_to_item)

    total_items = len(asin_to_item)
    kept_list = [asin_to_item[a] for a in keep_set if a in asin_to_item]
    kept_count = len(kept_list)
    removed_count = total_items - kept_count

    write_output(out_path, kept_list, json_array=json_array)

    print(f"Total items in input  : {total_items}")
    print(f"Seed ASINs provided   : {len(seeds)}")
    print(f"Kept items (seed+rel) : {kept_count}")
    print(f"Removed items         : {removed_count}")



def main():
    # Paths are hardcoded here, modify as needed.
    seeds_csv = "i_id_mapping.csv"        # Seed CSV file.
    items_path = "clothing_meta.json"     # Original items file (json/jsonl/single-quoted dicts are all fine).
    out_path = "clothing.jsonl"           # Output file.
    json_array = False                    # If True, output as a single JSON array, otherwise as JSONL.

    filter_items(seeds_csv, items_path, out_path, json_array)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import gzip
import json
import ast
import csv
from typing import Any, Dict, Iterator, List, Set, Optional


def log(msg: str) -> None:
    """Prints a message to stderr."""
    print(msg, file=sys.stderr, flush=True)


def _open_maybe_gz(path: str):
    """Opens a file, handling .gz compression transparently."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore")


def parse_loose_obj(s: str) -> Optional[Dict[str, Any]]:
    """Parses a string as either JSON or a Python literal, expecting a dict."""
    s = s.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
        elif isinstance(obj, list):
            return None # Skip lines that are just lists
    except json.JSONDecodeError:
        pass
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None


def iter_objects_any_format(path: str) -> Iterator[Dict[str, Any]]:
    """Iterates over objects from a file, supporting NDJSON, JSON array, and Python literal formats."""
    # First, try to parse line-by-line (NDJSON)
    good = 0
    bad = 0
    with _open_maybe_gz(path) as f:
        for line in f:
            obj = parse_loose_obj(line)
            if obj is None:
                bad += 1
                if bad > 5 and good == 0:
                    # Looks like a single bulk file, break to the next phase
                    break
                continue
            good += 1
            yield obj
        else:
            # If the loop completes normally, we're done.
            return

    # Fallback: attempt to parse the entire file at once
    log(f"[fallback] Attempting to parse {path} as a whole.")
    with _open_maybe_gz(path) as f:
        text = f.read()

    try: # As a single JSON object/array
        data = json.loads(text)
        if isinstance(data, list):
            for x in data:
                if isinstance(x, dict):
                    yield x
        elif isinstance(data, dict):
            if "asin" in data:
                yield data
        return
    except json.JSONDecodeError:
        pass

    try: # As a single Python literal
        data = ast.literal_eval(text)
        if isinstance(data, list):
            for x in data:
                if isinstance(x, dict):
                    yield x
        elif isinstance(data, dict) and "asin" in data:
            yield data
    except Exception:
        log(f"[error] Failed to parse {path}. It is neither NDJSON nor a parsable bulk JSON file.")
        return


def build_existing_asin_set(catalog_path: str) -> Set[str]:
    """Scans the catalog to build a set of all existing ASINs."""
    exists: Set[str] = set()
    cnt = 0
    for obj in iter_objects_any_format(catalog_path):
        if not isinstance(obj, dict):
            continue
        asin = str(obj.get("asin", "")).strip()
        if asin:
            exists.add(asin)
        cnt += 1
        if cnt and cnt % 200000 == 0:
            log(f"[index] Indexed {cnt} items, unique ASINs so far: {len(exists)}")
    log(f"[done] Indexing complete. Total unique ASINs: {len(exists)}")
    return exists


def merge_related_asins(obj: Dict[str, Any], existing: Set[str]) -> List[str]:
    """Extracts, cleans, and filters related ASINs against an existing set."""
    related = obj.get("related") or {}
    ab = related.get("also_bought") or []
    av = related.get("also_viewed") or []

    def norm_list(lst) -> List[str]:
        out: List[str] = []
        for x in lst:
            if isinstance(x, (str, int)):
                s = str(x).strip()
                if s:
                    out.append(s)
        return out

    ab = norm_list(ab)
    av = norm_list(av)

    self_asin = str(obj.get("asin", "")).strip()
    merged: List[str] = []
    seen: Set[str] = set()
    for src in (ab, av):
        for a in src:
            if a == self_asin:
                continue
            if a in existing and a not in seen:
                seen.add(a)
                merged.append(a)
    return merged


def read_id_mapping(csv_path: str) -> Dict[str, str]:
    """Reads an ASIN-to-ID mapping from a CSV, with robust column detection."""
    preferred_id_names = [
        "id", "item_id", "itemid", "product_id", "idx", "numeric_id"
    ]
    asin2id: Dict[str, str] = {}
    total = 0

    with open(csv_path, "rt", encoding="utf-8", errors="ignore") as f:
        sniffer = csv.Sniffer()
        sample = f.read(2048)
        f.seek(0)
        try:
            dialect = sniffer.sniff(sample)
            reader = csv.DictReader(f, dialect=dialect)
        except Exception:
            # Fallback to default CSV reader if sniffer fails
            reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise RuntimeError("[idmap] CSV has no header or is unreadable.")

        cols = {c.lower().strip(): c for c in reader.fieldnames}
        asin_col = cols.get("asin")

        id_col = None
        # Try preferred names first
        for cand in preferred_id_names:
            if cand in cols:
                id_col = cols[cand]
                break
        # Fallback: find a column with "id" in its name
        if not id_col:
            for lower_name, orig_name in cols.items():
                if lower_name != "asin" and "id" in lower_name:
                    id_col = orig_name
                    break
        # Fallback: if only two columns, the non-"asin" one is the id
        if not id_col and len(reader.fieldnames) == 2:
            for name in reader.fieldnames:
                if name.lower().strip() != "asin":
                    id_col = name
                    break
        if not asin_col or not id_col:
            raise RuntimeError(f"[idmap] Required columns not found. Detected columns: {reader.fieldnames}")

        log(f"[idmap] Identified columns: asin='{asin_col}', id='{id_col}'")

        for row in reader:
            total += 1
            a = str(row.get(asin_col, "")).strip()
            i = str(row.get(id_col, "")).strip()
            if a and i:
                asin2id[a] = i

    log(f"[idmap] Read {len(asin2id)} mappings (total rows in CSV: {total}).")
    return asin2id


def main():
    # ==== Modify parameters here ====
    catalog_path = "clothing.jsonl"             # Input catalog file
    idmap_path = "i_id_mapping_origin.csv" # ASIN->ID mapping table
    out_path = "related_by_id.json"          # Output JSON file
    missing_out = "related_by_id_missing.txt"  # Output for unmapped ASINs
    # ================================

    log(f"[step] First scan: building ASIN index...")
    existing = build_existing_asin_set(catalog_path)

    log(f"[step] Reading ID mapping: {idmap_path}")
    asin2id = read_id_mapping(idmap_path)

    log(f"[step] Second scan: generating ID-to-ID mapping (in input order)...")
    mapping_by_id: Dict[str, List[str]] = {}
    missing_asins: Set[str] = set()
    cnt = 0
    seeds_total = 0
    seeds_kept = 0
    dup_seed_ids = 0

    for obj in iter_objects_any_format(catalog_path):
        if not isinstance(obj, dict):
            continue
        seed_asin = str(obj.get("asin", "")).strip()
        if not seed_asin:
            continue
        seeds_total += 1

        seed_id = asin2id.get(seed_asin)
        if not seed_id:
            missing_asins.add(seed_asin)
            continue

        # NEW: Only consider seed items with seed_id <= 7049; skip others.
        try:
            seed_id_int = int(seed_id)
        except ValueError:
            # If the mapped seed_id is not an integer, count as missing and skip.
            missing_asins.add(seed_asin)
            continue
        if seed_id_int > 7049:
            # Do not consider seed_items with id > 7049
            continue

        related_asins = merge_related_asins(obj, existing)
        related_ids: List[str] = []
        seen_ids: Set[str] = set()
        for a in related_asins:
            i = asin2id.get(a)
            if not i:
                missing_asins.add(a)
                continue
            if i not in seen_ids:
                seen_ids.add(i)
                related_ids.append(i)

        # If the same seed_id has already appeared, prioritize the first one encountered.
        if seed_id in mapping_by_id:
            dup_seed_ids += 1
            continue

        mapping_by_id[seed_id] = related_ids
        seeds_kept += 1
        cnt += 1
        if cnt and cnt % 100000 == 0:
            log(f"[progress] Processed {cnt} items; total seeds={seeds_total}, kept={seeds_kept}, duplicate seed_ids={dup_seed_ids}")


    log(f"[stat] Total seeds={seeds_total}, output seeds={seeds_kept}, duplicate seed_ids skipped={dup_seed_ids}")
    log(f"[stat] Total number of unmapped ASINs={len(missing_asins)}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wt", encoding="utf-8") as f:
        json.dump(mapping_by_id, f, ensure_ascii=False)
    log(f"[ok] Output written to: {out_path} (contains {len(mapping_by_id)} seed_ids)")

    os.makedirs(os.path.dirname(os.path.abspath(missing_out)), exist_ok=True)
    with open(missing_out, "wt", encoding="utf-8") as f:
        for a in sorted(missing_asins):
            f.write(a + "\n")
    log(f"[ok] List of missing ASINs written to: {missing_out} (contains {len(missing_asins)} ASINs)")


if __name__ == "__main__":
    main()
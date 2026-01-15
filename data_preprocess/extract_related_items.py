#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import gzip
import json
import ast
import argparse
from typing import Any, Dict, Iterator, List, Tuple, Set, Optional

def log(msg: str) -> None:
    """Prints a message to stderr."""
    print(msg, file=sys.stderr, flush=True)

def _open_maybe_gz(path: str):
    """Opens a file, handling .gz compression automatically."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore")

def parse_loose_obj(s: str) -> Optional[Dict[str, Any]]:
    """
    Tries to parse a line into a dict:
    1) JSON (double quotes)
    2) Python literal (single quotes) via ast.literal_eval
    Returns a dict or None.
    """
    s = s.strip()
    if not s:
        return None
    # Standard JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
        elif isinstance(obj, list):
            # If the whole line is a list, skip it; list elements are handled in the bulk parsing phase.
            return None
    except json.JSONDecodeError:
        pass
    # Try Python literal (for compatibility with the provided example format)
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None

def iter_objects_any_format(path: str) -> Iterator[Dict[str, Any]]:
    """
    First, tries to read as a stream of NDJSON. If the whole file doesn't look like NDJSON,
    it falls back to parsing the entire file at once:
    - Large JSON list: Loads with json.loads and yields the dicts within.
    - Large JSON object (rare): Yields it if it contains 'asin'.
    """
    # First, try line-by-line NDJSON / loose object parsing
    good = 0
    bad = 0
    with _open_maybe_gz(path) as f:
        for line in f:
            obj = parse_loose_obj(line)
            if obj is None:
                bad += 1
                if bad > 5 and good == 0:
                    # Looks like a single bulk JSON file, break to the next phase
                    break
                continue
            good += 1
            yield obj
        else:
            # Finished reading normally: it was NDJSON
            return

    # Fallback: parse the whole file
    log(f"[fallback] Attempting to parse {path} as a whole.")
    with _open_maybe_gz(path) as f:
        text = f.read()

    # First, try strict JSON
    try:
        data = json.loads(text)
        if isinstance(data, list):
            for x in data:
                if isinstance(x, dict):
                    yield x
        elif isinstance(data, dict):
            # Also return a single object (rare)
            if "asin" in data:
                yield data
        return
    except json.JSONDecodeError:
        pass

    # Then, try loose Python literal for the whole file
    try:
        data = ast.literal_eval(text)
        if isinstance(data, list):
            for x in data:
                if isinstance(x, dict):
                    yield x
        elif isinstance(data, dict) and "asin" in data:
            yield data
    except Exception:
        # If all else fails
        log(f"[error] Failed to parse {path}, it's neither NDJSON nor a parsable bulk JSON file.")
        return

def load_seed(seed_arg: str, catalog_path: str, allow_missing_seed: bool=False) -> Optional[Dict[str, Any]]:
    """
    seed_arg: A path or an ASIN. If it's an ASIN, search for its full metadata in the catalog.
    """
    if os.path.isfile(seed_arg):
        log(f"[step] Reading seed file: {seed_arg}")
        with open(seed_arg, "rt", encoding="utf-8") as f:
            seed = json.load(f)
        if not isinstance(seed, dict) or "asin" not in seed:
            log("[error] Seed file is not a dictionary containing an 'asin'.")
            sys.exit(2)
        return seed

    seed_asin = seed_arg.strip()
    log(f"[step] Input is an ASIN: {seed_asin}, searching in {catalog_path}...")
    for obj in iter_objects_any_format(catalog_path):
        if not isinstance(obj, dict):
            continue
        asin = str(obj.get("asin", "")).strip()
        if asin == seed_asin:
            log(f"[ok] Found seed item: {asin}")
            return obj

    if allow_missing_seed:
        log(f"[warn] ASIN={seed_asin} not found in {catalog_path}, but allowing missing seed (--allow-missing-seed).")
        return None

    log(f"[error] ASIN={seed_asin} not found in {catalog_path}. Possible reasons:\n"
        f"  1) This ASIN does not belong to this category file.\n"
        f"  2) The catalog file is truncated or has encoding issues.\n"
        f"  3) The ASIN simply does not exist.")
    sys.exit(1)

def collect_related_asins(seed_obj: Dict[str, Any], max_related: int) -> List[str]:
    """
    Collects ASINs from the seed's related.also_bought / also_viewed,
    de-duplicates them, and limits the count.
    """
    related = seed_obj.get("related") or {}
    ab = related.get("also_bought") or []
    av = related.get("also_viewed") or []
    # Defensive: only accept strings or ints
    ab = [str(x).strip() for x in ab if isinstance(x, (str, int))]
    av = [str(x).strip() for x in av if isinstance(x, (str, int))]
    merged: List[str] = []
    seen: Set[str] = set()
    for src in (ab, av):
        for a in src:
            if a and a not in seen:
                seen.add(a)
                merged.append(a)
                if len(merged) >= max_related:
                    return merged
    return merged

def scan_catalog_for_targets(catalog_path: str, targets: Set[str]) -> Dict[str, Dict[str, Any]]:
    """
    Scans the catalog once to retrieve all ASINs that are in the targets set.
    """
    found: Dict[str, Dict[str, Any]] = {}
    if not targets:
        return found
    log(f"[step] Scanning {catalog_path} to extract details for {len(targets)} related ASINs (single pass)...")
    cnt = 0
    for obj in iter_objects_any_format(catalog_path):
        if not isinstance(obj, dict):
            continue
        asin = str(obj.get("asin", "")).strip()
        if asin in targets and asin not in found:
            found[asin] = obj
            if len(found) % 100 == 0:
                log(f"[info] Found {len(found)}/{len(targets)}")
            if len(found) == len(targets):
                break
        cnt += 1
        if cnt % 200000 == 0:
            log(f"[scan] Progress: {cnt} lines (found {len(found)}/{len(targets)})")
    log(f"[done] Scan complete: found {len(found)}/{len(targets)} items.")
    return found

def main():
    ap = argparse.ArgumentParser(description="Extract full details for also_bought/also_viewed items based on an ASIN/seed.json (robust, fault-tolerant version).")
    ap.add_argument("--seed", required=True, help="An ASIN or a JSON file containing an 'asin' field.")
    ap.add_argument("--catalog", required=True, help="The item metadata file (supports .gz, NDJSON, large JSON).")
    ap.add_argument("--out", required=True, help="Output JSON path.")
    ap.add_argument("--max-related", type=int, default=500, help="Maximum number of related ASINs to extract (default: 500).")
    ap.add_argument("--allow-missing-seed", action="store_true", help="Continue even if the seed is not found, outputting only the missing list.")
    args = ap.parse_args()

    # 1) Load/find the seed
    seed_obj = load_seed(args.seed, args.catalog, allow_missing_seed=args.allow_missing_seed)

    seed_asin = None if seed_obj is None else str(seed_obj.get("asin", "")).strip()
    # 2) Collect related ASINs
    if seed_obj is not None:
        target_asins = collect_related_asins(seed_obj, args.max_related)
        log(f"[step] Collected {len(target_asins)} related ASINs.")
    else:
        # If seed is missing but allowed, there's nothing to get related items from.
        target_asins = []

    # 3) Scan the catalog to get full details
    found_map: Dict[str, Dict[str, Any]] = scan_catalog_for_targets(args.catalog, set(target_asins))

    # 4) Tally the missing items
    missing = [a for a in target_asins if a not in found_map]
    for m in missing[:20]:
        log(f"[warn] Related ASIN not found in catalog: {m}")
    if len(missing) > 20:
        log(f"[warn] ...and {len(missing)-20} more missing items not printed individually.")

    # 5) Organize the output
    out_obj = {
        "seed_item": seed_obj,               # Can be None if --allow-missing-seed is used
        "found_items": list(found_map.values()),
        "missing_asins": missing
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wt", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2) # Added indent for readability
    log(f"[ok] Output written to: {args.out}")

if __name__ == "__main__":
    main()
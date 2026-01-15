#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-32B Parallel: Joint Extraction + Evidence-Constrained Enhancement (Sports, merged types version; single output.jsonl file)

Key differences from the original version:
- Canonical product_type changed to the "merged" Sports set (see SYS_JOINT).
- For restricted/weapon words (knife/pepper spray/stun gun/ammunition etc.), product_type is forced to "other",
  and a "restricted_category:*" violation is added, while keeping the copy neutral.
- Supports capacity in Liters: Added normalized.capacity_l (only when there's explicit evidence like "20L/25L...").
- Other logic for structure, concurrency, failure retries, minimal cleaning, and fallback paragraphs remains the same.
"""

import os
import re
import json
import argparse
import asyncio
from typing import Any, Dict, List, Tuple, Optional

# ========= Optional Progress Bar Dependency =========
try:
    from tqdm import tqdm
    _USE_TQDM = True
except Exception:
    _USE_TQDM = False

from openai import AsyncOpenAI  # pip install openai>=1.40.0

# =========================
# Evidence Contract Prompt (Sports Merged Version)
# =========================

SYS_JOINT = """
You are an evidence-bound information extractor and writer.

HARD RULES:
Only extract facts directly supported by verbatim snippets (≤12 words) from TITLE, DESCRIPTION, BRAND, or CATEGORIES; if no evidence, leave the field null or []. Do not invent or imply facts.

Use canonical product types {water_bottle, hydration_pack, backpack, sports_bag,
yoga_mat, resistance_band, jump_rope,
fitness_equipment, strength_training, recovery_tool,
ball, goal_net, protective_gear, helmet, sport_eyewear,
swim_gear, racquet_sport, golf_accessory, fishing_gear,
cycling_accessory, wheel_sports, camping_gear, trekking_pole,
lighting_outdoor, cooler, sports_towel, sports_socks,
sportswear, dance_apparel, other}; assign based on clear textual cues, otherwise default to "other".

For potentially regulated items (e.g., "knife", "tactical knife", "spring assist", "pepper spray", "stun gun", "ammunition"):
- set product_type to "other"
- add a violation tag like "restricted_category: knife"
- keep copy strictly neutral (no marketing or purchase encouragement)

List fields must always be arrays. Do not assign age_range unless an exact numeric or stage is stated.
Normalize units (oz → ml; inch → cm) when explicit; if a backpack/pack capacity in liters is explicit (e.g., "20L"), map to capacity_l (do not infer).

De-duplicate short values. Enhanced text must only use the extracted assertions, avoid marketing superlatives unless explicitly present, and maintain grammatical number consistency. Keep neutral tone; strictly neutral for regulated items.

STYLE:
- Keep outputs concise and concrete. No fluff.
- Avoid unverifiable attributes unless verbatim.
- Output VALID JSON only (no code fences).
"""

JOINT_SCHEMA = """
Output JSON with this schema:
{
  "normalized": {
    "product_type": "string|null",
    "materials": ["string", ...],
    "capacity_ml": number|null,
    "capacity_l": number|null,
    "age_range": "string|null",
    "compatibility": ["string", ...],
    "safety_claims": ["string", ...],
    "made_in": "string|null",
    "intended_use": ["string", ...]
  },

  "key_points": ["bullet 1", "bullet 2", "..."],

  "assertions": [
    {
      "claim": "string",
      "field": "product_type|materials|capacity_ml|capacity_l|age_range|compatibility|safety_claims|made_in|intended_use|other",
      "evidence": "verbatim snippet from input"
    }
  ],

  "enhanced_text": "string",

  "evidence": {
    "product_type": ["..."],
    "materials": ["..."],
    "capacity_ml": ["..."],
    "capacity_l": ["..."],
    "age_range": ["..."],
    "compatibility": ["..."],
    "safety_claims": ["..."],
    "made_in": ["..."],
    "intended_use": ["..."]
  },

  "rejected_claims": ["string", "..."],
  "violations": ["string", "..."]
}

CONSTRAINTS:
- If a field has no evidence, keep it null/[] and do NOT surface it in `assertions` or `enhanced_text`.
- Each assertion must quote its evidence verbatim (short phrases OK).
- If there is any uncertainty, prefer to omit and record it in `rejected_claims`.
- For regulated items (e.g., knives, pepper spray), do not include encouraging language; keep it purely descriptive using only evidence.
"""

# =========================
# Fallback Paragraph (used only when enhanced text is too short)
# =========================

def _fallback_paragraph(norm: Dict[str, Any]) -> str:
    parts = []
    if norm.get("product_type"): parts.append(f"Sports/outdoor {norm['product_type']} with practical details.")
    if norm.get("intended_use"): parts.append(f"Intended use: {', '.join(norm['intended_use'])}.")
    if norm.get("materials"): parts.append(f"Materials: {', '.join(norm['materials'])}.")
    if norm.get("compatibility"): parts.append(f"Compatibility: {', '.join(norm['compatibility'])}.")
    if norm.get("safety_claims"): parts.append(f"Safety: {', '.join(norm['safety_claims'])}.")
    if norm.get("made_in"): parts.append(f"Made in {norm['made_in']}.")
    txt = " ".join(parts) or "Sports/outdoor item with straightforward, utility-focused information."
    if len(txt) < 70:
        txt += " Description remains neutral and evidence-based."
    return txt[:700]

# =========================
# Lightweight Cleanup (no rule-based filtering, just fixes spaces/spelling)
# =========================

SPELLFIX = [
    (re.compile(r"\bstotters\b", re.I), "strollers"),
    (re.compile(r"\bTUtu\b", re.I), "tutu"),
]

def _sanitize_enhanced(enh: str) -> str:
    if not isinstance(enh, str) or not enh.strip():
        return ""
    txt = enh
    for pat, rep in SPELLFIX:
        txt = pat.sub(rep, txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    txt = re.sub(r"\s+([,;.])", r"\1", txt)
    return txt

# =========================
# Result Validation (minimum structural constraints)
# =========================

def _is_list(x) -> bool:
    return isinstance(x, list)

def _is_opt_str(x) -> bool:
    return (x is None) or isinstance(x, str)

def _is_opt_num(x) -> bool:
    return (x is None) or isinstance(x, (int, float))

def _is_obj(x) -> bool:
    return isinstance(x, dict)

def _validate_llm_output(obj: Any) -> Tuple[bool, str]:
    """Returns (ok, reason). If the structure is invalid, provides a reason."""
    if not isinstance(obj, dict):
        return False, "top-level-not-dict"

    must_keys = ["normalized", "key_points", "assertions", "enhanced_text",
                 "evidence", "rejected_claims", "violations"]
    for k in must_keys:
        if k not in obj:
            return False, f"missing-key:{k}"

    if not _is_obj(obj["normalized"]):
        return False, "normalized-not-obj"

    norm = obj["normalized"]
    if "product_type" in norm and not _is_opt_str(norm["product_type"]):
        return False, "normalized.product_type-type"
    if "materials" in norm and not _is_list(norm["materials"]):
        return False, "normalized.materials-type"
    if "capacity_ml" in norm and not _is_opt_num(norm["capacity_ml"]):
        return False, "normalized.capacity_ml-type"
    if "capacity_l" in norm and not _is_opt_num(norm["capacity_l"]):
        return False, "normalized.capacity_l-type"
    if "age_range" in norm and not _is_opt_str(norm["age_range"]):
        return False, "normalized.age_range-type"
    if "compatibility" in norm and not _is_list(norm["compatibility"]):
        return False, "normalized.compatibility-type"
    if "safety_claims" in norm and not _is_list(norm["safety_claims"]):
        return False, "normalized.safety_claims-type"
    if "made_in" in norm and not _is_opt_str(norm["made_in"]):
        return False, "normalized.made_in-type"
    if "intended_use" in norm and not _is_list(norm["intended_use"]):
        return False, "normalized.intended_use-type"

    if not _is_list(obj["key_points"]):
        return False, "key_points-not-list"
    if not _is_list(obj["assertions"]):
        return False, "assertions-not-list"
    if not isinstance(obj["enhanced_text"], str):
        return False, "enhanced_text-not-str"
    if not _is_obj(obj["evidence"]):
        return False, "evidence-not-obj"
    if not _is_list(obj["rejected_claims"]):
        return False, "rejected_claims-not-list"
    if not _is_list(obj["violations"]):
        return False, "violations-not-list"

    return True, "ok"

# =========================
# LLM Client (OpenAI Compatible)
# =========================

class Qwen3Client:
    def __init__(self, api_base="http://localhost:8000/v1", api_key="EMPTY",
                 model="Qwen/Qwen3-32B", temperature=0.1, max_tokens=900,
                 no_think=True, timeout=120):
        self.cli = AsyncOpenAI(base_url=api_base, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.no_think = no_think
        self.timeout = timeout

    def _suffix(self) -> str:
        return " /no_think" if self.no_think else ""

    async def joint(self, item: Dict[str, Any]) -> Dict[str, Any]:
        title = item.get("title")
        desc  = item.get("description")
        brand = item.get("brand")
        cats  = item.get("categories")

        user = (
            f"TITLE: {title}\n"
            f"DESCRIPTION: {desc}\n"
            f"BRAND: {brand}\n"
            f"CATEGORIES: {cats}\n\n" +
            JOINT_SCHEMA + self._suffix()
        )

        msgs = [
            {"role": "system", "content": SYS_JOINT},
            {"role": "user", "content": user},
        ]
        for attempt in range(2):  # First attempt + one compensatory retry
            try:
                resp = await self.cli.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    timeout=self.timeout,
                )
                return json.loads(resp.choices[0].message.content)
            except Exception:
                if attempt == 1:
                    raise
                await asyncio.sleep(0.8)

# =========================
# Parallel Processing (writes to a single output.jsonl + a failure file)
# =========================

async def _write_failure(fail_lock: asyncio.Lock, fail_path: str,
                         asin: Optional[str], reason: str,
                         raw: Optional[Any] = None):
    record = {
        "asin": asin,
        "reason": reason,
    }
    if raw is not None:
        try:
            raw_str = json.dumps(raw, ensure_ascii=False)
            if len(raw_str) > 800:
                raw_str = raw_str[:800] + "…"
            record["raw"] = raw_str
        except Exception:
            record["raw"] = f"<non-serializable:{type(raw).__name__}>"
    async with fail_lock:
        with open(fail_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

async def _generate_and_validate(client: Qwen3Client, item: Dict[str, Any]) -> Dict[str, Any]:
    llm = await client.joint(item)
    ok, reason = _validate_llm_output(llm)
    if not ok:
        raise ValueError(f"schema-invalid:{reason}")
    return llm

async def process_one(client: Qwen3Client, item: Dict[str, Any],
                      writer_lock: asyncio.Lock, out_path: str,
                      min_enhanced_len: int,
                      fail_lock: asyncio.Lock, fail_path: str):
    asin = item.get("asin")

    try:
        try:
            llm = await _generate_and_validate(client, item)
        except Exception as e1:
            try:
                llm = await _generate_and_validate(client, item)
            except Exception as e2:
                await _write_failure(fail_lock, fail_path, asin, f"generate-failed:{e2}", raw=str(e1))
                return
    except Exception as e:
        await _write_failure(fail_lock, fail_path, asin, f"unexpected:{e}")
        return

    try:
        title = item.get("title") or ""
        norm  = llm.get("normalized") or {}
        ev    = llm.get("evidence") or {}
        kps   = (llm.get("key_points") or [])[:7]
        enh   = (llm.get("enhanced_text") or "").strip()
        assertions = llm.get("assertions") or []
        rejected   = llm.get("rejected_claims") or []
        violations = llm.get("violations") or []

        # If the LLM has already flagged it as a restricted category, shorten the fallback slightly to maintain a more neutral tone.
        is_restricted = any(isinstance(v, str) and v.startswith("restricted_category:") for v in violations)

        enh = _sanitize_enhanced(enh)
        if len(enh) < (40 if is_restricted else min_enhanced_len):
            enh = _fallback_paragraph(norm)

        out = {
            "asin": asin,
            "title": title,
            "brand": item.get("brand"),
            "categories": item.get("categories"),
            "normalized": norm,
            "key_points": kps,
            "assertions": assertions,
            "enhanced_text": enh,
            "evidence": ev,
            "rejected_claims": rejected,
            "violations": violations
        }

        line = json.dumps(out, ensure_ascii=False)
        async with writer_lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        await _write_failure(fail_lock, fail_path, asin, f"write-failed:{e}", raw=llm)

async def run_all(args):
    # Read input
    items: List[Dict[str, Any]] = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                items.append(json.loads(line))

    total = len(items)
    if total == 0:
        print("No items found in input_jsonl.")
        return

    # Prepare output file (clear and rewrite)
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    open(args.output_jsonl, "w", encoding="utf-8").close()

    # Path for failure file
    fail_jsonl = args.fail_jsonl or (args.output_jsonl + ".failures.jsonl")
    os.makedirs(os.path.dirname(fail_jsonl) or ".", exist_ok=True)
    open(fail_jsonl, "w", encoding="utf-8").close()

    client = Qwen3Client(
        api_base=args.api_base, api_key=args.api_key, model=args.model,
        temperature=args.temperature, max_tokens=args.max_tokens,
        no_think=bool(args.no_think), timeout=args.timeout
    )

    writer_lock = asyncio.Lock()
    fail_lock   = asyncio.Lock()
    sem = asyncio.Semaphore(args.concurrency)

    async def _runner(it):
        async with sem:
            await process_one(client, it, writer_lock, args.output_jsonl,
                              args.min_enhanced_len, fail_lock, fail_jsonl)

    coros = [_runner(it) for it in items]

    if _USE_TQDM:
        for fut in tqdm(asyncio.as_completed(coros), total=total, desc="Processing", unit="item"):
            await fut
    else:
        completed = 0
        step = max(1, total // 100)
        for fut in asyncio.as_completed(coros):
            await fut
            completed += 1
            if completed % step == 0 or completed == total:
                pct = completed / total * 100.0
                print(f"Progress: {completed}/{total} ({pct:.1f}%)", flush=True)

def main():
    ap = argparse.ArgumentParser(description="Qwen3-32B Parallel: Joint extraction + evidence-constrained enhancement (Sports merged version) -> single file output")
    ap.add_argument("--input_jsonl", required=True, help="Input JSONL file")
    ap.add_argument("--output_jsonl", required=True, help="Output JSONL file (all samples written to the same file)")
    ap.add_argument("--fail_jsonl", default=None, help="JSONL for recording failed samples (default: <output>.failures.jsonl)")
    ap.add_argument("--api_base", default="http://localhost:8000/v1")
    ap.add_argument("--api_key", default="EMPTY")
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max_tokens", type=int, default=900)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--no_think", action="store_true", help="Force /no_think (prioritizes efficiency)")
    ap.add_argument("--min_enhanced_len", type=int, default=60, help="Trigger fallback if enhanced text is shorter than this (in characters)")
    args = ap.parse_args()
    asyncio.run(run_all(args))

if __name__ == "__main__":
    main()
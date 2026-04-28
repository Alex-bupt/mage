import ast
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _resolve_meta_path(dataset_path: str, dataset_name: str) -> str:
    candidates = [
        f"meta_{dataset_name}.json",
        f"meta_{dataset_name.capitalize()}.json",
        f"meta_{dataset_name.title()}.json",
    ]
    for name in candidates:
        p = os.path.join(dataset_path, name)
        if os.path.isfile(p):
            return p

    for name in os.listdir(dataset_path):
        if name.startswith("meta") and name.endswith(".json"):
            return os.path.join(dataset_path, name)

    raise FileNotFoundError(f"No meta json found under {dataset_path}")


def _build_copurchase_edges(meta_path: str, asin2item: Dict[str, int]) -> Dict[int, set]:
    edges = defaultdict(set)
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = ast.literal_eval(line)
            except Exception:
                continue

            asin = obj.get("asin")
            if asin not in asin2item:
                continue
            src = int(asin2item[asin])

            related = obj.get("related", {}) or {}
            for key in ("also_bought", "bought_together"):
                for target_asin in related.get(key, []) or []:
                    if target_asin not in asin2item:
                        continue
                    dst = int(asin2item[target_asin])
                    if dst != src:
                        edges[src].add(dst)

    # Symmetrize for robust matching between history and target.
    for src, dsts in list(edges.items()):
        for dst in dsts:
            edges[dst].add(src)
    return edges


def build_hard_eval_inter_file(
    dataset_path: str,
    dataset_name: str,
    inter_file_name: str,
    field_separator: str = "\t",
    quantile: float = 0.3,
    output_file_name: Optional[str] = None,
    text_raw_file_name: str = "text_feat_raw.npy",
    visual_file_name: str = "image_feat.npy",
    stats_output_path: Optional[str] = None,
) -> Tuple[str, Dict[str, float]]:
    inter_path = os.path.join(dataset_path, inter_file_name)
    if not os.path.isfile(inter_path):
        raise FileNotFoundError(f"Interaction file not found: {inter_path}")

    inter_df = pd.read_csv(inter_path, sep=field_separator)
    required_cols = {"userID", "itemID", "x_label"}
    if not required_cols.issubset(set(inter_df.columns)):
        raise ValueError(f"inter file missing required columns: {required_cols}")

    i_map_path = os.path.join(dataset_path, "i_id_mapping.csv")
    if not os.path.isfile(i_map_path):
        raise FileNotFoundError(f"Item mapping file not found: {i_map_path}")
    i_map_df = pd.read_csv(i_map_path, sep=field_separator)
    asin2item = dict(zip(i_map_df["asin"], i_map_df["itemID"]))

    meta_path = _resolve_meta_path(dataset_path, dataset_name)
    copurchase_edges = _build_copurchase_edges(meta_path, asin2item)

    text_path = os.path.join(dataset_path, text_raw_file_name)
    if not os.path.isfile(text_path):
        # Fallback to encoded text if raw is unavailable.
        text_path = os.path.join(dataset_path, "text_feat.npy")
    visual_path = os.path.join(dataset_path, visual_file_name)

    if not os.path.isfile(text_path):
        raise FileNotFoundError("Neither text_feat_raw.npy nor text_feat.npy exists.")
    if not os.path.isfile(visual_path):
        raise FileNotFoundError(f"Visual feature file not found: {visual_path}")

    x_t = _l2_normalize(np.load(text_path))
    x_v = _l2_normalize(np.load(visual_path))

    train_df = inter_df[inter_df["x_label"] == 0]
    test_df = inter_df[inter_df["x_label"] == 2].copy()

    user_train_hist = train_df.groupby("userID")["itemID"].apply(list).to_dict()

    anchor_text_sims: List[float] = []
    anchor_visual_sims: List[float] = []
    hard_row_index_candidates: List[int] = []

    for idx, row in test_df.iterrows():
        user = int(row["userID"])
        target_item = int(row["itemID"])
        history = user_train_hist.get(user, [])
        if not history:
            continue

        matched_hist = [h for h in history if target_item in copurchase_edges.get(int(h), set())]
        if not matched_hist:
            continue

        sims_t = np.array([float(np.dot(x_t[target_item], x_t[int(h)])) for h in matched_hist], dtype=np.float32)
        sims_v = np.array([float(np.dot(x_v[target_item], x_v[int(h)])) for h in matched_hist], dtype=np.float32)

        # Pick the most complement-like anchor from user history.
        anchor_idx = int(np.argmin((sims_t + sims_v) * 0.5))
        anchor_text_sims.append(float(sims_t[anchor_idx]))
        anchor_visual_sims.append(float(sims_v[anchor_idx]))
        hard_row_index_candidates.append(idx)

    if not hard_row_index_candidates:
        raise RuntimeError("No test interactions satisfy co-purchase condition; cannot build hard subset.")

    text_arr = np.array(anchor_text_sims, dtype=np.float32)
    visual_arr = np.array(anchor_visual_sims, dtype=np.float32)
    text_th = float(np.quantile(text_arr, quantile))
    visual_th = float(np.quantile(visual_arr, quantile))

    hard_test_indices = [
        row_idx
        for row_idx, ts, vs in zip(hard_row_index_candidates, anchor_text_sims, anchor_visual_sims)
        if ts <= text_th and vs <= visual_th
    ]

    hard_test_df = test_df.loc[hard_test_indices]

    keep_df = inter_df[inter_df["x_label"].isin([0, 1])]
    out_df = pd.concat([keep_df, hard_test_df], ignore_index=True)

    if output_file_name is None:
        output_file_name = f"{dataset_name}_hard_eval.inter"
    output_path = os.path.join(dataset_path, output_file_name)
    out_df.to_csv(output_path, sep=field_separator, index=False)

    stats = {
        "dataset": dataset_name,
        "meta_file": os.path.basename(meta_path),
        "source_inter_file": inter_file_name,
        "output_inter_file": output_file_name,
        "hard_quantile": quantile,
        "text_threshold": text_th,
        "visual_threshold": visual_th,
        "num_total_interactions": int(len(inter_df)),
        "num_train_interactions": int((inter_df["x_label"] == 0).sum()),
        "num_valid_interactions": int((inter_df["x_label"] == 1).sum()),
        "num_test_interactions_original": int((inter_df["x_label"] == 2).sum()),
        "num_test_with_copurchase_anchor": int(len(hard_row_index_candidates)),
        "num_test_interactions_hard": int(len(hard_test_df)),
        "hard_test_ratio_vs_original_test": float(len(hard_test_df) / max(1, (inter_df["x_label"] == 2).sum())),
    }

    if stats_output_path is not None:
        os.makedirs(os.path.dirname(stats_output_path), exist_ok=True)
        with open(stats_output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    return output_file_name, stats


if __name__ == "__main__":
    # In-code runner for quick local generation; avoids CLI dependency.
    DATASET_NAME = "dataset"
    DATASET_PATH = "C:/path/to/project/data/dataset"
    SOURCE_INTER = "dataset.inter"
    QUANTILE = 0.3

    out_name, out_stats = build_hard_eval_inter_file(
        dataset_path=DATASET_PATH,
        dataset_name=DATASET_NAME,
        inter_file_name=SOURCE_INTER,
        quantile=QUANTILE,
        stats_output_path="C:/path/to/project/results/dataset_hard_subset_stats.json",
    )
    print("Generated:", out_name)
    print(json.dumps(out_stats, indent=2))

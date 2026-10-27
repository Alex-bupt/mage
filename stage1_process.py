#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Method — text feature offline augmentation
#
# Core idea:
#   Use a co-purchase graph to split neighbors into:
#     - Substitute: high similarity, per-seed >= P85
#     - Complement: low similarity, per-seed <= P40
#   Use a soft conceptor (SVD) to extract residuals orthogonal to the substitute subspace,
#   then inject them into the seed vector to expand semantic coverage.
#
# Usage:
#   Update Config paths and SEED_MAX, then run.
#   Example SEED_MAX values depend on dataset size

import json
import numpy as np
import os
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, List


# ==========================
# 1. Core configuration
# ==========================
@dataclass
class Config:
    # --- File paths ---
    EMB_PATH: str = "data/text_feat.npy"
    RELATED_PATH: str = "data/related_by_id.json"
    OUT_PATH: str = "text_feat_enhanced.npy"

    # --- Range control ---
    SEED_MIN: int = 0
    SEED_MAX: int = 0  # Set to dataset size - 1

    # --- Per-seed percentile thresholds (aligns with similarity_islands.py) ---
    # Each seed ranks neighbors locally to avoid global distribution bias.
    # STRONG_PERCENTILE=85 -> top 15% as substitutes
    # WEAK_PERCENTILE=40  -> bottom 40% as complement candidates
    STRONG_PERCENTILE: float = 95.0
    WEAK_PERCENTILE: float = 45.0

    # --- Complement similarity lower bound ---
    # After percentile split, drop extremely low-similarity neighbors (noise).
    # Retained for backward compatibility; current flow applies soft weighting.
    WEAK_SIM_MIN: float = 0.20

    # --- Soft conceptor parameters ---
    # Larger aperture -> more aggressive filtering of substitute subspace.
    # Smaller aperture -> keep more complement signal.
    # Recommended range: 8.0 ~ 20.0
    CONCEPTOR_APERTURE: float = 12.0

    # --- Gap constraint for alpha projection ---
    # Keep local substitute-complement gap from collapsing after injection.
    # Constraint: gap_after >= GAP_KEEP_RATIO * gap_before
    GAP_KEEP_RATIO: float = 0.60
    MARGIN_EPS: float = 1e-8


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [Method] - %(levelname)s - %(message)s'
)


# ==========================
# 2. Algorithm implementation
# ==========================
class AdvancedPaCE:
    def __init__(self, config: Config):
        self.cfg = config
        self._gamma_sq_inv = config.CONCEPTOR_APERTURE ** (-2)

    def _soft_conceptor_weights(self, s_values: np.ndarray) -> np.ndarray:
        """
        Soft conceptor weights: w_i = s_i^2 / (s_i^2 + gamma^{-2})
        Larger s_i -> weight closer to 1 -> stronger suppression of that direction.
        Smaller s_i -> weight closer to 0 -> direction preserved in residuals.
        """
        s2 = s_values ** 2
        return s2 / (s2 + self._gamma_sq_inv + 1e-12)

    def process_vector(
        self,
        seed_vec: np.ndarray,
        strong_vecs: np.ndarray,
        weak_vecs: np.ndarray,
    ) -> np.ndarray:
        """
        Core enhancement logic returning final_residual.

        Steps:
        1) Build the core subspace from seed + substitutes (SVD)
        2) Filter complements with a soft conceptor to remove the core subspace
        3) Mean residual + orthogonalize against the seed direction
        4) Injection scale is derived from the local gap constraint in the main loop
        """
        # Step 1: Build the core matrix (seed + substitute neighbors)
        core_matrix = (
            np.vstack([seed_vec[None, :], strong_vecs])
            if len(strong_vecs) > 0
            else seed_vec[None, :]
        )

        try:
            _, S, Vt = np.linalg.svd(core_matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            return np.zeros_like(seed_vec)

        soft_weights = self._soft_conceptor_weights(S)

        # Step 2: Soft conceptor filtering
        coeffs = weak_vecs @ Vt.T                                  # (n_weak, rank)
        soft_projections = (coeffs * soft_weights[None, :]) @ Vt   # (n_weak, D)
        residuals = weak_vecs - soft_projections                    # (n_weak, D)

        # Step 3: Mean residual + orthogonalization
        mean_residual = np.mean(residuals, axis=0)
        proj_on_seed = np.dot(mean_residual, seed_vec) * seed_vec
        final_residual = mean_residual - proj_on_seed

        return final_residual


# ==========================
# 3. Data loading
# ==========================
def load_data(cfg: Config) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    if not os.path.exists(cfg.EMB_PATH):
        raise FileNotFoundError(f"Feature file not found: {cfg.EMB_PATH}")
    logging.info(f"Loading embeddings: {cfg.EMB_PATH} ...")
    emb = np.load(cfg.EMB_PATH)

    # Normalize to unit sphere (for cosine similarity)
    norm = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb = emb / norm

    if not os.path.exists(cfg.RELATED_PATH):
        raise FileNotFoundError(f"Relation file not found: {cfg.RELATED_PATH}")

    related_map: Dict[int, List[int]] = {}
    with open(cfg.RELATED_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        for k, v in raw_data.items():
            if not v:
                continue
            try:
                sid = int(k)
                neighs = [int(x) for x in v]
                related_map[sid] = neighs
            except ValueError:
                continue

    logging.info(f"Loaded: {len(emb)} items, {len(related_map)} items have neighbors")
    return emb, related_map


# ==========================
# 4. Main
# ==========================
def main(cfg=None):
    from tqdm import tqdm
    cfg = Config() if cfg is None else cfg
    if not 0.0 <= cfg.GAP_KEEP_RATIO <= 1.0:
        raise ValueError('GAP_KEEP_RATIO must be between 0 and 1 inclusive')
    # Fixed safety cap (kept as code constant, not a tunable config parameter).
    ALPHA_HARD_CAP = 0.35

    full_emb, related_graph = load_data(cfg)
    N_items, D_dim = full_emb.shape
    if cfg.SEED_MAX < 0:
        cfg.SEED_MAX = N_items - 1
    if not 0 <= cfg.SEED_MIN <= cfg.SEED_MAX < N_items:
        raise ValueError('Seed range must lie inside the feature matrix')

    pace = AdvancedPaCE(cfg)

    target_count = cfg.SEED_MAX - cfg.SEED_MIN + 1
    out_mat = np.zeros((target_count, D_dim), dtype=full_emb.dtype)

    logging.info(
        "Per-seed percentile strategy | "
        f"Substitute >= P{cfg.STRONG_PERCENTILE} (top {100 - cfg.STRONG_PERCENTILE:.0f}%) | "
        f"Complement <= P{cfg.WEAK_PERCENTILE} & sim >= {cfg.WEAK_SIM_MIN} | "
        f"Aperture={cfg.CONCEPTOR_APERTURE} | "
        "Alpha=derived_by_gap_constraint | "
        f"GAP_KEEP_RATIO={cfg.GAP_KEEP_RATIO} | "
        f"AlphaHardCap={ALPHA_HARD_CAP}"
    )

    stats = {"enhanced": 0, "skipped_no_neighbor": 0, "skipped_no_weak": 0, "margin_limited": 0}
    diag = {
        "n_strong": [],
        "n_weak": [],           # filtered complement count
        "n_weak_filtered": [],  # dropped by WEAK_SIM_MIN
        "sim_strong": [],
        "sim_weak": [],
        "alpha": [],
        "gap_before": [],
        "gap_target": [],
    }

    for i in tqdm(range(target_count), desc="Embedding Augmentation"):
        seed_id = cfg.SEED_MIN + i
        if seed_id >= N_items:
            continue

        original_vec = full_emb[seed_id]
        neighbors = related_graph.get(seed_id, [])
        valid_neighbors = [n for n in neighbors if 0 <= n < N_items and n != seed_id]

        if not valid_neighbors:
            out_mat[i] = original_vec
            stats["skipped_no_neighbor"] += 1
            continue

        neigh_vecs = full_emb[valid_neighbors]
        sims = np.dot(neigh_vecs, original_vec)

        # Per-seed percentile split
        strong_cut = float(np.percentile(sims, cfg.STRONG_PERCENTILE))
        weak_cut = float(np.percentile(sims, cfg.WEAK_PERCENTILE))
        mask_strong = sims >= strong_cut
        # Complement: percentile lower bound AND absolute similarity lower bound
        mask_weak_pct = sims <= weak_cut
        mask_weak = mask_weak_pct & (sims >= cfg.WEAK_SIM_MIN)

        strong_vecs = neigh_vecs[mask_strong]
        weak_vecs = neigh_vecs[mask_weak]

        n_filtered = int(mask_weak_pct.sum()) - int(mask_weak.sum())

        if len(weak_vecs) == 0:
            out_mat[i] = original_vec
            stats["skipped_no_weak"] += 1
            continue

        final_residual = pace.process_vector(original_vec, strong_vecs, weak_vecs)

        # Gap-constrained alpha projection to avoid gap collapse
        gap_before = float(sims[mask_strong].mean() - sims[mask_weak].mean())
        gap_target = cfg.GAP_KEEP_RATIO * gap_before

        # Constraint-derived alpha (no extra injection hyper-parameter)
        alpha_final = 0.0
        margin_limited = False
        if gap_before > 0.0:
            mu_strong = np.mean(strong_vecs, axis=0)
            mu_weak = np.mean(weak_vecs, axis=0)
            d_gap = float(np.dot(mu_strong - mu_weak, final_residual))
            alpha_margin = 1.0
            if d_gap < 0.0:
                alpha_margin = max((gap_before - gap_target) / (-d_gap + cfg.MARGIN_EPS), 0.0)
            alpha_final = alpha_margin
            margin_limited = d_gap < 0.0 and alpha_final < 1.0

        if margin_limited:
            stats["margin_limited"] += 1

        alpha_final = min(alpha_final, ALPHA_HARD_CAP)

        enhanced_vec = original_vec + alpha_final * final_residual
        enhanced_vec = enhanced_vec / (np.linalg.norm(enhanced_vec) + 1e-12)

        out_mat[i] = enhanced_vec
        stats["enhanced"] += 1

        diag["n_strong"].append(int(mask_strong.sum()))
        diag["n_weak"].append(int(mask_weak.sum()))
        diag["n_weak_filtered"].append(n_filtered)
        diag["sim_strong"].append(float(sims[mask_strong].mean()))
        diag["sim_weak"].append(float(sims[mask_weak].mean()))
        diag["alpha"].append(alpha_final)
        diag["gap_before"].append(gap_before)
        diag["gap_target"].append(gap_target)

    # ---- Output stats ----
    total = target_count
    logging.info("=" * 65)
    logging.info("Enhancement coverage summary")
    logging.info("=" * 65)
    logging.info(f"  Total seeds:           {total}")
    logging.info(f"  Enhanced:              {stats['enhanced']}  ({stats['enhanced'] / total:.2%})")
    logging.info(f"  Skipped - no neighbors: {stats['skipped_no_neighbor']}  ({stats['skipped_no_neighbor'] / total:.2%})")
    logging.info(f"  Skipped - no complements: {stats['skipped_no_weak']}  ({stats['skipped_no_weak'] / total:.2%})")
    logging.info(f"  Gap constraint active (alpha<1): {stats['margin_limited']}  ({stats['margin_limited'] / total:.2%})")

    if diag["n_strong"]:
        arr_ns = np.array(diag["n_strong"])
        arr_nw = np.array(diag["n_weak"])
        arr_nf = np.array(diag["n_weak_filtered"])
        arr_ss = np.array(diag["sim_strong"])
        arr_sw = np.array(diag["sim_weak"])
        arr_al = np.array(diag["alpha"])
        arr_gb = np.array(diag["gap_before"])
        arr_gt = np.array(diag["gap_target"])

        logging.info("-" * 65)
        logging.info("Substitute distribution:")
        logging.info(f"  Count: mean={arr_ns.mean():.1f}, median={np.median(arr_ns):.1f}, "
                     f"min={arr_ns.min()}, max={arr_ns.max()}")
        logging.info(f"  Similarity: mean={arr_ss.mean():.4f}, std={arr_ss.std():.4f}")

        logging.info("Complement distribution (filtered):")
        logging.info(f"  Count: mean={arr_nw.mean():.1f}, median={np.median(arr_nw):.1f}, "
                     f"min={arr_nw.min()}, max={arr_nw.max()}")
        logging.info(f"  Similarity: mean={arr_sw.mean():.4f}, std={arr_sw.std():.4f}")
        logging.info(f"  Dropped neighbors by WEAK_SIM_MIN={cfg.WEAK_SIM_MIN}: "
                     f"mean={arr_nf.mean():.1f}/seed, total={arr_nf.sum()}")

        gap = arr_ss.mean() - arr_sw.mean()
        logging.info(f"Mean gap (sub - comp): {gap:.4f}")
        logging.info("  (Larger gap => clearer substitute/complement separation)")
        logging.info(f"Gap target: mean={arr_gt.mean():.4f} (orig mean={arr_gb.mean():.4f}, keep_ratio={cfg.GAP_KEEP_RATIO})")

        logging.info("Constraint-derived alpha distribution:")
        logging.info(f"  mean={arr_al.mean():.4f}, std={arr_al.std():.4f}, "
                     f"min={arr_al.min():.4f}, max={arr_al.max():.4f}")

    logging.info("=" * 65)

    # ---- Preview first 10 ----
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    logging.info("PREVIEW: first 10 enhanced vectors")
    for i in range(min(10, out_mat.shape[0])):
        vec = out_mat[i]
        print(f"ID {i:05d} | Norm: {np.linalg.norm(vec):.4f} | first 10 dims: {vec[:10]} ...")

    # ---- Save ----
    np.save(cfg.OUT_PATH, out_mat)
    logging.info(f"Saved results to: {cfg.OUT_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Offline text-feature augmentation')
    parser.add_argument('--emb-path', required=True)
    parser.add_argument('--related-path', required=True)
    parser.add_argument('--out-path', required=True)
    parser.add_argument('--seed-max', type=int, default=-1, help='Inclusive last row; -1 processes all rows')
    parser.add_argument('--gap-keep-ratio', type=float, default=Config.GAP_KEEP_RATIO,
                        help='Fraction of the local substitute-complement gap to retain (0 to 1)')
    args = parser.parse_args()
    if not 0.0 <= args.gap_keep_ratio <= 1.0:
        parser.error('--gap-keep-ratio must be between 0 and 1 inclusive')
    if os.path.abspath(args.emb_path) == os.path.abspath(args.out_path):
        parser.error('Use a distinct output path to preserve raw features')
    output_parent = os.path.dirname(os.path.abspath(args.out_path))
    os.makedirs(output_parent, exist_ok=True)
    main(Config(EMB_PATH=args.emb_path, RELATED_PATH=args.related_path, OUT_PATH=args.out_path,
                SEED_MAX=args.seed_max, GAP_KEEP_RATIO=args.gap_keep_ratio))

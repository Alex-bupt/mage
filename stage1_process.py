#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm


# =============================================================================
# 1) Configuration (anonymized / double-blind)
# =============================================================================
@dataclass
class Config:
    # --- Paths (use neutral, non-identifying names) ---
    EMB_PATH: str = "data/embeddings.npy"          # input embedding matrix (N x D)
    RELATED_PATH: str = "data/neighbor_map.json"   # item->neighbors map
    OUT_PATH: str = "data/embeddings_out.npy"      # output embedding matrix

    # --- Target index range (inclusive) ---
    SEED_MIN: int = 0
    SEED_MAX: int = 7049

    # --- Optional automatic threshold calibration ---
    AUTO_CALIBRATE: bool = True
    CALIBRATION_SAMPLES: int = 6000

    # Percentiles used to infer “high-confidence” and “mid-confidence” similarity cutoffs
    HIGH_CONF_PERCENTILE: float = 95.0
    MID_CONF_PERCENTILE: float = 60.0

    # Fallback thresholds (used if calibration fails)
    HIGH_CONF_THRESHOLD: float = 0.85
    MID_CONF_THRESHOLD_LOW: float = 0.55

    # --- Algorithm hyperparameters (kept generic) ---
    # Controls the softness of singular-value attenuation
    APERTURE: float = 12.0
    # Base injection factor for the residual signal
    ALPHA_BASE: float = 0.20
    # Sensitivity factor for uncertainty-adaptive scaling
    UNCERTAINTY_SCALE: float = 1.0

    # --- Safety / performance knobs (neutral) ---
    MAX_NEIGHBORS_PER_SEED_IN_CALIB: int = 50


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [Pipeline] - %(levelname)s - %(message)s",
)


# =============================================================================
# 2) Core algorithm (anonymized / double-blind)
# =============================================================================
class ResidualEnhancer:
    """
    A generic embedding refinement routine:
    - Build a core matrix from (seed + high-confidence neighbors)
    - Compute a soft subspace filter using SVD singular values
    - Extract residual signals from mid-confidence neighbors after soft projection
    - Orthogonalize residual against the seed vector
    - Blend residual back into seed with uncertainty-adaptive scaling
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _soft_sv_weights(self, s_values: np.ndarray) -> np.ndarray:
        gamma_sq_inv = self.cfg.APERTURE ** (-2)
        return (s_values**2) / (s_values**2 + gamma_sq_inv + 1e-12)

    @staticmethod
    def _entropy_of_singular_values(s_values: np.ndarray) -> float:
        if s_values.size == 0:
            return 0.0
        p = s_values / (np.sum(s_values) + 1e-12)
        return float(-np.sum(p * np.log(p + 1e-12)))

    def enhance(
        self,
        seed_vec: np.ndarray,
        high_conf_vecs: np.ndarray,
        mid_conf_vecs: np.ndarray,
    ) -> np.ndarray:
        # Build the core matrix: seed plus high-confidence neighbors (if any)
        if high_conf_vecs.size > 0:
            core_matrix = np.vstack([seed_vec[None, :], high_conf_vecs])
        else:
            core_matrix = seed_vec[None, :]

        # SVD for a low-rank basis; guard against numerical failures
        try:
            _, s_vals, vt = np.linalg.svd(core_matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            return seed_vec

        # Soft attenuation of principal directions
        w = self._soft_sv_weights(s_vals)

        # Project mid-confidence neighbors into the softened subspace and take residuals
        coeffs = mid_conf_vecs @ vt.T
        soft_proj = (coeffs * w[None, :]) @ vt
        residuals = mid_conf_vecs - soft_proj

        # Aggregate residual signal
        mean_residual = np.mean(residuals, axis=0)

        # Orthogonalize residual against the seed direction
        proj_on_seed = np.dot(mean_residual, seed_vec) * seed_vec
        residual_orth = mean_residual - proj_on_seed

        # Uncertainty-adaptive mixing based on singular-value entropy
        ent = self._entropy_of_singular_values(s_vals)
        alpha = self.cfg.ALPHA_BASE * (1.0 + self.cfg.UNCERTAINTY_SCALE * ent)

        enhanced = seed_vec + alpha * residual_orth
        enhanced = enhanced / (np.linalg.norm(enhanced) + 1e-12)
        return enhanced


# =============================================================================
# 3) Utilities (anonymized / double-blind)
# =============================================================================
def load_inputs(cfg: Config) -> tuple[np.ndarray, dict[int, list[int]]]:
    """
    Load and normalize embedding matrix; load neighbor mapping from JSON.
    Neighbor JSON format: { "<int_id>": [<int_neighbor_id>, ...], ... }
    """
    if not os.path.exists(cfg.EMB_PATH):
        raise FileNotFoundError(f"Embedding file not found: {cfg.EMB_PATH}")
    logging.info(f"Loading embeddings: {cfg.EMB_PATH}")
    emb = np.load(cfg.EMB_PATH)

    # Row-wise L2 normalization
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

    if not os.path.exists(cfg.RELATED_PATH):
        raise FileNotFoundError(f"Neighbor map not found: {cfg.RELATED_PATH}")

    logging.info(f"Loading neighbor map: {cfg.RELATED_PATH}")
    with open(cfg.RELATED_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    neighbor_map: dict[int, list[int]] = {}
    for k, v in raw.items():
        if not v:
            continue
        try:
            sid = int(k)
            neighs = [int(x) for x in v]
            neighbor_map[sid] = neighs
        except (ValueError, TypeError):
            continue

    return emb, neighbor_map


def auto_calibrate(cfg: Config, emb: np.ndarray, neighbor_map: dict[int, list[int]]) -> tuple[float, float]:
    """
    Auto-calibrate cosine similarity thresholds by sampling neighbor similarities and
    taking configured percentiles as cutoffs.
    """
    logging.info("Running automatic threshold calibration...")

    keys = list(neighbor_map.keys())
    if not keys:
        logging.warning("Calibration failed: empty neighbor map. Using fallback thresholds.")
        return cfg.HIGH_CONF_THRESHOLD, cfg.MID_CONF_THRESHOLD_LOW

    if len(keys) > cfg.CALIBRATION_SAMPLES:
        sampled = np.random.choice(keys, cfg.CALIBRATION_SAMPLES, replace=False)
    else:
        sampled = keys

    sim_scores: list[float] = []
    n_items = emb.shape[0]

    for sid in sampled:
        sid = int(sid)
        if sid >= n_items:
            continue

        neighs = [n for n in neighbor_map.get(sid, []) if 0 <= n < n_items and n != sid]
        if not neighs:
            continue

        # Bound per-seed workload to avoid outliers dominating runtime
        if len(neighs) > cfg.MAX_NEIGHBORS_PER_SEED_IN_CALIB:
            neighs = np.random.choice(neighs, cfg.MAX_NEIGHBORS_PER_SEED_IN_CALIB, replace=False)

        seed_vec = emb[sid]
        neigh_vecs = emb[np.array(neighs, dtype=np.int64)]
        sims = neigh_vecs @ seed_vec
        sim_scores.extend(sims.tolist())

    if not sim_scores:
        logging.warning("Calibration failed: no similarities computed. Using fallback thresholds.")
        return cfg.HIGH_CONF_THRESHOLD, cfg.MID_CONF_THRESHOLD_LOW

    sims_arr = np.asarray(sim_scores, dtype=np.float64)
    high_th = float(np.percentile(sims_arr, cfg.HIGH_CONF_PERCENTILE))
    mid_th = float(np.percentile(sims_arr, cfg.MID_CONF_PERCENTILE))

    # Safety: ensure high-confidence cutoff is strictly greater than mid-confidence cutoff
    if high_th <= mid_th:
        logging.warning("Calibration anomaly: high_th <= mid_th; applying minimal separation.")
        high_th = mid_th + 0.1

    logging.info(f"Calibration done | #samples={len(sims_arr)}")
    logging.info(f"  -> P{int(cfg.HIGH_CONF_PERCENTILE)} (high): {high_th:.4f}")
    logging.info(f"  -> P{int(cfg.MID_CONF_PERCENTILE)}  (mid) : {mid_th:.4f}")

    return high_th, mid_th


def preview_rows(mat: np.ndarray, count: int = 10) -> None:
    """Print a short preview of the first `count` rows (first 10 dims each)."""
    logging.info("=" * 60)
    logging.info(f"PREVIEW: First {min(count, mat.shape[0])} output vectors")
    logging.info("=" * 60)

    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    limit = min(count, mat.shape[0])

    for i in range(limit):
        vec = mat[i]
        print(f"Row {i:05d} | Norm: {np.linalg.norm(vec):.4f} | First 10 dims: {vec[:10]} ...")
    print()


# =============================================================================
# 4) Main routine (anonymized / double-blind)
# =============================================================================
def main() -> None:
    cfg = Config()

    # Load inputs
    emb, neighbor_map = load_inputs(cfg)
    n_items, d_dim = emb.shape

    # Determine thresholds (auto-calibrated or fallback)
    high_th = cfg.HIGH_CONF_THRESHOLD
    mid_th = cfg.MID_CONF_THRESHOLD_LOW
    if cfg.AUTO_CALIBRATE:
        high_th, mid_th = auto_calibrate(cfg, emb, neighbor_map)

    enhancer = ResidualEnhancer(cfg)

    # Prepare output buffer
    target_count = cfg.SEED_MAX - cfg.SEED_MIN + 1
    out = np.zeros((target_count, d_dim), dtype=emb.dtype)

    logging.info(f"Processing range [{cfg.SEED_MIN}, {cfg.SEED_MAX}]")
    logging.info(f"Using thresholds: high >= {high_th:.4f}, mid >= {mid_th:.4f}")

    stats = {"enhanced": 0, "kept_no_neighbor": 0, "kept_no_mid": 0}

    for i in tqdm(range(target_count), desc="Embedding Refinement"):
        seed_id = cfg.SEED_MIN + i
        if seed_id >= n_items:
            continue

        seed_vec = emb[seed_id]
        neighs = neighbor_map.get(seed_id, [])
        valid = [n for n in neighs if 0 <= n < n_items and n != seed_id]

        if not valid:
            out[i] = seed_vec
            stats["kept_no_neighbor"] += 1
            continue

        neigh_vecs = emb[np.array(valid, dtype=np.int64)]
        sims = neigh_vecs @ seed_vec

        # Partition neighbors by similarity bands
        mask_high = sims >= high_th
        mask_mid = (sims < high_th) & (sims >= mid_th)

        high_vecs = neigh_vecs[mask_high]
        mid_vecs = neigh_vecs[mask_mid]

        if mid_vecs.shape[0] == 0:
            out[i] = seed_vec
            stats["kept_no_mid"] += 1
            continue

        out[i] = enhancer.enhance(seed_vec, high_vecs, mid_vecs)
        stats["enhanced"] += 1

    # Preview
    preview_rows(out, count=10)

    # Save
    logging.info(f"Done. Stats: {stats}")
    logging.info(f"Enhancement ratio: {stats['enhanced'] / max(target_count, 1):.2%}")
    np.save(cfg.OUT_PATH, out)
    logging.info(f"Saved output to: {cfg.OUT_PATH}")


if __name__ == "__main__":
    main()

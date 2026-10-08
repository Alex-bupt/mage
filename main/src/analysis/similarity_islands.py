import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


DEFAULT_DATASETS = ("baby", "sports", "clothing", "elec")
STRONG_PERCENTILE = 95
WEAK_PERCENTILE = 40


@dataclass
class SimilarityStats:
    dataset: str
    modality: str
    substitute_mean: float
    substitute_std: float
    complement_mean: float
    complement_std: float
    gap: float
    substitute_count: int
    complement_count: int


def _normalize_rows(feat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(feat, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return feat / norms


def _safe_std(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.std(values))


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def _display_dataset_name(dataset: str) -> str:
    return dataset.capitalize()


def load_copurchase_graph(dataset: str, graph_root: Path) -> Dict[int, List[int]]:
    dataset_dir = graph_root / dataset
    p2b_path = dataset_dir / "p2b_map_filtered.json"
    inter_path = dataset_dir / "inter.json"

    graph_path = p2b_path if p2b_path.exists() else inter_path
    if not graph_path.exists():
        raise FileNotFoundError(f"Missing co-purchase graph for {dataset}: {p2b_path} / {inter_path}")

    with open(graph_path, "r", encoding="utf-8") as f:
        raw_graph = json.load(f)

    graph: Dict[int, List[int]] = {}
    for key, value in raw_graph.items():
        try:
            src = int(key)
        except (TypeError, ValueError):
            continue

        neighbors: List[int] = []
        if isinstance(value, list):
            for item in value:
                try:
                    nid = int(item)
                except (TypeError, ValueError):
                    continue
                if nid != src:
                    neighbors.append(nid)
        graph[src] = neighbors
    return graph


def load_features(dataset: str, data_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    dataset_dir = data_root / dataset
    text_path = dataset_dir / "text_feat_raw.npy"
    visual_path = dataset_dir / "image_feat.npy"

    if not text_path.exists():
        raise FileNotFoundError(f"Missing required raw text features for {dataset}: {text_path}")
    if not visual_path.exists():
        raise FileNotFoundError(f"Missing required visual features for {dataset}: {visual_path}")

    print(f"[{dataset}] text features: {text_path} (raw)")
    print(f"[{dataset}] visual features: {visual_path}")

    text_feat = np.load(text_path)
    visual_feat = np.load(visual_path)
    return _normalize_rows(text_feat), _normalize_rows(visual_feat)


def get_seed_split(
    seed: int,
    candidates: Sequence[int],
    feat: np.ndarray,
    strong_percentile: int,
    weak_percentile: int,
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    if not candidates:
        return [], []

    valid_candidates = [nid for nid in candidates if 0 <= nid < feat.shape[0] and nid != seed]
    if not valid_candidates:
        return [], []

    sims = (feat[valid_candidates] @ feat[seed]).astype(np.float32)
    strong_cut = float(np.percentile(sims, strong_percentile).item())
    weak_cut = float(np.percentile(sims, weak_percentile).item())

    substitute_pairs = [(nid, float(sim)) for nid, sim in zip(valid_candidates, sims) if sim >= strong_cut]
    complement_pairs = [(nid, float(sim)) for nid, sim in zip(valid_candidates, sims) if sim <= weak_cut]

    substitute_pairs.sort(key=lambda x: x[1], reverse=True)
    complement_pairs.sort(key=lambda x: x[1])
    return substitute_pairs, complement_pairs


def compute_similarity_stats(
    dataset: str,
    modality: str,
    feat: np.ndarray,
    graph: Dict[int, List[int]],
    strong_percentile: int,
    weak_percentile: int,
) -> SimilarityStats:
    substitute_sims: List[float] = []
    complement_sims: List[float] = []

    for seed, neighbors in graph.items():
        if seed < 0 or seed >= feat.shape[0]:
            continue
        substitute_pairs, complement_pairs = get_seed_split(
            seed, neighbors, feat, strong_percentile=strong_percentile, weak_percentile=weak_percentile
        )
        substitute_sims.extend([sim for _, sim in substitute_pairs])
        complement_sims.extend([sim for _, sim in complement_pairs])

    substitute_mean = _safe_mean(substitute_sims)
    substitute_std = _safe_std(substitute_sims)
    complement_mean = _safe_mean(complement_sims)
    complement_std = _safe_std(complement_sims)
    gap = substitute_mean - complement_mean

    return SimilarityStats(
        dataset=_display_dataset_name(dataset),
        modality=modality,
        substitute_mean=substitute_mean,
        substitute_std=substitute_std,
        complement_mean=complement_mean,
        complement_std=complement_std,
        gap=gap,
        substitute_count=len(substitute_sims),
        complement_count=len(complement_sims),
    )


def _format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} +- {std:.4f}"


def save_stats_csv(stats: Sequence[SimilarityStats], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "modality",
                "substitute_mean",
                "substitute_std",
                "complement_mean",
                "complement_std",
                "gap",
                "substitute_count",
                "complement_count",
            ]
        )
        for row in stats:
            writer.writerow(
                [
                    row.dataset,
                    row.modality,
                    f"{row.substitute_mean:.6f}",
                    f"{row.substitute_std:.6f}",
                    f"{row.complement_mean:.6f}",
                    f"{row.complement_std:.6f}",
                    f"{row.gap:.6f}",
                    row.substitute_count,
                    row.complement_count,
                ]
            )


def save_latex_table(stats: Sequence[SimilarityStats], output_txt: Path) -> None:
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{l l l l l}",
        "Dataset & Modality & Substitute (mean$\\pm$std) & Complement (mean$\\pm$std) & Gap \\\\",
        "\\hline",
    ]
    for row in stats:
        lines.append(
            f"{row.dataset} & {row.modality.capitalize()} & "
            f"{row.substitute_mean:.4f} $\\pm$ {row.substitute_std:.4f} & "
            f"{row.complement_mean:.4f} $\\pm$ {row.complement_std:.4f} & "
            f"{row.gap:.4f} \\\\"
        )
    lines.append("\\end{tabular}")

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def print_stats_table(stats: Sequence[SimilarityStats]) -> None:
    header = (
        "Dataset  | Modality | Substitute (mean+-std) | Complement (mean+-std) | Gap",
        "---------|----------|------------------------|------------------------|-----",
    )
    print(header[0])
    print(header[1])
    for row in stats:
        print(
            f"{row.dataset:<8} | {row.modality.capitalize():<8} | "
            f"{_format_mean_std(row.substitute_mean, row.substitute_std):<22} | "
            f"{_format_mean_std(row.complement_mean, row.complement_std):<22} | "
            f"{row.gap:.4f}"
        )


def build_tsne_sample(
    feat: np.ndarray,
    graph: Dict[int, List[int]],
    seed_sample_count: int,
    top_k: int,
    strong_percentile: int,
    weak_percentile: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    candidates: List[int] = [
        seed for seed, neigh in graph.items() if 0 <= seed < feat.shape[0] and len(neigh) > 0
    ]
    if not candidates:
        raise ValueError("No eligible seed items found for t-SNE sampling.")

    rng = np.random.default_rng(random_state)
    seed_count = min(seed_sample_count, len(candidates))
    sampled_seeds = rng.choice(candidates, size=seed_count, replace=False)

    vectors: List[np.ndarray] = []
    labels: List[int] = []

    for seed in sampled_seeds.tolist():
        substitutes, complements = get_seed_split(
            seed,
            graph.get(seed, []),
            feat,
            strong_percentile=strong_percentile,
            weak_percentile=weak_percentile,
        )
        if not substitutes and not complements:
            continue

        vectors.append(feat[seed])
        labels.append(0)

        for nid, _ in substitutes[:top_k]:
            vectors.append(feat[nid])
            labels.append(1)

        for nid, _ in complements[:top_k]:
            vectors.append(feat[nid])
            labels.append(2)

    if len(vectors) < 3:
        raise ValueError("Not enough points to run t-SNE.")

    return np.asarray(vectors), np.asarray(labels)


def save_tsne_plot(
    dataset: str,
    feat: np.ndarray,
    graph: Dict[int, List[int]],
    output_dir: Path,
    seed_sample_count: int,
    top_k: int,
    strong_percentile: int,
    weak_percentile: int,
    random_state: int,
) -> Path:
    vectors, labels = build_tsne_sample(
        feat,
        graph,
        seed_sample_count=seed_sample_count,
        top_k=top_k,
        strong_percentile=strong_percentile,
        weak_percentile=weak_percentile,
        random_state=random_state,
    )

    perplexity = min(30, max(5, len(vectors) // 20))
    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )
    points_2d = tsne.fit_transform(vectors)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"tsne_{dataset}_text.png"

    plt.figure(figsize=(8, 6))
    seed_mask = labels == 0
    sub_mask = labels == 1
    comp_mask = labels == 2

    if np.any(seed_mask):
        plt.scatter(points_2d[seed_mask, 0], points_2d[seed_mask, 1], c="blue", marker="o", s=20, label="Seed")
    if np.any(sub_mask):
        plt.scatter(points_2d[sub_mask, 0], points_2d[sub_mask, 1], c="orange", marker="^", s=18, label="Substitute")
    if np.any(comp_mask):
        plt.scatter(points_2d[comp_mask, 0], points_2d[comp_mask, 1], c="red", marker="x", s=20, label="Complement")

    plt.title(f"Similarity Islands t-SNE ({_display_dataset_name(dataset)} - Text)")
    plt.legend(loc="best")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def run_similarity_islands_analysis(
    datasets: Optional[Sequence[str]] = None,
    strong_percentile: int = STRONG_PERCENTILE,
    weak_percentile: int = WEAK_PERCENTILE,
    seed_sample_count: int = 200,
    top_k: int = 5,
    random_state: int = 42,
    project_root: Optional[Path] = None,
) -> List[SimilarityStats]:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    if datasets is None:
        datasets = DEFAULT_DATASETS

    data_root = project_root / "data"
    graph_root = data_root
    results_dir = project_root / "results"

    all_stats: List[SimilarityStats] = []

    for dataset in datasets:
        dataset_key = dataset.lower()
        graph = load_copurchase_graph(dataset_key, graph_root)
        text_feat, visual_feat = load_features(dataset_key, data_root)

        text_stats = compute_similarity_stats(
            dataset_key,
            modality="text",
            feat=text_feat,
            graph=graph,
            strong_percentile=strong_percentile,
            weak_percentile=weak_percentile,
        )
        visual_stats = compute_similarity_stats(
            dataset_key,
            modality="visual",
            feat=visual_feat,
            graph=graph,
            strong_percentile=strong_percentile,
            weak_percentile=weak_percentile,
        )

        all_stats.extend([text_stats, visual_stats])

        tsne_path = save_tsne_plot(
            dataset=dataset_key,
            feat=text_feat,
            graph=graph,
            output_dir=results_dir,
            seed_sample_count=seed_sample_count,
            top_k=top_k,
            strong_percentile=strong_percentile,
            weak_percentile=weak_percentile,
            random_state=random_state,
        )
        print(f"Saved t-SNE plot: {tsne_path}")

    print_stats_table(all_stats)

    csv_path = results_dir / "similarity_island_stats.csv"
    latex_path = results_dir / "similarity_island_stats.txt"
    save_stats_csv(all_stats, csv_path)
    save_latex_table(all_stats, latex_path)
    print(f"Saved CSV stats: {csv_path}")
    print(f"Saved LaTeX stats: {latex_path}")

    return all_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Similarity Islands analysis on raw multimodal embeddings.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to analyze, e.g. baby sports clothing",
    )
    parser.add_argument("--seed_sample_count", type=int, default=200, help="Number of seed items for t-SNE")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k substitutes/complements per seed for t-SNE")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for sampling and t-SNE")
    parser.add_argument("--strong_percentile", type=int, default=STRONG_PERCENTILE, help="Strong split percentile")
    parser.add_argument("--weak_percentile", type=int, default=WEAK_PERCENTILE, help="Weak split percentile")
    parser.add_argument("--project_root", type=str, default=None, help="Project root containing data/, src/data/ and results/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve() if args.project_root else None
    run_similarity_islands_analysis(
        datasets=args.datasets,
        strong_percentile=args.strong_percentile,
        weak_percentile=args.weak_percentile,
        seed_sample_count=args.seed_sample_count,
        top_k=args.top_k,
        random_state=args.random_state,
        project_root=project_root,
    )


if __name__ == "__main__":
    main()


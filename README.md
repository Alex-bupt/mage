# MAGE

Code for **Can Complementary Signals Bridge Similarity Islands? Manifold-Augmented Graph Embedding for Multimodal Recommendation**.

MAGE combines offline text-feature augmentation with multimodal graph recommendation. The supplied implementation uses a soft-conceptor residual to augment text features, separate visual/text graph streams, complementary edges in the text stream, and a cross-modal gating module.

## Repository layout

| Path | Purpose |
| --- | --- |
| `stage1_process.py` | Offline text-feature augmentation |
| `main/src/models/mage.py` | MAGE graph model and losses |
| `main/src/main.py` | Training entry point |
| `main/src/configs/` | Dataset, model and explicit experiment configurations |
| `main/src/analysis/` | Similarity-island and hard-complement analysis utilities |
| `main/src/visualize_embeddings.py` | Checkpoint-based case-study visualization |
| `data_preprocess/` | Existing relation extraction and Qwen text-enhancement scripts |
| `tests/smoke_test.py` | Lightweight source, path and augmentation checks |

## Environment

The training code retains its legacy PyTorch 1.11 stack. Use a separate Python 3.8 environment; newer Python versions cannot install these historical pins directly. The original `python==3.7.11` entry was an interpreter requirement, not a pip package, and has been removed from `main/requirements.txt`. Missing imports such as torchvision, LMDB and plotting dependencies are now listed.

```bash
python -m pip install -r main/requirements.txt
```

Choose a PyTorch build compatible with your CUDA installation. Qwen endpoint clients have separate requirements in `data_preprocess/requirements.txt` and are not required for training on prepared features.

## Prepare data

See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md). The default root is `main/data/`, independent of the launch directory. A dataset directory contains aligned interaction splits, visual/text features and optional complementary edges:

```text
main/data/clothing/
  clothing.inter
  image_feat.npy
  text_feat_raw.npy
  text_feat.npy
  related_by_id.json
  p2b_map_filtered.json
```

The local source folder does not include the full visual/text feature matrices. Obtain the processed inputs separately and retain the exact ID mapping and train/validation/test split. Source upload does not reproduce the paper's result tables.

## Stage 1: augment text features

From the repository root:

```bash
python stage1_process.py --emb-path main/data/clothing/text_feat_raw.npy --related-path main/data/clothing/related_by_id.json --out-path main/data/clothing/text_feat.npy
```

The CLI processes all feature rows by default, preserving row IDs. `--seed-max` is an inclusive last row for an explicit prefix. Keep raw and augmented files separate. The supplied augmentation defaults are substitute percentile 95, complement percentile 45, aperture 12, gap retention 0.60 and an injection cap of 0.35. The analysis script separately defaults to percentiles 95/40; record these choices when comparing results.

## Stage 2: train

From the repository root, use the local Clothing configuration:

```bash
python main/src/main.py --model mage --dataset clothing --config main/src/configs/experiments/clothing_local.yaml
```

Or select the prior repository's Sports configuration:

```bash
python main/src/main.py --model mage --dataset sports --config main/src/configs/experiments/sports_legacy.yaml
```

| Setting | Local Clothing snapshot (default) | Prior repository Sports snapshot |
| --- | --- | --- |
| Seed | 13 | 999 |
| P2B edge weight | 0.2 | 0.7 |
| User-modality temperature | 0.2 | 0.1 |
| Visual/text alignment loss weight | 0.1 | 0.01 |

These are preserved experiment snapshots, not a claim that either configuration regenerates every paper table. Other datasets have loader configurations but need their own selected hyperparameters.

`--data-path /absolute/path/to/data` overrides the input root. Relative data paths resolve from `main/`. `--cpu` disables CUDA and `--epochs 1` shortens a diagnostic run. YAML passed through `--config` overrides bundled defaults; these explicit CLI options override the YAML. Training logs, checkpoints and top-k outputs remain relative to the launch directory.

The model caches feature KNN graphs (`*_adj_*.pt`) in each dataset directory. If feature matrices or their row ordering change, use a fresh data directory or remove the obsolete graph caches before training. A missing P2B map is accepted by the code but disables complementary-edge injection. Both visual and text features are required.

## Analysis and checks

```bash
python main/src/run_similarity_islands.py --datasets clothing --project_root main
python tests/smoke_test.py
```

Similarity analysis reads `text_feat_raw.npy`, `image_feat.npy` and `p2b_map_filtered.json` under `main/data/<dataset>/`; results go to `main/results/`. Use the raw text features when analyzing the pre-augmentation geometry. Case-study visualization additionally requires an explicit checkpoint and case JSON; use `python main/src/visualize_embeddings.py --help` in the full environment.

The smoke check needs only NumPy. It compiles all Python sources, exercises shared path resolution from different working directories, checks the augmentation residual on synthetic vectors, and checks the training and augmentation CLI help. It does not train a model or validate GPU execution. Full training and plotting require the dependencies and data above.

## Provenance

This update reconciles the supplied local research source with the existing MAGE repository. It retains the repository's preprocessing scripts and MAGE naming, synchronizes the local finite-value similarity guard, and makes the differing experiment settings explicit. Machine-specific training paths are replaced with a shared data-root resolver. Generated logs, bytecode, datasets, checkpoints and paper figures are excluded from the source update.

The implementation builds on **GUME** and the **MMRec** framework. The original source README, citation and academic-use notice are preserved in [docs/upstream/GUME_README.md](docs/upstream/GUME_README.md). That archived README describes upstream GUME, not MAGE; its original result image is not part of this release. No new license grant is introduced by this packaging update.

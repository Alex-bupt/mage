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

**[Download prepared datasets from Google Drive](https://drive.google.com/file/d/1EYtXqrTKnv2WqVvIsBezJ7qLKPnteg4f/view?usp=sharing)** — Baby, Sports, Clothing and Electronics.

The archive `MAGE_datasets_2026-10-08.zip` is approximately **1.88 GB** (1,882,496,634 bytes). It includes interaction splits and hard-test subsets, ID mappings, visual features, enhanced training text features, raw seed text features, expanded text catalogs, offline relation graphs and online complementary edges. The text features have already been augmented using the shared Stage-1 parameters below.

Extract the archive to obtain `MAGE_datasets/`, then point the training loader at that directory:

```bash
python main/src/main.py --model mage --dataset clothing --config main/src/configs/experiments/clothing_local.yaml --data-path /absolute/path/to/MAGE_datasets
```

Use the supplied `text_feat.npy` directly for training; rerun Stage 1 only when changing the augmentation settings. The included `README.txt` explains the package layout, and `SHA256SUMS.txt` provides per-file checksums. The release archive was read back and its contents verified locally before upload. Its SHA-256 is:

```text
8412c252d01efbcadc48101b3d3634191318e554d11219a93119b6253fb0651a
```

See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md). The default root is `main/data/`, independent of the launch directory. A dataset directory contains aligned interaction splits, visual/text features and optional complementary edges:

```text
main/data/clothing/
  clothing.inter
  image_feat.npy
  text_feat_full.npy
  text_feat_raw.npy
  text_feat.npy
  related_by_id.json
  p2b_map_filtered.json
```

Feature matrices are distributed separately from the source. Retain the exact ID mapping and train/validation/test split. Expanded text catalogs contain extra neighbor items for offline augmentation; they are not direct Stage-2 inputs. Source upload does not reproduce the paper's result tables.

## Stage 1: augment text features

From the repository root:

```bash
python stage1_process.py --emb-path main/data/clothing/text_feat_full.npy --related-path main/data/clothing/related_by_id.json --out-path main/data/clothing/text_feat.npy --seed-max 23032
```

For an expanded catalog, **set `--seed-max` to the last recommendation item ID**: Baby 7049, Sports 18356, Clothing 23032, Electronics 63000. The full matrix supplies neighbor embeddings, while only the verified seed prefix is enhanced and exported. Omitting this option processes the entire catalog, which is unsuitable for the supplied Stage-2 interaction files. Verify the prefix against `i_id_mapping_full.csv` and `i_id_mapping.csv` before running. Keep the full matrix as `text_feat_full.npy`, the unmodified seed rows as `text_feat_raw.npy`, and the enhanced seed rows as `text_feat.npy`.

The confirmed parameters are shared by all four datasets: substitute percentile 95, complement percentile 45, minimum weak similarity 0.20, aperture 12, gap retention 0.60, margin epsilon 1e-8 and a fixed injection cap of 0.35. The current code applies the minimum weak similarity as a hard floor; the inherited comment mentioning soft weighting is stale. The analysis script separately defaults to percentiles 95/40; record these choices when comparing results.

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
python tests/test_runtime_compat.py
python tests/test_feature_validation.py
```

Similarity analysis reads `text_feat_raw.npy`, `image_feat.npy` and `p2b_map_filtered.json` under `main/data/<dataset>/`; results go to `main/results/`. Use the raw text features when analyzing the pre-augmentation geometry. Case-study visualization additionally requires an explicit checkpoint and case JSON; use `python main/src/visualize_embeddings.py --help` in the full environment.

The smoke check needs only NumPy. It compiles Python sources, exercises shared path resolution from different working directories, checks augmentation residuals and CLI help. The runtime compatibility test additionally needs PyTorch and checks graph normalization and ranking metrics. Feature validation rejects expanded catalogs before allocating embeddings or quadratic KNN matrices.

Local integration validation completed one CPU epoch, validation and test for each of the four datasets using real interaction subsets (32 users each), supplied image features, newly augmented text features and restricted real P2B edges. This used Python 3.11, PyTorch 2.5.1+cpu, torchvision 0.20.1+cpu, NumPy 1.26.4, pandas 2.2.3 and SciPy 1.14.1, with reduced embedding size/depth. These are software checks, not full-dataset runs or paper metrics. Native PyTorch degree aggregation removes the extra `torch_scatter` dependency; metric code also supports modern NumPy without `np.float`.

## Provenance

This update reconciles the supplied local research source with the existing MAGE repository. It retains the repository's preprocessing scripts and MAGE naming, synchronizes the local finite-value similarity guard, and makes the differing experiment settings explicit. Machine-specific training paths are replaced with a shared data-root resolver. Generated logs, bytecode, datasets, checkpoints and paper figures are excluded from the source update.

The implementation builds on **GUME** and the **MMRec** framework. The original source README, citation and academic-use notice are preserved in [docs/upstream/GUME_README.md](docs/upstream/GUME_README.md). That archived README describes upstream GUME, not MAGE; its original result image is not part of this release. No new license grant is introduced by this packaging update.

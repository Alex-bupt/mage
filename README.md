# MAGE

Code for **Can Complementary Signals Bridge Similarity Islands? Manifold-Augmented Graph Embedding for Multimodal Recommendation**.

MAGE combines offline text-feature augmentation with multimodal graph recommendation: a soft-conceptor residual augments text features, complementary edges enter the text graph stream, and a cross-modal gate combines representations.

## Environment

The experiments used Python **3.12.3**, PyTorch **2.8.0+cu128**, torchvision **0.23.0+cu128**, CUDA **12.8**, and an NVIDIA RTX **3090**. In a fresh Python 3.12 environment:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r main/requirements-verified.txt
```

Choose a CUDA build compatible with your hardware. [requirements-verified.txt](main/requirements-verified.txt) records the directly used package versions; it is not a complete transitive lock. The older `main/requirements.txt` retains the prior PyTorch 1.11 environment and should not be applied over this environment. Qwen clients have separate requirements in `data_preprocess/requirements.txt` and are unnecessary for training on prepared features.

For the recorded thread settings, run the following in the shell that launches Stage 1 and Stage 2:

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONHASHSEED=13
```

`PYTHONHASHSEED` is separate from the per-dataset training seed. Fresh training can vary across hardware/software and GPU execution; replay results in the [experiment report](docs/VERIFIED_RESULTS.md) refer to the retained checkpoints.

## Prepare data

**[Download prepared datasets from Google Drive](https://drive.google.com/file/d/1EYtXqrTKnv2WqVvIsBezJ7qLKPnteg4f/view?usp=sharing)** — Baby, Sports, Clothing and Electronics.

`MAGE_datasets_2026-10-08.zip` contains interaction splits, ID mappings, visual features, raw seed text features, expanded text catalogs, offline relation graphs and complementary edges. Its original augmented `text_feat.npy` used gap retention **0.60**. **Run Stage 1 with the configurations below**, which use 0.20 or 0.40.

Archive size: 1,882,496,634 bytes. SHA-256:

```text
8412c252d01efbcadc48101b3d3634191318e554d11219a93119b6253fb0651a
```

Keep the original extraction unchanged. Prepare a fresh working copy with dataset folders at `main/data_verified/baby/`, `main/data_verified/sports/` and `main/data_verified/clothing/`. Each needs its `.inter`, `image_feat.npy`, `text_feat_full.npy`, `related_by_id.json` and `p2b_map_filtered.json`, with original mappings and splits preserved. Use a fresh directory without graph caches from different features. See [data format](docs/DATA_FORMAT.md).

## Stage 1: augment text features

Run from the repository root:

```bash
python stage1_process.py --emb-path main/data_verified/baby/text_feat_full.npy --related-path main/data_verified/baby/related_by_id.json --out-path main/data_verified/baby/text_feat.npy --seed-max 7049 --gap-keep-ratio 0.2
python stage1_process.py --emb-path main/data_verified/sports/text_feat_full.npy --related-path main/data_verified/sports/related_by_id.json --out-path main/data_verified/sports/text_feat.npy --seed-max 18356 --gap-keep-ratio 0.4
python stage1_process.py --emb-path main/data_verified/clothing/text_feat_full.npy --related-path main/data_verified/clothing/related_by_id.json --out-path main/data_verified/clothing/text_feat.npy --seed-max 23032 --gap-keep-ratio 0.4
```

`--seed-max` is the **inclusive last recommendation item ID**. The expanded matrix supplies neighbor embeddings; only the seed prefix is exported for Stage 2. Verify that prefix against `i_id_mapping_full.csv` and `i_id_mapping.csv`. Preserve `text_feat_full.npy` and `text_feat_raw.npy`; do not augment an already augmented matrix.

Remaining Stage-1 settings: substitute percentile 95, complement percentile 45, minimum weak similarity 0.20 (a hard floor), aperture 12, margin epsilon 1e-8 and injection cap 0.35. `gap_keep_ratio` in YAML records the preparation setting; Stage 1 receives it through `--gap-keep-ratio` and does not read the Stage-2 YAML automatically. Expected feature hashes are in the experiment report.

## Stage 2: train

Choose the preset for your dataset:

```bash
python main/src/main.py --model mage --dataset baby --config main/src/configs/experiments/baby_verified.yaml --data-path "$PWD/main/data_verified"
python main/src/main.py --model mage --dataset sports --config main/src/configs/experiments/sports_verified.yaml --data-path "$PWD/main/data_verified"
python main/src/main.py --model mage --dataset clothing --config main/src/configs/experiments/clothing_verified.yaml --data-path "$PWD/main/data_verified"
```

Omitting `--config` uses the dataset-specific defaults in `MAGE.yaml`. Explicit YAML values override those defaults, and explicit CLI options such as `--data-path`, `--epochs` and `--cpu` override YAML. Relative data roots resolve from `main/`; the examples use an absolute path to avoid ambiguity. Defaults allocate a fresh run directory under `saved/verified/`; if setting `run_dir` explicitly, use a new directory for each independent run.

| Parameter | Baby | Sports | Clothing |
| --- | ---: | ---: | ---: |
| `p2b_weight` | 0.9 | 0.4 | 0.2 |
| `max_p2b_k` | 10 | 20 | 15 |
| Stage-1 `gap_keep_ratio` | 0.2 | 0.4 | 0.4 |
| `um_loss` | 0.003 | 0.01 | 0.01 |
| `vt_loss` | 0.01 | 0.01 | 0.1 |
| `bm_loss` | 0.01 | 0.005 | 0.01 |
| `learning_rate` | 0.001 | 0.0007 | 0.001 |
| `bm_temp` | 0.2 | 0.2 | 0.2 |
| `um_temp` | 0.1 | 0.15 | 0.2 |

Shared training settings include 500 maximum epochs, validation Recall@20 for early stopping, patience 60, minimum 80 epochs, train batch size 8192 and evaluation batch size 4096. The full policy and fixed model parameters are in the report and YAML. `clothing_local.yaml` and `sports_legacy.yaml` are retained as legacy presets. See [experiment results](docs/VERIFIED_RESULTS.md) for full metrics and evaluation details.

The trainer writes raw paired metrics, sampled loss/gradient diagnostics, and snapshots of the saved epoch's weights. Validation-selected checkpoints remain separate from exploratory test/target selections. Check `run_result.json` and checkpoint metadata together; a running experiment can already have `run_result.json`.

KNN graphs (`*_adj_*.pt`) are cached in each dataset folder. Changing features, row order or Stage-1 settings requires a fresh cache directory. A missing P2B map disables complementary-edge injection and does not reproduce these configurations. Both visual and text features are required.

## Repository layout and checks

| Path | Purpose |
| --- | --- |
| `stage1_process.py` | Offline text-feature augmentation |
| `main/src/models/mage.py` | MAGE graph model and losses |
| `main/src/main.py` | Training entry point |
| `main/src/configs/` | Dataset defaults and explicit experiment presets |
| `main/src/analysis/` | Similarity-island and hard-complement analysis |
| `data_preprocess/` | Relation extraction and Qwen enhancement scripts |
| `docs/VERIFIED_RESULTS.md` | Experiment results and evaluation details |

```bash
python -B -m unittest discover -s tests -p 'test_*.py' -v
python -B tests/smoke_test.py
python main/src/run_similarity_islands.py --datasets clothing --project_root main
```

The smoke check exercises source compilation, paths and augmentation. Runtime checks require the training dependencies; feature validation rejects expanded catalogs as direct Stage-2 inputs. Additional regression checks cover dataset/override configuration, graph equivalence, ranking membership, validation selection and epoch-specific checkpoint snapshots. These software checks are separate from the full-data checkpoint replays reported above. Similarity analysis uses `text_feat_raw.npy`; its independent percentile defaults are 95/40, so record them when comparing with Stage 1's 95/45 settings.

The publication check passed all **29 CPU regression tests** and the smoke check (39 Python source files compiled, paths, tiny augmentation and CLI checks). No new full-data training was performed for this source publication.

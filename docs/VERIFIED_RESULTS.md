# MAGE experiment results

This report records the selected Baby, Sports and Clothing checkpoints. The criterion was **Recall@10 and Recall@20 both reaching the paper value at one checkpoint, one epoch and one split**. Each selected run uses one seed, and validation/test observations were used during selection. This is exploratory tuning, not a five-seed reproduction, an independent held-out estimate or a global optimum. Electronics was excluded.

## Selected results and paper comparison

Epoch numbers are one-based. Each row belongs to one saved checkpoint; different epochs' or splits' metric maxima are never combined.

| Dataset | Split | Epoch | Seed | R@10 | R@20 | N@10 | N@20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baby | valid | 97 | 42 | 0.07049112882489073 | 0.10819925553147461 | 0.03751816534688235 | 0.04706554069489575 |
| Sports | test | 264 | 999 | 0.08221159205484167 | 0.12219727432520287 | 0.04242058601996397 | 0.05273041803916519 |
| Clothing | valid | 190 | 13 | 0.07384272957431615 | 0.10890574333715473 | 0.03927774578546903 | 0.04809731282348563 |

| Dataset | Paper R@10 / R@20 | Achieved R@10 / R@20 relative to paper | Paper N@10 / N@20 |
| --- | --- | --- | --- |
| Baby | 0.0702 / 0.1077 | 100.41471343% / 100.46356131% | 0.0379 / 0.0475 |
| Sports | 0.0821 / 0.1221 | 100.13592211% / 100.07966775% | 0.0440 / 0.0544 |
| Clothing | 0.0738 / 0.1088 | 100.05789915% / 100.09719057% | 0.0396 / 0.0484 |

NDCG was not the stopping criterion and remains below the corresponding paper values. [The raw JSON](verified_results_20261027.json) contains **both validation and test metrics** at each selected epoch: Recall, NDCG, Precision and MAP at 5/10/20/50, without rounding. The complete runs finished 158, 293 and 247 epochs respectively; the selected epochs above are not their training duration.

## Configuration

Use `main/src/configs/model/MAGE.yaml` for automatic dataset selection or `main/src/configs/experiments/{dataset}_verified.yaml` for an explicit preset. The [README](../README.md#stage-1-augment-text-features) gives complete Stage-1 and training commands.

| Parameter | Baby | Sports | Clothing |
| --- | ---: | ---: | ---: |
| Stage-1 gap retention | 0.2 | 0.4 | 0.4 |
| Stage-1 inclusive seed maximum | 7049 | 18356 | 23032 |
| Training seed | 42 | 999 | 13 |
| P2B weight | 0.9 | 0.4 | 0.2 |
| Maximum P2B neighbors | 10 | 20 | 15 |
| UM loss weight | 0.003 | 0.01 | 0.01 |
| VT loss weight | 0.01 | 0.01 | 0.1 |
| BM loss weight | 0.01 | 0.005 | 0.01 |
| Learning rate | 0.001 | 0.0007 | 0.001 |
| BM temperature | 0.2 | 0.2 | 0.2 |
| UM temperature | 0.1 | 0.15 | 0.2 |

Shared model parameters: embedding size 64, feature KNN K=10, UI propagation depth 3, item propagation depth 1, `reg_weight_1=1.0e-5`, `reg_weight_2=0.1`. Optimizer: Adam with weight decay 0 and scheduler `[0.96, 50]`. Training uses one sampled negative, batch size 8192, evaluation batch size 4096, evaluation every epoch, 500 maximum epochs, patience 60 on validation Recall@20 and minimum 80 epochs. `grad_sample_batches=1` records a sampled batch, not an epoch-wide gradient average.

The selected Sports run uses default graph modes `visual_user_graph: shared_text` and `explicit_visual_graph: raw_knn`; it retains the BM constraint with weight 0.005. Optional graph alternatives and BM removal were exploratory comparisons, not part of this selected model. The final improvement followed experiments on learning rate and BM weight; it does not establish a new architecture or a universal causal benefit of these settings.

The Baby run originally used 99%-of-paper checkpoint-retention thresholds. Its retained epoch already exceeds the later 100% criterion, as checked using unrounded metrics. Public presets use the full paper Recall targets for new runs. These thresholds affect checkpoint selection, not the optimization loss or validation early stopping. Other packaging differences are portable output paths, one-element seed lists for the search interface, Stage-1 gap metadata and explicit default graph flags for Baby/Clothing.

## Stage-1 data provenance

The dataset archive's original text features used gap retention 0.60. Generate the configured text features from `text_feat_full.npy` and `related_by_id.json`; do not enhance `text_feat.npy` again. Use the inclusive seed maximum to export only the recommendation-item prefix. Preserve the original interactions, item ordering, image features and P2B edges.

Expected generated `text_feat.npy` SHA-256:

| Dataset | SHA-256 |
| --- | --- |
| Baby | `bbf454180d5de6d69b32811c34f930d1cc3b1945b66ba6c69d2256f0696a20e4` |
| Sports | `040e5f9d921a6e6bfcc78bc907be5d3fd4295decf93800289d6d65c0502747b9` |
| Clothing | `030bb829b9ef044b961165397adf255e9609fbbaa75fba908821f6f200a9743d` |

All used Stage-1 source SHA-256 `ea21500b024e85eafc83d119ee68cdede09d61fc849aeb4986e2ade88f21713f`. Input hashes for interactions, mappings, full/raw text matrices, image features and relation files are in the JSON. Feature hashes describe the files actually used for these checkpoints; compare regenerated features before expecting an exact replay.

## Verification scope

Training workers completed successfully. Retained checkpoint bytes were hashed, then each checkpoint was strictly loaded and evaluated on validation and test data. **Every recorded metric reproduced exactly: maximum absolute difference 0.0.** The exported copies were independently loaded and evaluated again with the same result. The exported ZIP and all 114 manifest-listed files were checked locally for size and SHA-256 consistency.

This verifies the saved models against their original training snapshots. It is not a fresh five-seed training campaign. The repository consolidates the runtime changes and supplies portable YAML defaults; fresh training can vary with hardware, libraries and GPU execution.

| Dataset | Selected checkpoint SHA-256 |
| --- | --- |
| Baby | `164f721fcf7a3ebf79bafd63b323feea9beece69fa6bc2d10f4430e6df07cbbf` |
| Sports | `c7ecd629444a665f3528ea7465f42a6f01cd4204d4151eff267de7dd7d8054a0` |
| Clothing | `eb8180b29c83fd23da74c66cc8396f8ac3f44131dfaf3671c6fe8a2c981742ff` |

The separately retained export contains models, paired metrics, full configurations, original source snapshots and `verify_results.py`. Checkpoints, data and the export archive are **not uploaded in this source commit**, and no public URL for that export is supplied here. Hashes identify retained artifacts; they do not provide model access.

If you already have that export and its prepared features, run from the export root:

```bash
python verify_results.py --dataset baby --data-root /absolute/path/to/prepared_data --output-dir re_evaluation/baby
python verify_results.py --dataset sports --data-root /absolute/path/to/prepared_data --output-dir re_evaluation/sports
python verify_results.py --dataset clothing --data-root /absolute/path/to/prepared_data --output-dir re_evaluation/clothing
```

Use fresh output directories and a writable data copy with exact features and compatible graph caches. Model initialization writes `inter.json`. The main training entry point is not a checkpoint-replay command.

## Runtime and environment

The runtime supports configurable P2B degree and gap retention, sparse graph assembly, efficient ranking-membership evaluation, finite loss/gradient diagnostics, and snapshots matching the recorded epoch. `metrics.jsonl` records raw paired metrics. `best_valid.pth` and the global best checkpoint use validation selection. `best_test_exploratory.pth`, `best_joint_exploratory.pth` and `target_match_exploratory.pth` are explicitly exploratory. Early stopping remains based on validation Recall@20.

The recorded environment was Linux, Python 3.12.3, PyTorch 2.8.0+cu128, CUDA 12.8, cuDNN 91002, RTX 3090; NumPy 1.26.4, SciPy 1.14.1, pandas 2.2.3, PyYAML 6.0.2, LMDB 1.5.1, matplotlib 3.9.2 and scikit-learn 1.5.2. Publication-time inspection of that environment also confirmed torchvision 0.23.0+cu128, Pillow 11.3.0 and tqdm 4.67.1. See [requirements-verified.txt](../main/requirements-verified.txt). Thread settings were OMP=4, MKL=4 and OPENBLAS=1; `PYTHONHASHSEED=13` is separate from the training seed.

The JSON records original source-manifest identities and proof hashes. Those identities refer to the archived training snapshots, not to an assertion that this reorganized repository has the same aggregate hash.

## Source-publication checks

The integrated runtime/configuration snapshot passed 29 CPU regression tests and the smoke check, including compilation of 39 Python source files, paths, tiny augmentation and CLI checks:

```bash
python -B -m unittest discover -s tests -p 'test_*.py' -v
python -B tests/smoke_test.py
```

These checks cover configuration selection and override precedence, graph and ranking equivalence, actual checkpoint snapshots and validation selection. They do not constitute another full-data training run or another GPU replay; the checkpoint replay evidence is described separately above.

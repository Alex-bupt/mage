# Input contract

Training resolves the same dataset directory for interactions, feature matrices and graph caches. With `data_path: data`, this is `main/data/<dataset>/`. Absolute paths are also accepted through `--data-path`.

## Interactions and features

- `<dataset>.inter` is a tab-separated file with `userID`, `itemID`, and `x_label` columns. Split labels are 0 for training, 1 for validation, and 2 for test. User/item IDs are zero-based integers aligned with the supplied mappings. The loader filters validation/test users absent from training when `filter_out_cod_start_users` is enabled.
- `image_feat.npy` is a finite numeric matrix with shape `(number_of_items, visual_dimension)`.
- `text_feat.npy` is the augmented text matrix used for training, with the same item rows as the visual features. Keep the unmodified recommendation-item prefix as `text_feat_raw.npy` for similarity analysis.
- `text_feat_full.npy` is the expanded text catalog used as Stage-1 input. Extra rows represent neighbor items outside the recommendation catalog. `i_id_mapping_full.csv` maps these rows to ASINs. Its first N rows must match the N-row recommendation mapping `i_id_mapping.csv` exactly. Stage 1 reads this full matrix but exports only the seed prefix via `--seed-max N-1`.
- `related_by_id.json` maps string item IDs to lists of numeric/string neighbor IDs. Stage 1 uses this to form local substitute/complement sets.
- `p2b_map_filtered.json` maps string item IDs to lists of neighbor IDs aligned to the training item rows. The MAGE text graph uses these optional complementary edges. Omitting this file changes the model to run without that edge source.

Data and feature files are external to this source repository. Raw benchmark preparation and item-feature generation remain prerequisites if using different data. The existing `data_preprocess/` scripts operate on catalog metadata and mappings; inspect each script's arguments or configuration before use. Qwen scripts expect an OpenAI-compatible model endpoint.

For the inspected supplemental datasets, the seed/full text row counts are Baby 7050/71317, Sports 18357/77832, Clothing 23033/84695 and Electronics 63001/498196. Text embeddings have 512 columns and image features 4096 columns. The supplied Electronics image matrix contains 480 zero rows; the similarity helper handles zero norms, but these rows still represent missing visual signal.

## Generated state

The model writes KNN adjacency caches and an intersection graph to the dataset directory. Input directories must therefore be writable. Cached graphs depend on feature values, ID ordering and KNN settings; they are not invalidated automatically when features change.

Keep raw and augmented matrices distinct and use identical item mappings across all files. Build features, profiles and relations according to the intended experimental split. A source release or successful loader run alone does not establish evaluation equivalence to the paper.

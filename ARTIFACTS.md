# Artifacts outside git

Weights, features, checkpoints and caches are gitignored (size). Everything
needed to re-run the **final evaluation** beyond git is in
https://drive.google.com/file/d/1yZiy94oUIwK-NsQVRBLqbPa_96GD6rKN/view?usp=sharing


Git holds the rest of the final test's inputs byte-exact: the registration
and its amendment, the frozen views and completion marker, the router, the
frozen adaptive routers, both latency profiles, and the split manifest. Their
registered hashes reproduce from a fresh clone (checked 2026-10-03).

Note on history: commit `0a4d83a` holds the final collection's views and
completion marker, committed before the results commit `53fbbb7`. The
features, policy and cache it refers to are not in git; the marker binds the
features by hash, and the evaluator verified it.

| path | size | role | retrieve from |
|---|---|---|---|---|
| `artifacts/final/feats_final.npz` | 30.9 MB | final objects' ResNet features + poses (evaluation input; never a training input) | archive; output of the Kaggle notebook version that ran the final collection (2026-10-03) |
| `cache/utility_cache.json` | 29.1 MB | utility cache, 121,376 view sets incl. every final-test reconstruction (evaluation input) | archive; same notebook version (its cache plus the local one, merged with merge_caches) |
| `artifacts/tier1/set_pose_s164864.pt` | 9.5 MB  selected view policy (pre-registered sha256) | archive; Kaggle input dataset of the final collection; Phase 1 notebook output |
| `artifacts/router/feats.npz` | 652.8 MB  training/dev ResNet features for 6,594 objects (router training input) | output of the Kaggle notebook version that ran Phase 1b (2026-10-02) |
| `artifacts/tier1/set_pose_last.pt` | 28.7 MB  Phase 1 full training state (resume) | Phase 1 notebook output |
| `artifacts/tier1/set_pose_s62464.pt` | 9.5 MB  Phase 1 kept checkpoint | Phase 1 notebook output |
| `artifacts/tier1/set_pose_s123904.pt` | 9.5 MB  Phase 1 kept checkpoint | Phase 1 notebook output |
| `artifacts/tier1/set_pose.pt` | 9.5 MB  Phase 1 final checkpoint (s200704) | Phase 1 notebook output |
| `artifacts/tier0b/set_pose.pt` | 9.5 MB | Tier 0B policy (official-train) | Kaggle dataset rl-tier0b-outputs; also in dataset tier1input |
| `artifacts/tier0b/set_index.pt` | 9.3 MB | Tier 0B index-head ablation | Kaggle dataset rl-tier0b-outputs |
| `artifacts/tier0b/mean_pose.pt` | 3.2 MB | Tier 0B mean-pool ablation | Kaggle dataset rl-tier0b-outputs |
| `cache/utility_cache_legacy.json` | 8.7 MB | original Tier 0B cache, pre-conversion | local only (the Tier 0B cache before conversion); also inside rl-tier0b-outputs |

Backbone weights: `build/kaggle_weights.zip` / Kaggle dataset `weights1`;
their content hashes are recorded in the pre-registration
(`artifacts.backbone_checkpoints`).

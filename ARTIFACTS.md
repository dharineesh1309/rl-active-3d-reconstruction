## Artifacts outside Git

Weights, features, checkpoints, and caches are gitignored due to their size.

Everything needed to re-run the **final evaluation** beyond Git is available here:

[Final evaluation artifacts — Google Drive](https://drive.google.com/file/d/1yZiy94oUIwK-NsQVRBLqbPa_96GD6rKN/view?usp=sharing&utm_source=chatgpt.com)

The archive contains the artifacts required for the final evaluation.

The Git repository contains the remaining final-test inputs byte-exact, including the registration and its amendment, the frozen views and completion marker, the router, the frozen adaptive routers, both latency profiles, and the split manifest. Their registered hashes reproduce from a fresh clone (checked 2026-10-03).

### History Note

Commit `0a4d83a` contains the final collection's views and completion marker and was committed before the results commit `53fbbb7`.

The features, policy, and cache it refers to are not stored in Git. The completion marker binds the features by hash, and the evaluator verifies this binding.

### Artifact Inventory

| Path | Size | Role | Retrieve from |
|---|---:|---|---|
| `artifacts/final/feats_final.npz` | 30.9 MB | Final objects' ResNet features and poses; evaluation input, never used for training | Archive; output of the final-collection Kaggle notebook (2026-10-03) |
| `cache/utility_cache.json` | 29.1 MB | Utility cache containing 121,376 view sets, including every final-test reconstruction | Archive; same notebook version |
| `artifacts/tier1/set_pose_s164864.pt` | 9.5 MB | Selected view policy | Archive; final-collection Kaggle dataset; Phase 1 notebook output |
| `artifacts/router/feats.npz` | 652.8 MB | Training/dev ResNet features for 6,594 objects | Phase 1b Kaggle notebook output (2026-10-02) |
| `artifacts/tier1/set_pose_last.pt` | 28.7 MB | Phase 1 full training state for resuming | Phase 1 notebook output |
| `artifacts/tier1/set_pose_s62464.pt` | 9.5 MB | Phase 1 checkpoint | Phase 1 notebook output |
| `artifacts/tier1/set_pose_s123904.pt` | 9.5 MB | Phase 1 checkpoint | Phase 1 notebook output |
| `artifacts/tier1/set_pose.pt` | 9.5 MB | Phase 1 final checkpoint (s200704) | Phase 1 notebook output |
| `artifacts/tier0b/set_pose.pt` | 9.5 MB | Tier 0B policy trained on official-train objects | Kaggle dataset `rl-tier0b-outputs`; also in `tier1input` |
| `artifacts/tier0b/set_index.pt` | 9.3 MB | Tier 0B index-head ablation | Kaggle dataset `rl-tier0b-outputs` |
| `artifacts/tier0b/mean_pose.pt` | 3.2 MB | Tier 0B mean-pooling ablation | Kaggle dataset `rl-tier0b-outputs` |
| `cache/utility_cache_legacy.json` | 8.7 MB | Original Tier 0B cache before conversion | Local copy; also in `rl-tier0b-outputs` |

### Backbone Weights

Backbone weights are stored separately:

- **Archive:** `build/kaggle_weights.zip`
- **Kaggle dataset:** `weights1`

Their content hashes are recorded in the pre-registration under `artifacts.backbone_checkpoints`.

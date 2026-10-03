# Artifacts outside git

Weights, features, checkpoints and caches are gitignored (size). Everything
needed to re-run the **final evaluation** beyond git is in
`build/final_artifacts_archive.zip` (34.3 MB, sha256
`16067690004c7ac732e96e7d60d64a6d60076b9a180b5796dba4d4aae68ef347`), which
carries its own `SHA256SUMS`. Keep a copy off this machine (for example as a
private Kaggle dataset).

Git holds the rest of the final test's inputs byte-exact: the registration
and its amendment, the frozen views and completion marker, the router, the
frozen adaptive routers, both latency profiles, and the split manifest. Their
registered hashes reproduce from a fresh clone (checked 2026-10-03).

Note on history: commit `e98244a` holds the final collection's views and
completion marker, committed before the results commit `053e390`. The
features, policy and cache it refers to are not in git; the marker binds the
features by hash, and the evaluator verified it.

| path | size | sha256 | role | retrieve from |
|---|---|---|---|---|
| `artifacts/final/feats_final.npz` | 30.9 MB | `4c69cc820fe3b077e0e4ea5569bcaec1765d46c871b2fe3745dcff367ff897a2` | final objects' ResNet features + poses (evaluation input; never a training input) | archive; Kaggle notebook <user>/notebook07814c3f19, version that ran the final collection (2026-10-03) |
| `cache/utility_cache.json` | 29.1 MB | `d5953f7e8567852d99c9a6363232735ce6b49f81101a941683b8ff3b5aebaab8` | utility cache, 121,376 view sets incl. every final-test reconstruction (evaluation input) | archive; same notebook version (its cache plus the local one, merged with merge_caches) |
| `artifacts/tier1/set_pose_s164864.pt` | 9.5 MB | `84d66f32d8f4878e3b198607773c087c26c7ec054c3a6b59eb66980e2b6937f2` | selected view policy (pre-registered sha256) | archive; Kaggle dataset <user>/inputsfinal; Phase 1 notebook <user>/rlproj output |
| `artifacts/router/feats.npz` | 652.8 MB | `dfbaa2a15eb33e2417aca4d5b7946b412fcb06e3621ed23481fee9eba37f6c46` | training/dev ResNet features for 6,594 objects (router training input) | Kaggle notebook <user>/notebook07814c3f19, version that ran Phase 1b (2026-10-02) |
| `artifacts/tier1/set_pose_last.pt` | 28.7 MB | `47e24b6f8346e04c9f7f7c273dbdb47148f9bf5bce113d68d179b8b63ccaceca` | Phase 1 full training state (resume) | Phase 1 notebook <user>/rlproj output |
| `artifacts/tier1/set_pose_s62464.pt` | 9.5 MB | `7e7bf8db9b1f992ffb874cbf58c65784c3173ca1d85a67b4081c2997f38487fa` | Phase 1 kept checkpoint | Phase 1 notebook <user>/rlproj output |
| `artifacts/tier1/set_pose_s123904.pt` | 9.5 MB | `9aedcb220da7e85820159847822d091d7eecc7c9c902df51438d135c42865726` | Phase 1 kept checkpoint | Phase 1 notebook <user>/rlproj output |
| `artifacts/tier1/set_pose.pt` | 9.5 MB | `e28607ba74cc30441b0854d2859b0abc5d47a1882e534509a287951811136470` | Phase 1 final checkpoint (s200704) | Phase 1 notebook <user>/rlproj output |
| `artifacts/tier0b/set_pose.pt` | 9.5 MB | `18778868a95ccca8f1a9bd794fe237ba995ca3e209e142d396f6f36cbc880366` | Tier 0B policy (official-train) | Kaggle dataset rl-tier0b-outputs; also in <user>/inputsfinal-era tier1 inputs |
| `artifacts/tier0b/set_index.pt` | 9.3 MB | `1a04a2eb61683b887d2fe72bed0f6cbc4897823fdf6aa5016aa01fbaebf08ebd` | Tier 0B index-head ablation | Kaggle dataset rl-tier0b-outputs |
| `artifacts/tier0b/mean_pose.pt` | 3.2 MB | `1e9fea83d0d462578535f273b8cc0206496444190ca2f5f4e141133bd3738c55` | Tier 0B mean-pool ablation | Kaggle dataset rl-tier0b-outputs |
| `cache/utility_cache_legacy.json` | 8.7 MB | `93fbf8ba3bb72887d73fa43dea3c14f790b97fedee9936cfb222f3208fb49aa3` | original Tier 0B cache, pre-conversion | local only (the Tier 0B cache before conversion); also inside rl-tier0b-outputs |

Backbone weights: `build/kaggle_weights.zip` / Kaggle dataset `weights1`;
their content hashes are recorded in the pre-registration
(`artifacts.backbone_checkpoints`).

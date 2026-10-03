# Running the GPU jobs on Kaggle

Every GPU step of this project ran as a Kaggle notebook with one code cell. The
cells live in `build/` and are complete as written; nothing in them refers to a
dataset by name -- each input is found by its content, so name your uploads
however you like.

## Inputs to upload as Kaggle datasets

| upload | built by | contents |
|---|---|---|
| code | `python build/package_for_kaggle.py` -> `build/kaggle_code.zip` | `rl_pipeline/` with a `BUILD.txt` stamp (the git commit) |
| weights | same script -> `build/kaggle_weights.zip` | `Pix2Vox-F-ShapeNet.pth`, `UMIFormer-ShapeNet.pth`, `UMIFormerPlus-ShapeNet.pth` |
| voxels | same script -> `build/kaggle_voxels.zip` | `ShapeNetVox32/` |
| renderings | public Kaggle dataset "ShapeNet-Pix2Vox" (12.9 GB) | `ShapeNetRendering/` (24 views per object) |
| per-job inputs | the zips named below | checkpoints, cache, frozen view lists |

Each cell starts with `EXPECT_BUILD`: the code dataset must carry that exact
`BUILD.txt`, so a notebook can never run code other than the version it was
written for. Turn on **GPU** and **Internet** (ResNet-50 weights download on
first use) and run with **Save Version -> Save & Run All**.

## The jobs, in order

| job | cell | extra inputs | produces |
|---|---|---|---|
| Phase 1: train the view policy on `controller_train` (~9 h) | `build/kaggle_phase1_cell.py` | `kaggle_tier1_inputs.zip` | `artifacts/tier1/*` checkpoints, utility cache |
| Phase 1b: paired dev comparison + router features | `build/kaggle_phase1b_cell.py` | Phase 1 notebook output; `kaggle_tier1_inputs.zip` | `eval_dev.json`, `feats.npz`, cache |
| Phase 3: heuristic view sets + GPU latency | `build/kaggle_phase3_cell.py` | `kaggle_phase3_inputs.zip` | cache, `latency_gpu.json`, 26-object benchmark cohort |
| Final test: collection only | `build/kaggle_final_cell.py` | `kaggle_final_inputs.zip` | `views_final.json`, `feats_final.npz`, `collection_complete.json`, cache |

Interrupted runs: re-run the same cell with the interrupted run's output
attached. Caches are merged by key (`merge_caches`), never replaced, and every
reconstruction already done is reused.

## Getting outputs back

```
kaggle kernels output <user>/<notebook> -p <dest> --file-pattern ".*(artifacts|cache)/.*"
```

Set `PYTHONUTF8=1` on Windows, or the notebook log fails to save.

Everything after collection -- router training (`train_router.py`), the dev
evaluation and the final evaluation (`phase3_eval.py`), the figure -- runs on a
CPU. See the top-level `README.md` and `BENCHMARK.md`.

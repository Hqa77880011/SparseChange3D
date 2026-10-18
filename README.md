# SparseChange3D

Implementation of **SparseChange3D: Observability-Aware Signed 3D Change Fields from Sparse Revisit Images**.

## Installation

Requires Python `>=3.11`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, matplotlib 3.10.0, pillow 12.3.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

Obtain ChangeSim or 3RScan/RIO under the relevant dataset license. Keep all rescans of an environment in one development/evaluation partition. Jointly align the two RGB visits using calibrated poses or COLMAP structure from motion. The field receives only RGB images, intrinsics, camera-to-world poses and a scene bounding box. Depth and meshes are reserved for evaluator ground truth.

Create a scene JSON:

```json
{"bounds": [[-2,-2,-1],[2,2,3]], "cameras": [{"path": "visit0/image.png", "visit": 0, "K": [[500,0,320],[0,500,240],[0,0,1]], "T": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}]}
```

Include cameras from both visits. `T` maps camera coordinates to the common world frame; cameras look along positive camera Z. World coordinates, mesh coordinates, translations, depth and voxel size must share units. Derive bounds from the aligned sparse reconstruction or a fixed registered region. Each matrix is supplied explicitly, avoiding assumptions about pose-file naming or row conventions.

```bash
sparsechange3d prepare --scene data/raw-scene.json --output data/scene --views 5 --maximum-side 960 --seed 7
```

Preparation samples at most five views per visit, resizes RGB images and adjusts intrinsics. `--views 0` keeps all listed images. Preserve manifests for reproducible view-count comparisons. Translation perturbation uses `--translation-noise` in world units; `--rotation-noise` is a Gaussian rotation-vector scale in degrees.

Optional evaluator truth is created from aligned reference/revisit meshes and a depth manifest; these inputs stay outside field fitting:

```bash
python prepare_labels.py --reference-mesh data/reference.ply --revisit-mesh data/revisit.ply --depth-manifest data/depths.json --scene data/raw-scene.json --output data/truth.npz --voxel 0.02
```

## Training and inference

```bash
sparsechange3d fit --data data/scene --output outputs/full --iterations 30000 --warmup 5000 --rays 1024 --samples 64 --seed 7
```

## Evaluation

```bash
sparsechange3d evaluate --data data/scene --model outputs/full/model.pt --output outputs/field --voxel 0.02 --eta 0.20 --probability-threshold 0.25 --change-threshold 0.15
sparsechange3d evaluate --data data/scene --model outputs/full/model.pt --truth data/truth.npz --output outputs/evaluation
```

## Comparisons and ablations

```bash
sparsechange3d fit --data data/scene --output outputs/independent --variant independent
sparsechange3d fit --data data/scene --output outputs/no-sparsity --variant no_sparsity
sparsechange3d fit --data data/scene --output outputs/no-appearance --variant no_appearance
sparsechange3d fit --data data/scene --output outputs/union --variant union_support
sparsechange3d fit --data data/scene --output outputs/no-observability --variant no_observability
sparsechange3d fit --data data/scene --output outputs/no-pose --variant no_pose
```

## Outputs

Fitting writes `model.pt`, `history.json` and `optimization.png`. Evaluation exports `field.npz`, signed-change visualization and `metrics.json`; labeled metrics require `--truth`. Retain the prepared scene manifest because pose parameters index its cameras.

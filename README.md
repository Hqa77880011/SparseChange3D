# SparseChange3D

Implementation of **SparseChange3D: Observability-Aware Signed 3D Change Fields from Sparse Revisit Images**.

The scene field factorizes occupancy logits as `logit_0 = canonical - change` and `logit_1 = canonical + change`. Sparse signed residuals identify additions/removals, a restricted visit appearance branch explains moderate lighting changes, and two-visit visibility and angular diversity determine whether a voxel can receive a directional label.

## Installation

Use Python 3.11+ in an activated virtual environment:

```bash
python -m pip install .
```

The multiresolution hash encoder is implemented directly in PyTorch. Use `--device cuda` after installing the appropriate PyTorch build for your GPU. CPU execution is supported and can be slow for large scene volumes.

## Prepare sparse revisits

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

## Fit the two-visit field

```bash
sparsechange3d fit --data data/scene --output outputs/full --iterations 30000 --warmup 5000 --rays 1024 --samples 64
```

Field fitting is scene-specific photometric optimization, so it uses the sparse RGB input from both visits without accessing change labels. The canonical field is fitted first, then the residual is enabled after warm-up, and pose increments are enabled after two warm-up periods. A small SE(3) matrix exponential composes local increments with each initial camera pose.

The objective combines ray RGB L1 and 4×4 patch SSIM, observability-weighted residual L1, local residual total variation, temporal appearance amplitude/code penalties and pose regularization. Observability weights are detached before the sparsity penalty. The two visit codes are opposite vectors, giving a zero-mean pair by construction.

| Option | Default | Meaning |
| --- | --- | --- |
| `--rays`, `--samples` | 1024, 64 | Rays and ordered samples per optimization step |
| `--observable-samples` | 32 | Sampled ray points receiving multi-camera observability evaluation |
| `--lr`, `--pose-lr` | 0.001, 0.0001 | Initial field/pose learning rates with cosine decay |
| `--sparsity`, `--tv` | 0.0002, 0.00005 | Signed residual penalties |
| `--appearance`, `--pose-weight` | 0.001, 0.01 | Appearance and pose regularization |

`model.pt`, `history.json`, and `optimization.png` contain fitted parameters and actual objective history. The input manifest must be retained alongside the checkpoint because pose increments index its cameras.

## Export and evaluate signed change

```bash
sparsechange3d evaluate --data data/scene --model outputs/full/model.pt --output outputs/field --voxel 0.02 --eta 0.20 --probability-threshold 0.25 --change-threshold 0.15
```

The method computes pre-point ray transmittance, multi-view reachability, angular diversity and the minimum support across visits. It does not confuse visible free space with an unobserved voxel. Labels are `0 stable`, `1 added`, `2 removed`, `3 unknown`. An addition/removal requires both the occupancy-probability difference and signed residual to pass their respective thresholds. Low joint support yields unknown.

`field.npz` contains query coordinates, four-state predictions, signed residuals, observability, and directional confidence. `signed_changes.png` plots added/removed voxels. `metrics.json` records thresholds, parameter count and elapsed export time.

For quantitative scoring provide an evaluator NPZ with `xyz[N,3]`, `label[N]` in `{0,1,2}`, and optional joint full-stream visibility `valid[N]`:

```bash
sparsechange3d evaluate --data data/scene --model outputs/full/model.pt --truth data/truth.npz --output outputs/evaluation
```

The evaluator reports added/removed IoU and F1, signed macro-F1, stable false-change percentage, unknown fraction and directional risk–coverage. Unknown outputs count as missed changed voxels in IoU/F1. Selective coverage divides retained directional predictions by all evaluable ground-truth changed voxels; it can remain below one when observability is insufficient. Stable false alarms remain a separate metric.

An optional `object_size[N]` records controlled cube-edge lengths, with stable voxels set to zero. The evaluator reports signed performance by size and the smallest size for which both directional F1 scores reach 0.70. Run separate view-count and common-overlap manifests to form the detectability sweep.

## Construct evaluator labels

For registered meshes and complete-stream depth:

```bash
python prepare_labels.py --reference-mesh data/reference.ply --revisit-mesh data/revisit.ply --depth-manifest data/depths.json --scene data/raw-scene.json --output data/truth.npz --voxel 0.02
```

`depths.json` is a list of `path,visit,K,T` objects using the full RGB-D streams. Depth arrays contain camera-Z depth in world units; convert range/depth units before export. The script voxelizes mesh surfaces in the shared scene grid, labels occupied-only-in-revisit voxels as added and occupied-only-in-reference voxels as removed, and keeps only voxels reached by a valid full-stream depth ray in both visits. This evaluator archive is loaded only by `evaluate` and is never passed to `fit`.

## Baselines and ablations

```bash
sparsechange3d fit --data data/scene --output outputs/independent --variant independent
sparsechange3d fit --data data/scene --output outputs/no-sparsity --variant no_sparsity
sparsechange3d fit --data data/scene --output outputs/no-appearance --variant no_appearance
sparsechange3d fit --data data/scene --output outputs/union --variant union_support
sparsechange3d fit --data data/scene --output outputs/no-observability --variant no_observability
sparsechange3d fit --data data/scene --output outputs/no-pose --variant no_pose
```

Independent geometry uses separate hash tables and occupancy heads for the visits, then derives a signed difference for the same decoder. It retains the common color/appearance machinery and jointly scheduled optimizer. Other variants remove residual sparsity, visit appearance, the conservative minimum across visits, observability rejection, or pose refinement. Evaluate all checkpoints with the same query grid, truth masks, thresholds and camera budget. Fit each stochastic stage with the same three seeds before reporting mean/standard deviation.

```bash
python compare.py --runs outputs/full-evaluation/metrics.json outputs/independent-evaluation/metrics.json --output outputs/comparison
```

The script plots computed directional, stable-region and detectability metrics from the supplied evaluations. View-count/overlap sweeps can use the same interface with one evaluation directory per registered condition.

## Implementation choices

The hash representation uses 16 levels, two features per level and 32,768 entries per level, with 64-unit canonical and 32-unit residual heads. Rendering uses uniform midpoint ray samples rather than hierarchical importance sampling. Appearance corrections are bounded to ±0.1 RGB and use an eight-dimensional zero-mean code. Observability uses 24 transmittance samples per camera/query and a soft image-edge mask with exact out-of-frustum rejection. The smaller default ray batch and pure PyTorch encoder affect speed; increase the ray batch to 4,096 to follow the paper's optimization budget when memory permits. Thresholds must be selected once on development scenes and retained for test and cross-dataset evaluation.

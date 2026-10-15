import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--reference-mesh", required=True); parser.add_argument("--revisit-mesh", required=True); parser.add_argument("--depth-manifest", required=True); parser.add_argument("--scene", required=True); parser.add_argument("--output", required=True); parser.add_argument("--voxel", type=float, default=.02); args = parser.parse_args()
    scene = json.loads(Path(args.scene).read_text()); bounds = np.array(scene["bounds"]); dimensions = np.ceil((bounds[1] - bounds[0]) / args.voxel).astype(int)
    axes = [a + (np.arange(n) + .5) * args.voxel for a, n in zip(bounds[0], dimensions)]; points = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    occupancy = []
    for path in [args.reference_mesh, args.revisit_mesh]:
        mesh = trimesh.load(path, force="mesh", process=False); surface = mesh.voxelized(args.voxel).points
        index = np.floor((surface - bounds[0]) / args.voxel).astype(int); valid = ((index >= 0) & (index < dimensions)).all(1); occupied = np.zeros(tuple(dimensions), dtype=bool)
        occupied[tuple(index[valid].T)] = True; occupancy.append(occupied.ravel())
    label = np.zeros(len(points), dtype=np.int64); label[~occupancy[0] & occupancy[1]] = 1; label[occupancy[0] & ~occupancy[1]] = 2
    source = Path(args.depth_manifest); records = json.loads(source.read_text()); visible = [np.zeros(len(points), dtype=bool), np.zeros(len(points), dtype=bool)]
    for record in records:
        depth = np.load(source.parent / record["path"], allow_pickle=False); intrinsic, pose = np.array(record["K"]), np.array(record["T"])
        for start in range(0, len(points), 65536):
            xyz = points[start:start + 65536]; local = (xyz - pose[:3, 3]) @ pose[:3, :3]; projected = local @ intrinsic.T
            uv = np.rint(projected[:, :2] / np.maximum(projected[:, 2:3], 1e-8)).astype(int)
            valid = (local[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < depth.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < depth.shape[0])
            ids = np.flatnonzero(valid); measured = depth[uv[ids, 1], uv[ids, 0]]
            ids = ids[np.isfinite(measured) & (measured > 0) & (local[ids, 2] <= measured + args.voxel)]
            visible[int(record["visit"])][start + ids] = True
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(args.output, xyz=points.astype(np.float32), label=label, valid=visible[0] & visible[1])


if __name__ == "__main__":
    main()

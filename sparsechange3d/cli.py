import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .model import SceneField, corrected_pose, decode, observability, rays, render


def prepare(args):
    source = Path(args.scene); scene = json.loads(source.read_text(encoding="utf-8"))
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True); cameras = []; rng = np.random.default_rng(args.seed)
    for visit in [0, 1]:
        group = [r for r in scene["cameras"] if int(r["visit"]) == visit]
        if args.views and len(group) > args.views:
            group = [group[i] for i in np.sort(rng.choice(len(group), args.views, replace=False))]
        if not group:
            raise ValueError("Both visits require cameras")
        for record in group:
            image = Image.open(source.parent / record["path"]).convert("RGB")
            scale = min(1., args.maximum_side / max(image.size)); intrinsic = np.array(record["K"], dtype=float)
            width, height = [max(4, round(v * scale)) for v in image.size]
            intrinsic[0] *= width / image.width; intrinsic[1] *= height / image.height
            name = f"view_{len(cameras):04d}.png"; image.resize((width, height), Image.Resampling.LANCZOS).save(output / name)
            pose = np.array(record["T"], dtype=float)
            if intrinsic.shape != (3, 3) or pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.isfinite(intrinsic).all():
                raise ValueError("Finite K[3,3] and camera-to-world T[4,4] required")
            pose[:3, 3] += rng.normal(0, args.translation_noise, 3)
            if args.rotation_noise:
                from scipy.spatial.transform import Rotation
                pose[:3, :3] = pose[:3, :3] @ Rotation.from_rotvec(rng.normal(0, np.deg2rad(args.rotation_noise), 3)).as_matrix()
            cameras.append(dict(path=name, K=intrinsic.tolist(), T=pose.tolist(), visit=visit))
    bounds = np.array(scene["bounds"])
    if bounds.shape != (2, 3) or not np.all(bounds[1] > bounds[0]):
        raise ValueError("bounds must contain minimum/maximum XYZ in world units")
    (output / "scene.json").write_text(json.dumps(dict(bounds=scene["bounds"], cameras=cameras), indent=2))


def load_scene(root, device):
    root = Path(root); scene = json.loads((root / "scene.json").read_text()); cameras = []
    for row in scene["cameras"]:
        cameras.append({"visit": row["visit"], "K": torch.tensor(row["K"], device=device, dtype=torch.float32), "T": torch.tensor(row["T"], device=device, dtype=torch.float32), "image": torch.tensor(np.array(Image.open(root / row["path"])), device=device, dtype=torch.float32) / 255})
    return scene, cameras


def ssim_patches(prediction, target):
    mean_x, mean_y = prediction.mean(1), target.mean(1)
    variance_x, variance_y = prediction.var(1, unbiased=False), target.var(1, unbiased=False)
    covariance = ((prediction - mean_x[:, None]) * (target - mean_y[:, None])).mean(1)
    return (((2 * mean_x * mean_y + .01 ** 2) * (2 * covariance + .03 ** 2)) / ((mean_x ** 2 + mean_y ** 2 + .01 ** 2) * (variance_x + variance_y + .03 ** 2))).mean()


def fit(args):
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    scene, cameras = load_scene(args.data, args.device); model = SceneField(scene["bounds"], len(cameras), args.variant).to(args.device)
    field = [p for name, p in model.named_parameters() if name != "pose"]
    optimizer = torch.optim.Adam([{"params": field, "lr": args.lr}, {"params": [model.pose], "lr": args.pose_lr}])
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True); history = []; started = time.perf_counter()
    for step in range(args.iterations):
        model.residual_enabled = step >= args.warmup or args.variant == "independent"
        model.pose.requires_grad_(step >= 2 * args.warmup and args.variant != "no_pose")
        i = int(rng.integers(len(cameras))); camera = cameras[i]; image = camera["image"]; height, width = image.shape[:2]
        patches = max(1, args.rays // 16)
        start = np.column_stack([rng.integers(0, width - 3, patches), rng.integers(0, height - 3, patches)])
        offset = np.array([(x, y) for y in range(4) for x in range(4)])
        pixels = torch.tensor((start[:, None] + offset[None]).reshape(-1, 2), dtype=torch.float32, device=args.device)
        target = image[pixels[:, 1].long(), pixels[:, 0].long()]
        pose = corrected_pose(camera["T"], model.pose[i]); origin, direction = rays(pixels, camera["K"], pose)
        prediction, valid, points, delta, appearance = render(model, origin, direction, camera["visit"], args.samples)
        if not valid.any():
            raise ValueError("Sampled camera rays do not intersect the supplied scene bounds")
        photo = (prediction[valid] - target[valid]).abs().mean()
        patch_valid = valid.reshape(-1, 16).all(1)
        if patch_valid.any():
            photo = photo + .2 * (1 - ssim_patches(prediction.reshape(-1, 16, 3)[patch_valid], target.reshape(-1, 16, 3)[patch_valid]))
        positions = points[valid].reshape(-1, 3)
        ids = torch.randint(len(positions), (min(args.observable_samples, len(positions)),), device=args.device)
        subset = positions[ids]
        with torch.no_grad():
            support = observability(model, subset, cameras)
        _, change, _ = model.geometry(subset, camera["visit"])
        sparse = (support * change.abs()).mean()
        neighbor = subset + torch.randn_like(subset) * .002 * (model.bounds[1] - model.bounds[0])
        _, adjacent, _ = model.geometry(neighbor, camera["visit"])
        tv = (adjacent - change).abs().mean()
        if args.variant in {"no_sparsity", "independent"}:
            sparse = sparse * 0
        pose_penalty = model.pose.square().mean()
        objective = photo + args.sparsity * sparse + args.tv * tv + args.appearance * (appearance + model.visit_code.square().mean()) + args.pose_weight * pose_penalty
        optimizer.zero_grad(); objective.backward(); optimizer.step()
        fraction = .5 * (1 + np.cos(np.pi * (step + 1) / args.iterations))
        optimizer.param_groups[0]["lr"] = args.lr * (.05 + .95 * fraction); optimizer.param_groups[1]["lr"] = args.pose_lr * (.05 + .95 * fraction)
        if step % args.log_interval == 0 or step == args.iterations - 1:
            history.append(dict(iteration=step, photo=float(photo.detach()), objective=float(objective.detach()), sparse=float(sparse.detach()), appearance=float(appearance.detach())))
    torch.save({"state": model.state_dict(), "bounds": scene["bounds"], "cameras": len(cameras), "config": vars(args)}, output / "model.pt")
    (output / "history.json").write_text(json.dumps(dict(seconds=time.perf_counter() - started, iterations=history), indent=2))
    plt.plot([r["iteration"] for r in history], [r["photo"] for r in history]); plt.xlabel("Optimization iteration"); plt.ylabel("Photometric objective"); plt.savefig(output / "optimization.png", dpi=180)


def signed_metrics(truth, prediction):
    result = {}; f1 = []
    for label, name in [(1, "added"), (2, "removed")]:
        intersection = np.sum((truth == label) & (prediction == label)); union = np.sum((truth == label) | (prediction == label))
        iou = intersection / union if union else 0.; value = 2 * iou / (1 + iou); result[name + "_iou"] = iou; result[name + "_f1"] = value; f1.append(value)
    result["signed_macro_f1"] = float(np.mean(f1)); stable = truth == 0
    result["stable_false_change_percent"] = float(np.mean(np.isin(prediction[stable], [1, 2])) * 100) if stable.any() else None
    result["unknown_fraction"] = float(np.mean(prediction == 3))
    return result


def evaluate(args):
    checkpoint = torch.load(args.model, map_location=args.device, weights_only=True); _, cameras = load_scene(args.data, args.device)
    if len(cameras) != checkpoint["cameras"]:
        raise ValueError("Use the same camera manifest as scene fitting")
    model = SceneField(checkpoint["bounds"], len(cameras), checkpoint["config"]["variant"]).to(args.device); model.load_state_dict(checkpoint["state"]); model.eval()
    if args.truth:
        truth = dict(np.load(args.truth, allow_pickle=False)); points = truth["xyz"]
    else:
        bounds = np.array(checkpoint["bounds"]); axes = [np.arange(a + args.voxel / 2, b, args.voxel) for a, b in zip(bounds[0], bounds[1])]
        points = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3); truth = None
    predictions, confidences, supports, changes = [], [], [], []; started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(points), args.chunk):
            xyz = torch.tensor(points[start:start + args.chunk], device=args.device, dtype=torch.float32)
            values = decode(model, xyz, cameras, args.eta, args.probability_threshold, args.change_threshold)
            for store, value in zip([predictions, confidences, supports, changes], values):
                store.extend(value.cpu().numpy())
    prediction, confidence = np.array(predictions), np.array(confidences); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "field.npz", xyz=points, prediction=prediction, confidence=confidence, observability=np.array(supports), signed_residual=np.array(changes))
    result = {"seconds": time.perf_counter() - started, "parameters": sum(p.numel() for p in model.parameters()), "thresholds": dict(eta=args.eta, probability=args.probability_threshold, residual=args.change_threshold)}
    if truth is not None:
        valid = truth.get("valid", np.ones(len(points), dtype=bool)).astype(bool); y, pred, score = truth["label"][valid], prediction[valid], confidence[valid]
        result.update(signed_metrics(y, pred)); changed = np.isin(y, [1, 2]); directional = changed & np.isin(pred, [1, 2]); curve = []
        for threshold in np.linspace(0, 1, 101):
            retain = directional & (score >= threshold)
            curve.append(dict(threshold=float(threshold), coverage=float(retain.sum() / max(changed.sum(), 1)), risk=float(np.mean(pred[retain] != y[retain])) if retain.any() else None))
        result["directional_risk_coverage"] = curve
        rows = [r for r in curve if r["risk"] is not None]
        if rows:
            plt.plot([r["coverage"] for r in rows], [r["risk"] for r in rows]); plt.xlabel("Changed-voxel coverage"); plt.ylabel("Addition/removal risk"); plt.savefig(output / "risk_coverage.png", dpi=180)
        if "object_size" in truth:
            sizes = truth["object_size"][valid]; by_size = []
            for size in np.unique(sizes[sizes > 0]):
                selected = (sizes == size) | (y == 0); metrics = signed_metrics(y[selected], pred[selected]); by_size.append(dict(size=float(size), **metrics))
            detected = [r["size"] for r in by_size if min(r["added_f1"], r["removed_f1"]) >= .7]
            result["object_size_metrics"] = by_size; result["minimum_detectable_size"] = min(detected) if detected else None
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    selection = np.isin(prediction, [1, 2]); figure = plt.figure(); axis = figure.add_subplot(projection="3d")
    if selection.any():
        visible = np.flatnonzero(selection)[::max(1, int(selection.sum()) // 20000)]; axis.scatter(*points[visible].T, c=prediction[visible], cmap="coolwarm", s=2)
    figure.savefig(output / "signed_changes.png", dpi=180)


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(required=True)
    p = sub.add_parser("prepare"); p.add_argument("--scene", required=True); p.add_argument("--output", required=True)
    p.add_argument("--maximum-side", type=int, default=960); p.add_argument("--views", type=int, default=5); p.add_argument("--seed", type=int, default=7); p.add_argument("--translation-noise", type=float, default=0); p.add_argument("--rotation-noise", type=float, default=0); p.set_defaults(func=prepare)
    p = sub.add_parser("fit"); p.add_argument("--data", required=True); p.add_argument("--output", required=True); p.add_argument("--device", default="cpu")
    p.add_argument("--variant", choices=["full", "independent", "no_sparsity", "no_appearance", "union_support", "no_observability", "no_pose"], default="full")
    for key, value in [("iterations", 30000), ("warmup", 5000), ("rays", 1024), ("samples", 64), ("observable-samples", 32), ("log-interval", 100), ("seed", 7)]:
        p.add_argument("--" + key, type=int, default=value)
    for key, value in [("lr", .001), ("pose-lr", .0001), ("sparsity", .0002), ("tv", .00005), ("appearance", .001), ("pose-weight", .01)]:
        p.add_argument("--" + key, type=float, default=value)
    p.set_defaults(func=fit)
    p = sub.add_parser("evaluate"); p.add_argument("--data", required=True); p.add_argument("--model", required=True); p.add_argument("--output", required=True); p.add_argument("--truth"); p.add_argument("--device", default="cpu")
    p.add_argument("--voxel", type=float, default=.02); p.add_argument("--chunk", type=int, default=1024); p.add_argument("--eta", type=float, default=.2); p.add_argument("--probability-threshold", type=float, default=.25); p.add_argument("--change-threshold", type=float, default=.15); p.set_defaults(func=evaluate)
    args = parser.parse_args()
    if args.func == fit and (min(args.iterations, args.rays, args.samples, args.observable_samples, args.log_interval, args.lr, args.pose_lr) <= 0 or args.warmup < 0):
        parser.error("Positive optimization sizes/rates and nonnegative warmup required")
    if args.func == prepare and (args.views < 0 or args.maximum_side < 4 or min(args.translation_noise, args.rotation_noise) < 0):
        parser.error("Nonnegative views/noise and image maximum-side at least four required")
    if args.func == evaluate and (min(args.voxel, args.chunk, args.change_threshold) <= 0 or not 0 <= args.eta <= 1 or not 0 <= args.probability_threshold <= 1):
        parser.error("Positive export sizes/residual threshold and probability thresholds in [0,1] required")
    args.func(args)

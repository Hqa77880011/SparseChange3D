import itertools

import torch
from torch import nn
from torch.nn import functional as F


class HashEncoding(nn.Module):
    def __init__(self, levels=16, features=2, table_size=32768):
        super().__init__()
        self.tables = nn.Parameter(torch.empty(levels, table_size, features).uniform_(-1e-4, 1e-4))
        self.register_buffer("resolutions", torch.exp(torch.linspace(torch.log(torch.tensor(16.)), torch.log(torch.tensor(512.)), levels)).long())
        self.register_buffer("corners", torch.tensor(list(itertools.product([0, 1], repeat=3))))

    def forward(self, x):
        output = []
        for level, resolution in enumerate(self.resolutions):
            position = x * resolution; lower = position.floor().long(); fraction = position - lower
            corner = lower[:, None] + self.corners[None]
            index = (corner[..., 0] * 73856093 ^ corner[..., 1] * 19349663 ^ corner[..., 2] * 83492791) % self.tables.shape[1]
            weight = torch.where(self.corners[None] > 0, fraction[:, None], 1 - fraction[:, None]).prod(-1)
            output.append((self.tables[level, index] * weight[..., None]).sum(1))
        return torch.cat(output, -1)


class SceneField(nn.Module):
    def __init__(self, bounds, cameras, variant="full"):
        super().__init__()
        self.register_buffer("bounds", torch.as_tensor(bounds, dtype=torch.float32))
        self.encoding = HashEncoding(); self.variant = variant
        self.second_encoding = HashEncoding() if variant == "independent" else None
        self.canonical = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        self.change = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.constant_(self.canonical[-1].bias, -3)
        nn.init.zeros_(self.change[-1].weight); nn.init.zeros_(self.change[-1].bias)
        self.color = nn.Sequential(nn.Linear(35, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
        self.appearance = nn.Sequential(nn.Linear(43, 16), nn.ReLU(), nn.Linear(16, 3))
        self.visit_code = nn.Parameter(torch.randn(8) * .01)
        self.pose = nn.Parameter(torch.zeros(cameras, 6))
        self.residual_enabled = True

    def geometry(self, x, visit):
        normalized = (x - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        feature = self.encoding(normalized.clamp(0, 1)); base = self.canonical(feature).squeeze(-1); delta = self.change(feature).squeeze(-1)
        if not self.residual_enabled:
            delta = delta * 0
        if self.variant == "independent":
            second_feature = self.second_encoding(normalized.clamp(0, 1))
            first, second = base, self.change(second_feature).squeeze(-1)
            delta = (second - first) / 2; base = (first + second) / 2
        logit = base + (2 * visit - 1) * delta
        return logit, delta, feature

    def forward(self, x, direction, visit):
        logit, delta, feature = self.geometry(x, visit)
        shared = self.color(torch.cat([feature, direction], -1)).sigmoid()
        code = self.visit_code * (2 * visit - 1)
        appearance = .1 * self.appearance(torch.cat([feature, direction, code.expand(len(x), -1)], -1)).tanh()
        if self.variant == "no_appearance":
            appearance = appearance * 0
        return F.softplus(logit), (shared + appearance).clamp(0, 1), delta, appearance


def corrected_pose(initial, increment):
    x, y, z = increment[3:]
    zero = x * 0
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero]).reshape(3, 3)
    tangent = torch.cat([torch.cat([skew, increment[:3, None]], 1), increment.new_zeros(1, 4)], 0)
    return initial @ torch.linalg.matrix_exp(tangent)


def rays(pixels, intrinsic, pose):
    directions = torch.cat([pixels, torch.ones_like(pixels[:, :1])], -1) @ torch.linalg.inv(intrinsic).T
    directions = directions @ pose[:3, :3].T; directions = F.normalize(directions, dim=-1)
    return pose[:3, 3].expand_as(directions), directions


def intersections(origin, direction, bounds):
    safe = torch.where(direction.abs() < 1e-8, torch.full_like(direction, 1e-8), direction)
    distance = (bounds[None] - origin[:, None]) / safe[:, None]
    near = distance.amin(1).amax(-1).clamp_min(0); far = distance.amax(1).amin(-1)
    return near, far, far > near


def render(model, origin, direction, visit, samples=64):
    near, far, valid = intersections(origin, direction, model.bounds)
    far = torch.maximum(far, near + 1e-6)
    step = (far - near) / samples
    fractions = (torch.arange(samples, device=origin.device) + .5) / samples
    distance = near[:, None] + (far - near)[:, None] * fractions
    points = origin[:, None] + direction[:, None] * distance[..., None]
    sigma, color, delta, appearance = model(points.reshape(-1, 3), direction[:, None].expand(-1, samples, -1).reshape(-1, 3), visit)
    sigma = sigma.reshape(-1, samples); color = color.reshape(-1, samples, 3)
    optical = sigma * step[:, None]; transmittance = torch.exp(-torch.cumsum(optical, -1) + optical)
    weight = (1 - torch.exp(-optical)) * transmittance
    prediction = (weight[..., None] * color).sum(1)
    return prediction, valid, points, delta.reshape(-1, samples), appearance.square().mean()


def observability(model, points, cameras, samples=24):
    support = []
    for visit in [0, 1]:
        no_reach = torch.ones(len(points), device=points.device); weighted_direction = torch.zeros_like(points); mass = torch.zeros(len(points), device=points.device)
        for i, camera in enumerate(cameras):
            if camera["visit"] != visit:
                continue
            pose = corrected_pose(camera["T"], model.pose[i]); local = (points - pose[:3, 3]) @ pose[:3, :3]
            projected = local @ camera["K"].T; uv = projected[:, :2] / projected[:, 2:3].clamp_min(1e-8)
            height, width = camera["image"].shape[:2]
            frustum = (local[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)
            soft = torch.sigmoid(uv[:, 0]) * torch.sigmoid(width - 1 - uv[:, 0]) * torch.sigmoid(uv[:, 1]) * torch.sigmoid(height - 1 - uv[:, 1])
            origin = pose[:3, 3].expand_as(points); vector = points - origin; distance = vector.norm(dim=-1); direction = F.normalize(vector, dim=-1)
            near, _, _ = intersections(origin, direction, model.bounds)
            length = (distance - near).clamp_min(0)
            fractions = (torch.arange(samples, device=points.device) + .5) / samples
            positions = origin[:, None] + direction[:, None] * (near[:, None] + length[:, None] * fractions)[..., None]
            logit, _, _ = model.geometry(positions.reshape(-1, 3), visit)
            reach = torch.exp(-(F.softplus(logit).reshape(-1, samples).sum(1) * length / samples)) * frustum * soft
            no_reach = no_reach * (1 - reach); weighted_direction += reach[:, None] * direction; mass += reach
        diversity = (1 - weighted_direction.norm(dim=-1) / mass.clamp_min(1e-8)).clamp(0, 1)
        support.append((1 - no_reach) * diversity)
    if model.variant == "union_support":
        return torch.maximum(support[0], support[1])
    if model.variant == "no_observability":
        return torch.ones_like(support[0])
    return torch.minimum(support[0], support[1])


def decode(model, points, cameras, eta, probability_threshold, change_threshold):
    first, delta, _ = model.geometry(points, 0); second, _, _ = model.geometry(points, 1)
    difference = second.sigmoid() - first.sigmoid(); support = observability(model, points, cameras)
    prediction = torch.zeros(len(points), dtype=torch.long, device=points.device)
    prediction[(difference > probability_threshold) & (delta > change_threshold)] = 1
    prediction[(difference < -probability_threshold) & (delta < -change_threshold)] = 2
    prediction[support < eta] = 3
    confidence = support * (delta.abs() / change_threshold).clamp_max(1)
    return prediction, confidence, support, delta

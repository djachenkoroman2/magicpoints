#!/usr/bin/env python3
"""
Synthetic labeled landscape point cloud generator.

The script generates a procedural scene where each point has:
    x, y, z, label

Labels:
    0 - natural surface
    1 - artificial surface
    2 - low vegetation
    3 - high vegetation
    4 - buildings
    5 - structures
    6 - vehicles
    7 - artifacts
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np


CLASS_NAMES: Dict[int, str] = {
    0: "Natural surface",
    1: "Artificial surface",
    2: "Low vegetation",
    3: "High vegetation",
    4: "Buildings",
    5: "Structures",
    6: "Vehicles",
    7: "Artifacts",
}

CLASS_COLORS: Dict[int, Tuple[float, float, float]] = {
    0: (0.55, 0.39, 0.22),  # brown
    1: (0.35, 0.35, 0.35),  # asphalt gray
    2: (0.35, 0.75, 0.30),  # light green
    3: (0.05, 0.45, 0.12),  # dark green
    4: (0.82, 0.22, 0.18),  # red
    5: (0.85, 0.68, 0.20),  # yellow
    6: (0.15, 0.40, 0.85),  # blue
    7: (0.68, 0.18, 0.72),  # magenta
}

Rect = Tuple[float, float, float, float]  # center_x, center_y, size_x, size_y


def _validate_positive(value: float, name: str) -> None:
    """Validate positive numeric inputs."""
    if value <= 0.0:
        raise ValueError(f"`{name}` must be > 0, got {value}.")


def _validate_unit_interval(value: float, name: str) -> None:
    """Validate that a value is inside [0, 1]."""
    if value < 0.0 or value > 1.0:
        raise ValueError(f"`{name}` must be in [0, 1], got {value}.")


def allocate_points(total_points: int, class_ratios: Dict[int, float]) -> Dict[int, int]:
    """
    Split total points across classes according to ratios.
    The output always sums to total_points.
    """
    labels = sorted(class_ratios)
    weights = np.array([class_ratios[k] for k in labels], dtype=np.float64)
    if np.any(weights < 0):
        raise ValueError("Class ratios must be non-negative.")
    if np.isclose(weights.sum(), 0.0):
        raise ValueError("At least one class ratio must be positive.")

    weights = weights / weights.sum()
    raw = weights * int(total_points)
    counts = np.floor(raw).astype(int)
    remainder = int(total_points) - int(counts.sum())
    if remainder > 0:
        frac = raw - counts
        for idx in np.argsort(frac)[::-1][:remainder]:
            counts[idx] += 1

    return {label: int(count) for label, count in zip(labels, counts)}


def _split_count_by_weights(total: int, weights: Sequence[float]) -> np.ndarray:
    """Split integer `total` by continuous `weights` while preserving sum."""
    weights_arr = np.array(weights, dtype=np.float64)
    if np.isclose(weights_arr.sum(), 0.0):
        out = np.zeros(len(weights_arr), dtype=int)
        if len(out) > 0:
            out[0] = total
        return out

    weights_arr = weights_arr / weights_arr.sum()
    raw = weights_arr * total
    base = np.floor(raw).astype(int)
    remainder = total - int(base.sum())
    if remainder > 0:
        frac = raw - base
        for idx in np.argsort(frac)[::-1][:remainder]:
            base[idx] += 1
    return base


def _split_count_evenly(total: int, n_chunks: int) -> np.ndarray:
    """Even integer split where first chunks get remainder."""
    if n_chunks <= 0:
        return np.array([], dtype=int)
    base = np.full(n_chunks, total // n_chunks, dtype=int)
    base[: (total % n_chunks)] += 1
    return base


def _split_count_random(
    total: int, n_chunks: int, rng: np.random.Generator, min_per_chunk: int = 1
) -> np.ndarray:
    """
    Random integer split that preserves the exact total.
    Falls back to even split when `total` is too small.
    """
    if n_chunks <= 0:
        return np.array([], dtype=int)
    if total <= 0:
        return np.zeros(n_chunks, dtype=int)

    min_per_chunk = max(0, int(min_per_chunk))
    if min_per_chunk * n_chunks > total:
        return _split_count_evenly(total, n_chunks)

    counts = np.full(n_chunks, min_per_chunk, dtype=int)
    remaining = total - int(counts.sum())
    if remaining > 0:
        weights = rng.dirichlet(np.ones(n_chunks, dtype=np.float64))
        raw = weights * remaining
        extra = np.floor(raw).astype(int)
        remainder = remaining - int(extra.sum())
        if remainder > 0:
            frac = raw - extra
            for idx in np.argsort(frac)[::-1][:remainder]:
                extra[idx] += 1
        counts += extra

    rng.shuffle(counts)
    return counts


def _inside_rectangles(
    x: np.ndarray, y: np.ndarray, rectangles: Sequence[Rect], margin: float = 0.0
) -> np.ndarray:
    """Mask of points that are inside at least one axis-aligned rectangle."""
    if len(rectangles) == 0:
        return np.zeros_like(x, dtype=bool)

    mask = np.zeros_like(x, dtype=bool)
    for cx, cy, sx, sy in rectangles:
        hx = max(0.0, sx * 0.5 - margin)
        hy = max(0.0, sy * 0.5 - margin)
        mask |= (np.abs(x - cx) <= hx) & (np.abs(y - cy) <= hy)
    return mask


def _sample_xy(
    rng: np.random.Generator,
    n: int,
    area_size: Tuple[float, float],
    forbidden_rects: Sequence[Rect] | None = None,
    allowed_rects: Sequence[Rect] | None = None,
    margin: float = 0.0,
    max_tries: int = 80,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample XY points in area with optional constraints.
    Falls back to unconstrained sampling if constraints are too strict.
    """
    if n <= 0:
        return np.empty(0), np.empty(0)

    forbidden_rects = list(forbidden_rects or [])
    allowed_rects = list(allowed_rects or [])

    w, h = area_size
    x_min, x_max = -w * 0.5, w * 0.5
    y_min, y_max = -h * 0.5, h * 0.5

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    remaining = n
    tries = 0

    while remaining > 0 and tries < max_tries:
        batch = max(512, remaining * 4)
        x_try = rng.uniform(x_min, x_max, size=batch)
        y_try = rng.uniform(y_min, y_max, size=batch)

        mask = np.ones(batch, dtype=bool)
        if allowed_rects:
            mask &= _inside_rectangles(x_try, y_try, allowed_rects, margin=margin)
        if forbidden_rects:
            mask &= ~_inside_rectangles(x_try, y_try, forbidden_rects, margin=margin)

        if np.any(mask):
            accepted = np.flatnonzero(mask)
            take = min(remaining, accepted.size)
            idx = accepted[:take]
            xs.append(x_try[idx])
            ys.append(y_try[idx])
            remaining -= take

        tries += 1

    # Fallback: ensure exact count even if strict constraints cannot be fully satisfied.
    if remaining > 0:
        xs.append(rng.uniform(x_min, x_max, size=remaining))
        ys.append(rng.uniform(y_min, y_max, size=remaining))

    return np.concatenate(xs), np.concatenate(ys)


def _sample_inside_rect(
    rng: np.random.Generator, n: int, rect: Rect, margin: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample points uniformly inside one rectangle."""
    if n <= 0:
        return np.empty(0), np.empty(0)

    cx, cy, sx, sy = rect
    hx = max(0.2, sx * 0.5 - margin)
    hy = max(0.2, sy * 0.5 - margin)
    x = rng.uniform(cx - hx, cx + hx, size=n)
    y = rng.uniform(cy - hy, cy + hy, size=n)
    return x, y


def _sample_single_xy(
    rng: np.random.Generator,
    area_size: Tuple[float, float],
    forbidden_rects: Sequence[Rect] | None = None,
    allowed_rects: Sequence[Rect] | None = None,
    margin: float = 0.0,
) -> Tuple[float, float]:
    """Sample one XY point."""
    x, y = _sample_xy(
        rng,
        n=1,
        area_size=area_size,
        forbidden_rects=forbidden_rects,
        allowed_rects=allowed_rects,
        margin=margin,
    )
    return float(x[0]), float(y[0])


def _rect_overlap_ratio(rect_a: Rect, rect_b: Rect) -> float:
    """Overlap ratio normalized by the smaller rectangle area."""
    ax, ay, aw, ah = rect_a
    bx, by, bw, bh = rect_b

    a_left, a_right = ax - 0.5 * aw, ax + 0.5 * aw
    a_bottom, a_top = ay - 0.5 * ah, ay + 0.5 * ah
    b_left, b_right = bx - 0.5 * bw, bx + 0.5 * bw
    b_bottom, b_top = by - 0.5 * bh, by + 0.5 * bh

    inter_w = max(0.0, min(a_right, b_right) - max(a_left, b_left))
    inter_h = max(0.0, min(a_top, b_top) - max(a_bottom, b_bottom))
    inter_area = inter_w * inter_h
    min_area = max(1e-9, min(aw * ah, bw * bh))
    return inter_area / min_area


def generate_terrain(
    area_size: Tuple[float, float] = (240.0, 220.0),
    n_points: int = 40_000,
    seed: int = 42,
    terrain_relief: float = 1.0,
) -> Tuple[np.ndarray, Callable[[np.ndarray, np.ndarray], np.ndarray], List[Dict[str, object]]]:
    """
    Generate natural terrain (class 0) and return:
      - terrain points (N, 4)
      - terrain height function z=f(x,y)
      - artificial zones metadata for roads/plazas
    `terrain_relief` controls elevation amplitude in [0,1]:
      0 -> almost flat, 1 -> mountainous.
    """
    rng = np.random.default_rng(seed)
    area_w, area_h = area_size
    scale = max(area_w, area_h)
    relief = float(terrain_relief)
    _validate_unit_interval(relief, "terrain_relief")

    def terrain_fn(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        # Multi-frequency sinusoidal field + Gaussian hills/valleys.
        z = 2.2 * np.sin(2.0 * np.pi * x_arr / (0.75 * area_w))
        z += 1.5 * np.cos(2.0 * np.pi * y_arr / (0.60 * area_h))
        z += 1.1 * np.sin(2.0 * np.pi * (x_arr + y_arr) / (0.92 * scale))
        z += 0.7 * np.cos(2.0 * np.pi * (x_arr - 1.7 * y_arr) / (1.05 * scale))

        hill = 4.0 * np.exp(
            -(
                ((x_arr - 0.20 * area_w) ** 2) / (2.0 * (0.16 * area_w) ** 2)
                + ((y_arr + 0.24 * area_h) ** 2) / (2.0 * (0.18 * area_h) ** 2)
            )
        )
        valley = -3.0 * np.exp(
            -(
                ((x_arr + 0.28 * area_w) ** 2) / (2.0 * (0.20 * area_w) ** 2)
                + ((y_arr - 0.18 * area_h) ** 2) / (2.0 * (0.16 * area_h) ** 2)
            )
        )
        return relief * (z + hill + valley)

    # Random artificial zones: roads + plazas.
    artificial_zones: List[Dict[str, object]] = []
    min_side = min(area_w, area_h)
    road_min_w = max(6.0, 0.03 * min_side)
    road_max_w = max(road_min_w + 1.5, 0.09 * min_side)

    n_roads = int(rng.integers(2, 6))
    orientations = rng.choice(["horizontal", "vertical"], size=n_roads, p=[0.52, 0.48])
    if n_roads >= 2:
        orientations[0] = "horizontal"
        orientations[1] = "vertical"

    for idx, orientation in enumerate(orientations, start=1):
        road_rect: Rect = (0.0, 0.0, 10.0, 8.0)
        for _ in range(40):
            width = float(rng.uniform(road_min_w, road_max_w))
            if orientation == "horizontal":
                length = float(rng.uniform(0.50 * area_w, 0.98 * area_w))
                cx = float(rng.uniform(-0.18 * area_w, 0.18 * area_w))
                cy = float(rng.uniform(-0.45 * area_h, 0.45 * area_h))
                candidate = (cx, cy, length, width)
            else:
                length = float(rng.uniform(0.50 * area_h, 0.98 * area_h))
                cx = float(rng.uniform(-0.45 * area_w, 0.45 * area_w))
                cy = float(rng.uniform(-0.18 * area_h, 0.18 * area_h))
                candidate = (cx, cy, width, length)

            if not artificial_zones:
                road_rect = candidate
                break

            overlap = max(
                _rect_overlap_ratio(candidate, zone["rect"])  # type: ignore[index]
                for zone in artificial_zones
            )
            if overlap < 0.88:
                road_rect = candidate
                break

        artificial_zones.append({"name": f"road_{idx}", "kind": "road", "rect": road_rect})

    n_plazas = int(rng.integers(1, 3))
    for idx in range(1, n_plazas + 1):
        plaza_rect: Rect = (0.0, 0.0, max(16.0, 0.15 * area_w), max(14.0, 0.13 * area_h))
        for _ in range(40):
            sx = float(rng.uniform(max(14.0, 0.08 * area_w), max(20.0, 0.24 * area_w)))
            sy = float(rng.uniform(max(12.0, 0.08 * area_h), max(18.0, 0.24 * area_h)))
            cx = float(rng.uniform(-0.42 * area_w, 0.42 * area_w))
            cy = float(rng.uniform(-0.42 * area_h, 0.42 * area_h))
            candidate = (cx, cy, sx, sy)
            overlap = (
                max(
                    _rect_overlap_ratio(candidate, zone["rect"])  # type: ignore[index]
                    for zone in artificial_zones
                )
                if artificial_zones
                else 0.0
            )
            if overlap < 0.92:
                plaza_rect = candidate
                break
        artificial_zones.append({"name": f"plaza_{idx}", "kind": "plaza", "rect": plaza_rect})

    for zone in artificial_zones:
        cx, cy, _, _ = zone["rect"]  # type: ignore[index]
        zone["z0"] = float(terrain_fn(np.array([cx]), np.array([cy]))[0] + rng.uniform(0.03, 0.12))

    x_nat, y_nat = _sample_xy(
        rng,
        n=n_points,
        area_size=area_size,
        forbidden_rects=[zone["rect"] for zone in artificial_zones],  # type: ignore[index]
        margin=0.0,
    )
    terrain_noise = 0.01 + 0.05 * relief
    z_nat = terrain_fn(x_nat, y_nat) + rng.normal(0.0, terrain_noise, size=n_points)
    labels = np.zeros(n_points, dtype=np.int32)
    terrain_points = np.column_stack((x_nat, y_nat, z_nat, labels))
    return terrain_points, terrain_fn, artificial_zones


def _generate_building_points(
    rng: np.random.Generator,
    n_points: int,
    cx: float,
    cy: float,
    width: float,
    depth: float,
    height: float,
    base_z: float,
) -> np.ndarray:
    """Points on walls and roof of one rectangular building."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)

    roof_n = max(1, int(n_points * 0.22))
    wall_n = n_points - roof_n

    x_roof = rng.uniform(cx - width * 0.5, cx + width * 0.5, size=roof_n)
    y_roof = rng.uniform(cy - depth * 0.5, cy + depth * 0.5, size=roof_n)
    z_roof = np.full(roof_n, base_z + height) + rng.normal(0.0, 0.02, size=roof_n)

    face = rng.integers(0, 4, size=wall_n)
    x_wall = np.empty(wall_n, dtype=np.float64)
    y_wall = np.empty(wall_n, dtype=np.float64)
    z_wall = rng.uniform(base_z, base_z + height, size=wall_n)
    u = rng.uniform(-0.5, 0.5, size=wall_n)

    # Sample one of 4 walls per point.
    left = face == 0
    right = face == 1
    front = face == 2
    back = face == 3

    x_wall[left] = cx - width * 0.5
    y_wall[left] = cy + u[left] * depth
    x_wall[right] = cx + width * 0.5
    y_wall[right] = cy + u[right] * depth
    x_wall[front] = cx + u[front] * width
    y_wall[front] = cy + depth * 0.5
    x_wall[back] = cx + u[back] * width
    y_wall[back] = cy - depth * 0.5

    x = np.concatenate([x_roof, x_wall]) + rng.normal(0.0, 0.015, size=n_points)
    y = np.concatenate([y_roof, y_wall]) + rng.normal(0.0, 0.015, size=n_points)
    z = np.concatenate([z_roof, z_wall])
    labels = np.full(n_points, 4, dtype=np.int32)
    return np.column_stack((x, y, z, labels))


def _generate_tree_points(
    rng: np.random.Generator,
    n_points: int,
    cx: float,
    cy: float,
    ground_z: float,
    trunk_radius: float,
    trunk_height: float,
    crown_radius: float,
) -> np.ndarray:
    """Points for one tree: trunk cylinder + crown sphere volume."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)

    trunk_n = max(1, int(n_points * 0.35))
    crown_n = n_points - trunk_n

    # Trunk as points near cylinder surface.
    theta = rng.uniform(0.0, 2.0 * np.pi, size=trunk_n)
    radial = trunk_radius + rng.normal(0.0, trunk_radius * 0.08, size=trunk_n)
    x_trunk = cx + radial * np.cos(theta)
    y_trunk = cy + radial * np.sin(theta)
    z_trunk = ground_z + rng.uniform(0.0, trunk_height, size=trunk_n)

    # Crown as random points inside a sphere around trunk top.
    if crown_n > 0:
        direction = rng.normal(size=(crown_n, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-12
        radius = crown_radius * np.cbrt(rng.uniform(0.0, 1.0, size=crown_n))
        crown = direction * radius[:, None]
        crown[:, 0] += cx
        crown[:, 1] += cy
        crown[:, 2] += ground_z + trunk_height + 0.45 * crown_radius
        crown[:, 2] = np.maximum(crown[:, 2], ground_z + 0.7 * trunk_height)
        x_crown, y_crown, z_crown = crown[:, 0], crown[:, 1], crown[:, 2]
    else:
        x_crown = np.empty(0)
        y_crown = np.empty(0)
        z_crown = np.empty(0)

    x = np.concatenate([x_trunk, x_crown])
    y = np.concatenate([y_trunk, y_crown])
    z = np.concatenate([z_trunk, z_crown])
    labels = np.full(x.size, 3, dtype=np.int32)
    return np.column_stack((x, y, z, labels))


def _generate_fence_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_points: int,
    x_center: float,
    y_center: float,
    length: float,
    angle: float,
) -> np.ndarray:
    """Generate elongated fence-like structure."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)

    n_rail = int(n_points * 0.65)
    n_post = n_points - n_rail

    direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)

    # Rails: two horizontal lines along fence.
    t = rng.uniform(-0.5, 0.5, size=n_rail) * length
    lateral = rng.normal(0.0, 0.05, size=n_rail)
    x_rail = x_center + direction[0] * t + normal[0] * lateral
    y_rail = y_center + direction[1] * t + normal[1] * lateral
    base_rail = terrain_fn(x_rail, y_rail)
    rail_height = rng.choice([0.75, 1.25], size=n_rail)
    z_rail = base_rail + rail_height + rng.normal(0.0, 0.03, size=n_rail)

    # Posts: vertical elements near discrete fence positions.
    if n_post > 0:
        n_segments = max(4, int(length // 3.0))
        post_t = rng.integers(0, n_segments + 1, size=n_post) / max(1, n_segments)
        post_t = (post_t - 0.5) * length
        post_lateral = rng.normal(0.0, 0.04, size=n_post)
        x_post = x_center + direction[0] * post_t + normal[0] * post_lateral
        y_post = y_center + direction[1] * post_t + normal[1] * post_lateral
        post_base = terrain_fn(x_post, y_post)
        z_post = post_base + rng.uniform(0.0, 1.9, size=n_post)
    else:
        x_post = np.empty(0)
        y_post = np.empty(0)
        z_post = np.empty(0)

    x = np.concatenate([x_rail, x_post])
    y = np.concatenate([y_rail, y_post])
    z = np.concatenate([z_rail, z_post])
    labels = np.full(x.size, 5, dtype=np.int32)
    return np.column_stack((x, y, z, labels))


def _generate_support_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_points: int,
    x_center: float,
    y_center: float,
    height: float,
) -> np.ndarray:
    """Generate pole/support structure with a cross-beam."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)

    n_pole = int(n_points * 0.6)
    n_beam = n_points - n_pole

    base = float(terrain_fn(np.array([x_center]), np.array([y_center]))[0])
    pole_radius = 0.18

    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_pole)
    x_pole = x_center + pole_radius * np.cos(theta)
    y_pole = y_center + pole_radius * np.sin(theta)
    z_pole = base + rng.uniform(0.0, height, size=n_pole)

    if n_beam > 0:
        beam_length = rng.uniform(4.0, 9.0)
        beam_angle = rng.uniform(0.0, 2.0 * np.pi)
        t = rng.uniform(-0.5, 0.5, size=n_beam) * beam_length
        beam_radius = rng.uniform(0.05, 0.12, size=n_beam)
        normal_angle = beam_angle + np.pi * 0.5
        x_beam = (
            x_center
            + np.cos(beam_angle) * t
            + np.cos(normal_angle) * beam_radius * rng.choice([-1.0, 1.0], size=n_beam)
        )
        y_beam = (
            y_center
            + np.sin(beam_angle) * t
            + np.sin(normal_angle) * beam_radius * rng.choice([-1.0, 1.0], size=n_beam)
        )
        z_beam = np.full(n_beam, base + 0.8 * height) + rng.normal(0.0, 0.04, size=n_beam)
    else:
        x_beam = np.empty(0)
        y_beam = np.empty(0)
        z_beam = np.empty(0)

    x = np.concatenate([x_pole, x_beam])
    y = np.concatenate([y_pole, y_beam])
    z = np.concatenate([z_pole, z_beam])
    labels = np.full(x.size, 5, dtype=np.int32)
    return np.column_stack((x, y, z, labels))


def _generate_lattice_structure_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_points: int,
    x_center: float,
    y_center: float,
    width: float,
    length: float,
    height: float,
    yaw: float,
) -> np.ndarray:
    """Generate random truss/lattice-like structure."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)

    base = float(terrain_fn(np.array([x_center]), np.array([y_center]))[0])
    if n_points < 3:
        x = x_center + rng.normal(0.0, 0.08, size=n_points)
        y = y_center + rng.normal(0.0, 0.08, size=n_points)
        z = base + rng.uniform(0.0, height, size=n_points)
        labels = np.full(n_points, 5, dtype=np.int32)
        return np.column_stack((x, y, z, labels))

    n_columns = max(1, int(n_points * 0.40))
    n_beams = max(1, int(n_points * 0.32))
    if n_columns + n_beams >= n_points:
        n_beams = max(1, n_points - n_columns)
    n_braces = max(0, n_points - n_columns - n_beams)

    corners = np.array(
        [
            [-0.5 * width, -0.5 * length],
            [0.5 * width, -0.5 * length],
            [0.5 * width, 0.5 * length],
            [-0.5 * width, 0.5 * length],
        ],
        dtype=np.float64,
    )

    # Vertical columns near corner points.
    col_idx = rng.integers(0, 4, size=n_columns)
    x_col = corners[col_idx, 0] + rng.normal(0.0, 0.04, size=n_columns)
    y_col = corners[col_idx, 1] + rng.normal(0.0, 0.04, size=n_columns)
    z_col = rng.uniform(0.0, height, size=n_columns)

    # Horizontal beams along perimeter at random levels.
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
    edge_idx = rng.integers(0, 4, size=n_beams)
    t_beam = rng.uniform(0.0, 1.0, size=n_beams)
    p0 = corners[edges[edge_idx, 0]]
    p1 = corners[edges[edge_idx, 1]]
    beam_xy = p0 + (p1 - p0) * t_beam[:, None]
    x_beam = beam_xy[:, 0] + rng.normal(0.0, 0.03, size=n_beams)
    y_beam = beam_xy[:, 1] + rng.normal(0.0, 0.03, size=n_beams)
    level = rng.choice([0.25, 0.5, 0.75, 1.0], size=n_beams, p=[0.25, 0.30, 0.30, 0.15])
    z_beam = level * height + rng.normal(0.0, 0.03, size=n_beams)

    # Diagonal braces between different corners.
    if n_braces > 0:
        brace_pairs = np.array([[0, 2], [1, 3], [0, 1], [2, 3], [1, 2], [3, 0]], dtype=np.int32)
        pair_idx = rng.integers(0, len(brace_pairs), size=n_braces)
        t_brace = rng.uniform(0.0, 1.0, size=n_braces)
        b0 = corners[brace_pairs[pair_idx, 0]]
        b1 = corners[brace_pairs[pair_idx, 1]]
        brace_xy = b0 + (b1 - b0) * t_brace[:, None]
        x_brace = brace_xy[:, 0] + rng.normal(0.0, 0.03, size=n_braces)
        y_brace = brace_xy[:, 1] + rng.normal(0.0, 0.03, size=n_braces)
        up = pair_idx % 2 == 0
        z_brace = np.where(up, t_brace * height, (1.0 - t_brace) * height)
        z_brace += rng.normal(0.0, 0.03, size=n_braces)
    else:
        x_brace = np.empty(0)
        y_brace = np.empty(0)
        z_brace = np.empty(0)

    x_local = np.concatenate([x_col, x_beam, x_brace])
    y_local = np.concatenate([y_col, y_beam, y_brace])
    z_local = np.concatenate([z_col, z_beam, z_brace])

    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    x = x_center + x_local * cos_yaw - y_local * sin_yaw
    y = y_center + x_local * sin_yaw + y_local * cos_yaw
    z = base + z_local
    labels = np.full(x.size, 5, dtype=np.int32)
    return np.column_stack((x, y, z, labels))


def _generate_bridge_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_points: int,
    x_center: float,
    y_center: float,
    length: float,
    width: float,
    clearance: float,
    yaw: float,
) -> np.ndarray:
    """Generate bridge-like structure: deck + side rails + vertical supports."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)

    center_ground = float(terrain_fn(np.array([x_center]), np.array([y_center]))[0])
    deck_level = center_ground + clearance
    deck_level += float(rng.uniform(0.0, 0.35))
    pier_radius = float(rng.uniform(0.14, 0.28))

    n_deck = max(1, int(n_points * 0.48))
    n_rail = max(1, int(n_points * 0.20))
    n_support = max(0, n_points - n_deck - n_rail)

    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

    def _local_to_world(x_local: np.ndarray, y_local: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_world = x_center + x_local * cos_yaw - y_local * sin_yaw
        y_world = y_center + x_local * sin_yaw + y_local * cos_yaw
        return x_world, y_world

    # Deck surface.
    t_deck = rng.uniform(-0.5, 0.5, size=n_deck) * length
    lat_deck = rng.uniform(-0.5, 0.5, size=n_deck) * width
    x_deck, y_deck = _local_to_world(t_deck, lat_deck)
    z_deck = np.full(n_deck, deck_level) + rng.normal(0.0, 0.02, size=n_deck)

    # Side rails.
    side = rng.choice([-1.0, 1.0], size=n_rail)
    t_rail = rng.uniform(-0.5, 0.5, size=n_rail) * length
    lat_rail = side * (0.5 * width - 0.1) + rng.normal(0.0, 0.03, size=n_rail)
    x_rail, y_rail = _local_to_world(t_rail, lat_rail)
    z_rail = deck_level + rng.uniform(0.55, 1.25, size=n_rail) + rng.normal(0.0, 0.02, size=n_rail)

    # Vertical supports.
    if n_support > 0:
        n_piers = int(np.clip(length / rng.uniform(7.0, 12.0), 2, 8))
        t_centers = np.linspace(-0.42 * length, 0.42 * length, n_piers)
        t_centers += rng.normal(0.0, 0.02 * length, size=n_piers)
        per_pier = _split_count_random(
            n_support,
            n_piers,
            rng=rng,
            min_per_chunk=1 if n_support < n_piers * 3 else 3,
        )

        support_x: List[np.ndarray] = []
        support_y: List[np.ndarray] = []
        support_z: List[np.ndarray] = []
        for t0, pier_n in zip(t_centers, per_pier):
            if pier_n <= 0:
                continue
            lat0 = float(rng.normal(0.0, 0.10 * width))
            x0_arr, y0_arr = _local_to_world(np.array([t0]), np.array([lat0]))
            x0 = float(x0_arr[0])
            y0 = float(y0_arr[0])
            ground = float(terrain_fn(np.array([x0]), np.array([y0]))[0])
            height = max(0.8, deck_level - ground)

            theta = rng.uniform(0.0, 2.0 * np.pi, size=int(pier_n))
            radial = pier_radius + rng.normal(0.0, pier_radius * 0.10, size=int(pier_n))
            support_x.append(x0 + radial * np.cos(theta))
            support_y.append(y0 + radial * np.sin(theta))
            support_z.append(ground + rng.uniform(0.0, height, size=int(pier_n)))

        if support_x:
            x_sup = np.concatenate(support_x)
            y_sup = np.concatenate(support_y)
            z_sup = np.concatenate(support_z)
        else:
            x_sup = np.empty(0)
            y_sup = np.empty(0)
            z_sup = np.empty(0)
    else:
        x_sup = np.empty(0)
        y_sup = np.empty(0)
        z_sup = np.empty(0)

    x = np.concatenate([x_deck, x_rail, x_sup])
    y = np.concatenate([y_deck, y_rail, y_sup])
    z = np.concatenate([z_deck, z_rail, z_sup])
    labels = np.full(x.size, 5, dtype=np.int32)
    return np.column_stack((x, y, z, labels))


def _generate_vehicle_points(
    rng: np.random.Generator,
    n_points: int,
    x_center: float,
    y_center: float,
    base_z: float,
    length: float,
    width: float,
    height: float,
    yaw: float,
) -> np.ndarray:
    """Generate compact cuboid-like vehicle points."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)

    face = rng.integers(0, 6, size=n_points)
    u = rng.uniform(-0.5, 0.5, size=n_points)
    v = rng.uniform(-0.5, 0.5, size=n_points)

    local = np.empty((n_points, 3), dtype=np.float64)

    plus_x = face == 0
    minus_x = face == 1
    plus_y = face == 2
    minus_y = face == 3
    plus_z = face == 4
    minus_z = face == 5

    local[plus_x] = np.column_stack(
        [
            np.full(np.sum(plus_x), +0.5 * length),
            u[plus_x] * width,
            v[plus_x] * height,
        ]
    )
    local[minus_x] = np.column_stack(
        [
            np.full(np.sum(minus_x), -0.5 * length),
            u[minus_x] * width,
            v[minus_x] * height,
        ]
    )
    local[plus_y] = np.column_stack(
        [
            u[plus_y] * length,
            np.full(np.sum(plus_y), +0.5 * width),
            v[plus_y] * height,
        ]
    )
    local[minus_y] = np.column_stack(
        [
            u[minus_y] * length,
            np.full(np.sum(minus_y), -0.5 * width),
            v[minus_y] * height,
        ]
    )
    local[plus_z] = np.column_stack(
        [
            u[plus_z] * length,
            v[plus_z] * width,
            np.full(np.sum(plus_z), +0.5 * height),
        ]
    )
    local[minus_z] = np.column_stack(
        [
            u[minus_z] * length,
            v[minus_z] * width,
            np.full(np.sum(minus_z), -0.5 * height),
        ]
    )

    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    x = x_center + local[:, 0] * cos_yaw - local[:, 1] * sin_yaw
    y = y_center + local[:, 0] * sin_yaw + local[:, 1] * cos_yaw
    z = base_z + local[:, 2] + 0.5 * height
    z += rng.normal(0.0, 0.01, size=n_points)

    labels = np.full(n_points, 6, dtype=np.int32)
    return np.column_stack((x, y, z, labels))


def place_objects(
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    area_size: Tuple[float, float],
    points_per_class: Dict[int, int],
    artificial_zones: Sequence[Dict[str, object]],
    num_trees: int = 70,
    num_buildings: int = 14,
    num_structures: int = 10,
    num_vehicles: int = 24,
    num_artifact_clusters: int = 40,
    seed: int = 43,
) -> np.ndarray:
    """
    Place all non-terrain classes and return one (N, 4) array.
    Uses classes:
      1 artificial surfaces
      2 low vegetation
      3 high vegetation
      4 buildings
      5 structures
      6 vehicles
      7 artifacts
    """
    rng = np.random.default_rng(seed)
    cloud_parts: List[np.ndarray] = []

    artificial_rects: List[Rect] = [zone["rect"] for zone in artificial_zones]  # type: ignore[index]
    building_rects: List[Rect] = []

    # ------------------------------------------------------------------
    # Class 1: artificial surfaces (roads/plaza as flattened patches)
    # ------------------------------------------------------------------
    n_artificial = int(points_per_class.get(1, 0))
    if n_artificial > 0 and len(artificial_zones) > 0:
        areas = [rect[2] * rect[3] for rect in artificial_rects]
        zone_counts = _split_count_by_weights(n_artificial, areas)
        zones_points = []
        for zone, zone_n in zip(artificial_zones, zone_counts):
            if zone_n <= 0:
                continue
            rect = zone["rect"]  # type: ignore[index]
            x, y = _sample_inside_rect(rng, zone_n, rect, margin=0.35)
            z0 = float(zone["z0"])  # type: ignore[index]
            z = np.full(zone_n, z0) + rng.normal(0.0, 0.01, size=zone_n)
            labels = np.full(zone_n, 1, dtype=np.int32)
            zones_points.append(np.column_stack((x, y, z, labels)))
        if zones_points:
            cloud_parts.append(np.vstack(zones_points))

    # ------------------------------------------------------------------
    # Class 4: buildings (cuboids sampled on walls + roof)
    # ------------------------------------------------------------------
    n_building_points = int(points_per_class.get(4, 0))
    if n_building_points > 0:
        n_buildings_eff = max(1, min(num_buildings, n_building_points))
        points_per_building = _split_count_evenly(n_building_points, n_buildings_eff)
        for n_pts in points_per_building:
            width = float(rng.uniform(8.0, 20.0))
            depth = float(rng.uniform(8.0, 18.0))
            height = float(rng.uniform(7.0, 28.0))

            # Keep buildings away from roads and other buildings.
            forbidden = artificial_rects + [
                (cx, cy, sx + 6.0, sy + 6.0) for cx, cy, sx, sy in building_rects
            ]
            cx, cy = _sample_single_xy(
                rng, area_size=area_size, forbidden_rects=forbidden, margin=0.2
            )
            base_z = float(terrain_fn(np.array([cx]), np.array([cy]))[0] + 0.08)

            building_rects.append((cx, cy, width, depth))
            cloud_parts.append(
                _generate_building_points(
                    rng=rng,
                    n_points=int(n_pts),
                    cx=cx,
                    cy=cy,
                    width=width,
                    depth=depth,
                    height=height,
                    base_z=base_z,
                )
            )

    # ------------------------------------------------------------------
    # Class 3: high vegetation (trees)
    # ------------------------------------------------------------------
    n_tree_points = int(points_per_class.get(3, 0))
    if n_tree_points > 0:
        n_trees_eff = max(1, min(num_trees, n_tree_points))
        per_tree = _split_count_evenly(n_tree_points, n_trees_eff)
        tree_forbidden = artificial_rects + [
            (cx, cy, sx + 3.5, sy + 3.5) for cx, cy, sx, sy in building_rects
        ]
        for n_pts in per_tree:
            cx, cy = _sample_single_xy(
                rng, area_size=area_size, forbidden_rects=tree_forbidden, margin=0.2
            )
            ground_z = float(terrain_fn(np.array([cx]), np.array([cy]))[0])
            trunk_radius = float(rng.uniform(0.14, 0.35))
            trunk_height = float(rng.uniform(2.5, 6.5))
            crown_radius = float(rng.uniform(1.2, 3.6))
            cloud_parts.append(
                _generate_tree_points(
                    rng=rng,
                    n_points=int(n_pts),
                    cx=cx,
                    cy=cy,
                    ground_z=ground_z,
                    trunk_radius=trunk_radius,
                    trunk_height=trunk_height,
                    crown_radius=crown_radius,
                )
            )

    # ------------------------------------------------------------------
    # Class 5: structures (fully randomized type + parameters, incl. bridges)
    # ------------------------------------------------------------------
    n_structure_points = int(points_per_class.get(5, 0))
    if n_structure_points > 0:
        n_struct_eff = max(1, min(num_structures, n_structure_points))
        per_struct = _split_count_random(
            n_structure_points, n_struct_eff, rng=rng, min_per_chunk=8
        )
        struct_forbidden = [
            (cx, cy, sx + 2.0, sy + 2.0) for cx, cy, sx, sy in building_rects
        ]
        structure_types = rng.choice(
            ["fence", "support", "lattice", "bridge"],
            size=n_struct_eff,
            p=[0.32, 0.24, 0.20, 0.24],
        )
        if n_struct_eff >= 3 and np.all(structure_types != "bridge") and rng.random() < 0.75:
            structure_types[int(rng.integers(0, n_struct_eff))] = "bridge"
        for n_pts, structure_type in zip(per_struct, structure_types):
            cx, cy = _sample_single_xy(
                rng, area_size=area_size, forbidden_rects=struct_forbidden, margin=0.2
            )
            if structure_type == "fence":
                cloud_parts.append(
                    _generate_fence_points(
                        rng=rng,
                        terrain_fn=terrain_fn,
                        n_points=int(n_pts),
                        x_center=cx,
                        y_center=cy,
                        length=float(rng.uniform(8.0, 46.0)),
                        angle=float(rng.uniform(0.0, 2.0 * np.pi)),
                    )
                )
            elif structure_type == "support":
                cloud_parts.append(
                    _generate_support_points(
                        rng=rng,
                        terrain_fn=terrain_fn,
                        n_points=int(n_pts),
                        x_center=cx,
                        y_center=cy,
                        height=float(rng.uniform(3.5, 12.0)),
                    )
                )
            else:
                if structure_type == "lattice":
                    cloud_parts.append(
                        _generate_lattice_structure_points(
                            rng=rng,
                            terrain_fn=terrain_fn,
                            n_points=int(n_pts),
                            x_center=cx,
                            y_center=cy,
                            width=float(rng.uniform(2.0, 7.0)),
                            length=float(rng.uniform(3.0, 12.0)),
                            height=float(rng.uniform(3.0, 11.0)),
                            yaw=float(rng.uniform(0.0, 2.0 * np.pi)),
                        )
                    )
                else:
                    cloud_parts.append(
                        _generate_bridge_points(
                            rng=rng,
                            terrain_fn=terrain_fn,
                            n_points=int(n_pts),
                            x_center=cx,
                            y_center=cy,
                            length=float(rng.uniform(10.0, max(20.0, 0.42 * min(area_size)))),
                            width=float(rng.uniform(2.8, 7.8)),
                            clearance=float(rng.uniform(1.2, 5.2)),
                            yaw=float(rng.uniform(0.0, 2.0 * np.pi)),
                        )
                    )

    # ------------------------------------------------------------------
    # Class 6: vehicles (compact cuboids on roads/plaza)
    # ------------------------------------------------------------------
    n_vehicle_points = int(points_per_class.get(6, 0))
    if n_vehicle_points > 0 and len(artificial_zones) > 0:
        n_vehicle_eff = max(1, min(num_vehicles, n_vehicle_points))
        per_vehicle = _split_count_evenly(n_vehicle_points, n_vehicle_eff)
        zone_prob = np.array([rect[2] * rect[3] for rect in artificial_rects], dtype=np.float64)
        zone_prob /= zone_prob.sum()
        for n_pts in per_vehicle:
            zone_idx = int(rng.choice(len(artificial_zones), p=zone_prob))
            zone = artificial_zones[zone_idx]
            rect = zone["rect"]  # type: ignore[index]

            length = float(rng.uniform(3.4, 5.8))
            width = float(rng.uniform(1.6, 2.3))
            height = float(rng.uniform(1.2, 2.2))
            margin = 0.6 * max(length, width)
            cx, cy = _sample_inside_rect(rng, 1, rect, margin=margin)
            cx_f, cy_f = float(cx[0]), float(cy[0])

            sx, sy = rect[2], rect[3]
            base_angle = 0.0 if sx >= sy else np.pi * 0.5
            yaw = float(base_angle + rng.normal(0.0, 0.18))
            base_z = float(zone["z0"]) + 0.06  # type: ignore[index]

            cloud_parts.append(
                _generate_vehicle_points(
                    rng=rng,
                    n_points=int(n_pts),
                    x_center=cx_f,
                    y_center=cy_f,
                    base_z=base_z,
                    length=length,
                    width=width,
                    height=height,
                    yaw=yaw,
                )
            )

    # ------------------------------------------------------------------
    # Class 2: low vegetation (small offsets above ground)
    # ------------------------------------------------------------------
    n_low_veg = int(points_per_class.get(2, 0))
    if n_low_veg > 0:
        low_veg_forbidden = artificial_rects + building_rects
        x, y = _sample_xy(
            rng=rng,
            n=n_low_veg,
            area_size=area_size,
            forbidden_rects=low_veg_forbidden,
            margin=0.1,
        )
        z = terrain_fn(x, y) + rng.uniform(0.05, 0.85, size=n_low_veg)
        labels = np.full(n_low_veg, 2, dtype=np.int32)
        cloud_parts.append(np.column_stack((x, y, z, labels)))

    # ------------------------------------------------------------------
    # Class 7: artifacts (small random clusters and isolated points)
    # ------------------------------------------------------------------
    n_artifacts = int(points_per_class.get(7, 0))
    if n_artifacts > 0:
        n_clusters = max(1, min(num_artifact_clusters, n_artifacts))
        per_cluster = _split_count_evenly(n_artifacts, n_clusters)
        artifact_parts = []
        for cluster_size in per_cluster:
            cx, cy = _sample_single_xy(rng, area_size=area_size)
            spread = float(rng.uniform(0.05, 0.45))
            x = cx + rng.normal(0.0, spread, size=int(cluster_size))
            y = cy + rng.normal(0.0, spread, size=int(cluster_size))
            z = terrain_fn(x, y) + rng.uniform(0.0, 1.4, size=int(cluster_size))
            labels = np.full(int(cluster_size), 7, dtype=np.int32)
            artifact_parts.append(np.column_stack((x, y, z, labels)))
        cloud_parts.append(np.vstack(artifact_parts))

    if not cloud_parts:
        return np.empty((0, 4), dtype=np.float64)
    return np.vstack(cloud_parts)


def visualize_point_cloud(points: np.ndarray, labels: np.ndarray) -> None:
    """
    Interactive 3D scatter visualization with color-by-class and legend.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    xyz = np.asarray(points, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("`points` must be an array of shape (N, 3).")
    if labels_arr.shape[0] != xyz.shape[0]:
        raise ValueError("`labels` size must match number of points.")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    legend_handles: List[Line2D] = []
    for class_id in sorted(CLASS_NAMES):
        mask = labels_arr == class_id
        count = int(mask.sum())
        if count == 0:
            continue
        color = CLASS_COLORS[class_id]
        ax.scatter(
            xyz[mask, 0],
            xyz[mask, 1],
            xyz[mask, 2],
            c=[color],
            s=1.2,
            alpha=0.95,
            linewidths=0.0,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"{class_id}: {CLASS_NAMES[class_id]} ({count})",
                markerfacecolor=color,
                markersize=7,
            )
        )

    ax.set_title("Synthetic Labeled Point Cloud")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()


def export_to_csv(point_cloud: np.ndarray, output_path: Path) -> None:
    """Export (N,4) cloud to CSV."""
    cloud = np.array(point_cloud, copy=True)
    cloud[:, 3] = cloud[:, 3].astype(np.int32)
    fmt = ["%.6f", "%.6f", "%.6f", "%d"]
    np.savetxt(
        output_path,
        cloud,
        delimiter=",",
        header="x,y,z,label",
        comments="",
        fmt=fmt,
    )


def export_to_ply(point_cloud: np.ndarray, output_path: Path) -> None:
    """
    Export to ASCII PLY with per-point color and label.
    Useful for CloudCompare / MeshLab import.
    """
    xyz = point_cloud[:, :3]
    labels = point_cloud[:, 3].astype(np.int32)
    rgb = np.array([CLASS_COLORS[int(l)] for l in labels], dtype=np.float64)
    rgb = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar label\n")
        f.write("end_header\n")
        for p, c, l in zip(xyz, rgb, labels):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])} {int(l)}\n"
            )


def _print_stats(labels: Iterable[int]) -> None:
    """Print total count and per-class statistics."""
    labels_arr = np.asarray(list(labels), dtype=np.int32)
    print(f"Total points: {labels_arr.size}")
    for class_id in sorted(CLASS_NAMES):
        count = int(np.sum(labels_arr == class_id))
        print(f"Class {class_id} ({CLASS_NAMES[class_id]}): {count}")


def _generate_object_counts(
    area_size: Tuple[float, float],
    rng: np.random.Generator,
    randomize_counts: bool,
) -> Dict[str, int]:
    """
    Generate object counts based on scene area.
    If randomize_counts is True, values are sampled from area-scaled ranges.
    """
    width_m, length_m = area_size
    area_m2 = width_m * length_m
    area_ha = area_m2 / 10_000.0

    # Expected object density per hectare.
    densities_per_ha = {
        "num_trees": 14.0,
        "num_buildings": 2.6,
        "num_structures": 1.6,
        "num_vehicles": 4.8,
        "num_artifact_clusters": 7.5,
    }
    min_counts = {
        "num_trees": 8,
        "num_buildings": 3,
        "num_structures": 2,
        "num_vehicles": 3,
        "num_artifact_clusters": 8,
    }

    counts: Dict[str, int] = {}
    for key, density in densities_per_ha.items():
        expected = max(float(min_counts[key]), density * area_ha)
        if randomize_counts:
            low = max(min_counts[key], int(np.floor(expected * 0.65)))
            high = max(low + 1, int(np.ceil(expected * 1.45 + 2.0)))
            counts[key] = int(rng.integers(low, high + 1))
        else:
            counts[key] = max(min_counts[key], int(round(expected)))
    return counts


def _print_scene_params(
    area_size: Tuple[float, float],
    seed: int,
    terrain_relief: float,
    randomize_object_counts: bool,
    object_counts: Dict[str, int],
) -> None:
    """Print generated scene parameters for easier reproducibility."""
    width_m, length_m = area_size
    area_m2 = width_m * length_m
    mode = "randomized" if randomize_object_counts else "deterministic"
    print(
        f"Scene area: width={width_m:.1f} m, length={length_m:.1f} m, "
        f"total={area_m2:.1f} m^2"
    )
    print(f"Random seed: {seed}")
    print(f"Terrain relief [0..1]: {terrain_relief:.2f}")
    print(f"Object count mode: {mode}")
    print(
        "Object counts: "
        f"trees={object_counts['num_trees']}, "
        f"buildings={object_counts['num_buildings']}, "
        f"structures={object_counts['num_structures']}, "
        f"vehicles={object_counts['num_vehicles']}, "
        f"artifact_clusters={object_counts['num_artifact_clusters']}"
    )


def _run_pipeline(
    total_points: int,
    show_visualization: bool,
    save_csv: bool,
    save_ply: bool,
    area_width: float,
    area_length: float,
    terrain_relief: float,
    randomize_object_counts: bool,
    seed: int,
) -> np.ndarray:
    """
    Internal execution pipeline with configurable runtime switches.
    """
    if total_points <= 0:
        raise ValueError("`total_points` must be > 0.")
    _validate_positive(area_width, "area_width")
    _validate_positive(area_length, "area_length")
    _validate_unit_interval(terrain_relief, "terrain_relief")

    # ----------------------------- Scene config -----------------------------
    area_size = (float(area_width), float(area_length))
    scene_rng = np.random.default_rng(seed + 100)
    object_counts = _generate_object_counts(
        area_size=area_size,
        rng=scene_rng,
        randomize_counts=randomize_object_counts,
    )
    num_trees = object_counts["num_trees"]
    num_buildings = object_counts["num_buildings"]
    num_structures = object_counts["num_structures"]
    num_vehicles = object_counts["num_vehicles"]
    num_artifact_clusters = object_counts["num_artifact_clusters"]
    _print_scene_params(
        area_size=area_size,
        seed=seed,
        terrain_relief=terrain_relief,
        randomize_object_counts=randomize_object_counts,
        object_counts=object_counts,
    )

    # Proportional allocation across all 8 classes.
    class_ratios = {
        0: 0.38,
        1: 0.13,
        2: 0.16,
        3: 0.14,
        4: 0.10,
        5: 0.04,
        6: 0.03,
        7: 0.02,
    }

    points_per_class = allocate_points(total_points=total_points, class_ratios=class_ratios)

    terrain_points, terrain_fn, artificial_zones = generate_terrain(
        area_size=area_size,
        n_points=points_per_class[0],
        seed=seed,
        terrain_relief=terrain_relief,
    )
    n_roads = sum(1 for zone in artificial_zones if zone.get("kind") == "road")
    n_plazas = sum(1 for zone in artificial_zones if zone.get("kind") == "plaza")
    print(f"Artificial zones: roads={n_roads}, plazas={n_plazas}")

    object_points = place_objects(
        terrain_fn=terrain_fn,
        area_size=area_size,
        points_per_class=points_per_class,
        artificial_zones=artificial_zones,
        num_trees=num_trees,
        num_buildings=num_buildings,
        num_structures=num_structures,
        num_vehicles=num_vehicles,
        num_artifact_clusters=num_artifact_clusters,
        seed=seed + 1,
    )

    point_cloud = np.vstack([terrain_points, object_points])

    # Shuffle rows so classes are not grouped by generation stage.
    rng = np.random.default_rng(seed + 2)
    rng.shuffle(point_cloud, axis=0)

    # Ensure exact requested count if constraints fallback produced extras.
    if point_cloud.shape[0] > total_points:
        point_cloud = point_cloud[:total_points]
    elif point_cloud.shape[0] < total_points:
        missing = total_points - point_cloud.shape[0]
        x_pad, y_pad = _sample_xy(rng, missing, area_size=area_size)
        z_pad = terrain_fn(x_pad, y_pad)
        l_pad = np.zeros(missing, dtype=np.int32)
        point_cloud = np.vstack([point_cloud, np.column_stack((x_pad, y_pad, z_pad, l_pad))])

    labels = point_cloud[:, 3].astype(np.int32)
    _print_stats(labels)

    if save_csv:
        export_to_csv(point_cloud, Path("synthetic_landscape_point_cloud.csv"))
        print("Saved CSV: synthetic_landscape_point_cloud.csv")
    if save_ply:
        export_to_ply(point_cloud, Path("synthetic_landscape_point_cloud.ply"))
        print("Saved PLY: synthetic_landscape_point_cloud.ply")

    if show_visualization:
        visualize_point_cloud(point_cloud[:, :3], labels)

    return point_cloud


def generate_point_cloud(
    total_points: int = 100_000,
    area_width: float = 240.0,
    area_length: float = 220.0,
    terrain_relief: float = 1.0,
    randomize_object_counts: bool = True,
    seed: int = 12,
) -> np.ndarray:
    """
    Public API for programmatic generation without side effects:
      - no matplotlib window
      - no CSV/PLY files written
    Returns point cloud array with shape (N, 4): x, y, z, label.
    """
    return _run_pipeline(
        total_points=int(total_points),
        show_visualization=False,
        save_csv=False,
        save_ply=False,
        area_width=float(area_width),
        area_length=float(area_length),
        terrain_relief=float(terrain_relief),
        randomize_object_counts=bool(randomize_object_counts),
        seed=int(seed),
    )


def main(
    total_points: int = 100_000,
    area_width: float = 240.0,
    area_length: float = 220.0,
    terrain_relief: float = 1.0,
    randomize_object_counts: bool = True,
    seed: int = 12,
) -> np.ndarray:
    """
    Entry point required by task.
    By default:
      - generates 100k points
      - uses area 240x220 meters
      - uses mountainous terrain relief (`terrain_relief=1.0`)
      - randomizes object counts based on area
      - prints stats
      - saves CSV + PLY
      - opens interactive visualization
    """
    return _run_pipeline(
        total_points=int(total_points),
        show_visualization=True,
        save_csv=True,
        save_ply=True,
        area_width=float(area_width),
        area_length=float(area_length),
        terrain_relief=float(terrain_relief),
        randomize_object_counts=bool(randomize_object_counts),
        seed=int(seed),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic labeled landscape point cloud."
    )
    parser.add_argument(
        "--total-points",
        type=int,
        default=100_000,
        help="Total number of generated points.",
    )
    parser.add_argument(
        "--area-width",
        type=float,
        default=240.0,
        help="Scene width along X axis in meters.",
    )
    parser.add_argument(
        "--area-length",
        type=float,
        default=220.0,
        help="Scene length along Y axis in meters.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12,
        help="Random seed for reproducible scene generation.",
    )
    parser.add_argument(
        "--terrain-relief",
        type=float,
        default=1.0,
        help="Terrain elevation intensity in [0,1]: 0=flat, 1=mountainous.",
    )
    parser.add_argument(
        "--random-object-counts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable randomized object counts (trees/buildings/structures/vehicles/artifacts) "
            "based on scene area."
        ),
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable interactive visualization window.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Disable CSV export.",
    )
    parser.add_argument(
        "--no-ply",
        action="store_true",
        help="Disable PLY export.",
    )
    return parser


def cli(argv: Sequence[str] | None = None) -> int:
    """Command line entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    _run_pipeline(
        total_points=int(args.total_points),
        show_visualization=not args.no_visualize,
        save_csv=not args.no_csv,
        save_ply=not args.no_ply,
        area_width=float(args.area_width),
        area_length=float(args.area_length),
        terrain_relief=float(args.terrain_relief),
        randomize_object_counts=bool(args.random_object_counts),
        seed=int(args.seed),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SyntheticSplatWorld:
    xyz: np.ndarray
    object_id: np.ndarray
    ring_index: np.ndarray

    @property
    def n(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def object_count(self) -> int:
        return int(np.max(self.object_id)) + 1


def make_braided_world(
    n_per_object: int = 60,
    object_count: int = 3,
    braid_radius: float = 0.08,
) -> SyntheticSplatWorld:
    """Create interleaved 3-D strands that are visually close but logically distinct.

    The geometry is only for visualization in Gate 0. The relation graph is supplied
    separately so the first gate is an explicit positive control rather than a claim
    that object structure was discovered from XYZ.
    """
    if object_count < 2:
        raise ValueError("object_count must be >= 2")
    if n_per_object < 8:
        raise ValueError("n_per_object must be >= 8")

    theta = np.linspace(0.0, 2.0 * np.pi, n_per_object, endpoint=False)
    xyz_parts: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    ring_indices: list[np.ndarray] = []

    for obj in range(object_count):
        phase = 2.0 * np.pi * obj / object_count
        radial = 1.0 + braid_radius * np.cos(3.0 * theta + phase)
        x = radial * np.cos(theta)
        y = radial * np.sin(theta)
        z = braid_radius * np.sin(3.0 * theta + phase)
        xyz_parts.append(np.column_stack([x, y, z]))
        ids.append(np.full(n_per_object, obj, dtype=np.int32))
        ring_indices.append(np.arange(n_per_object, dtype=np.int32))

    return SyntheticSplatWorld(
        xyz=np.vstack(xyz_parts).astype(np.float64),
        object_id=np.concatenate(ids),
        ring_index=np.concatenate(ring_indices),
    )


def build_constraint_laplacian(
    object_id: np.ndarray,
    ring_index: np.ndarray,
    *,
    ring_weight: float = 1.0,
    cross_weight: float = 0.25,
) -> np.ndarray:
    """Sparse local constraint graph, returned as its dense Laplacian for tiny toys.

    Each object is a ring. At every ring index, corresponding splats across objects
    are coupled. The graph therefore contains both within-object propagation paths
    and cross-object opportunities for interference.
    """
    object_id = np.asarray(object_id, dtype=np.int32)
    ring_index = np.asarray(ring_index, dtype=np.int32)
    if object_id.shape != ring_index.shape:
        raise ValueError("object_id and ring_index must have equal shape")

    n = object_id.size
    adjacency = np.zeros((n, n), dtype=np.float64)
    objects = np.unique(object_id)

    for obj in objects:
        idx = np.where(object_id == obj)[0]
        order = idx[np.argsort(ring_index[idx])]
        for a, b in zip(order, np.roll(order, -1), strict=True):
            adjacency[a, b] += ring_weight
            adjacency[b, a] += ring_weight

    unique_ring = np.unique(ring_index)
    for r in unique_ring:
        idx = np.where(ring_index == r)[0]
        for a_pos in range(len(idx)):
            for b_pos in range(a_pos + 1, len(idx)):
                a, b = int(idx[a_pos]), int(idx[b_pos])
                adjacency[a, b] += cross_weight
                adjacency[b, a] += cross_weight

    degree = adjacency.sum(axis=1)
    return np.diag(degree) - adjacency


def build_material_stiffness(
    laplacian: np.ndarray,
    object_id: np.ndarray,
    base_frequencies: Iterable[float],
) -> np.ndarray:
    """Compile sparse constraints plus local material frequencies into K."""
    base = np.asarray(tuple(base_frequencies), dtype=np.float64)
    object_id = np.asarray(object_id, dtype=np.int32)
    if base.size <= int(np.max(object_id)):
        raise ValueError("one base frequency is required per object")
    local_stiffness = base[object_id] ** 2
    return np.asarray(laplacian, dtype=np.float64) + np.diag(local_stiffness)


def calibrated_uniform_mode_addresses(
    stiffness: np.ndarray,
    object_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positive-control address calibration for Gate 0.

    We project K onto the subspace that is constant within each known synthetic
    object, diagonalize that small matrix, and use sqrt(eigenvalue) as a drive
    frequency. This intentionally uses the benchmark labels. Gate 0 therefore tests
    whether the operator *can* ring a whole object from a fragment, not whether it
    can discover semantic addresses unaided.
    """
    object_id = np.asarray(object_id, dtype=np.int32)
    objects = np.unique(object_id)
    n = object_id.size
    U = np.zeros((n, len(objects)), dtype=np.float64)
    for col, obj in enumerate(objects):
        idx = np.where(object_id == obj)[0]
        U[idx, col] = 1.0 / np.sqrt(len(idx))

    reduced = U.T @ stiffness @ U
    evals, evecs = np.linalg.eigh(reduced)
    if np.any(evals <= 0.0):
        raise ValueError("positive resonance frequencies require positive eigenvalues")

    frequencies = np.sqrt(evals)
    dominant_object = objects[np.argmax(np.abs(evecs), axis=0)]
    if len(np.unique(dominant_object)) != len(objects):
        raise RuntimeError("uniform modes are not one-to-one with synthetic objects")
    return frequencies, dominant_object.astype(np.int32), evecs


def resonant_response(
    stiffness: np.ndarray,
    cue: np.ndarray,
    omega: float,
    *,
    damping: float = 0.01,
) -> np.ndarray:
    """Steady-state response of a driven damped second-order splat medium.

    H(omega) = [K - omega^2 I + i gamma omega I]^-1
    """
    stiffness = np.asarray(stiffness, dtype=np.float64)
    cue = np.asarray(cue, dtype=np.complex128)
    n = stiffness.shape[0]
    if stiffness.shape != (n, n) or cue.shape != (n,):
        raise ValueError("shape mismatch")
    A = (
        stiffness
        - (float(omega) ** 2) * np.eye(n)
        + 1j * float(damping) * float(omega) * np.eye(n)
    )
    return np.linalg.solve(A, cue)


def passive_response(
    laplacian: np.ndarray,
    cue: np.ndarray,
    *,
    leak: float = 0.1,
) -> np.ndarray:
    """Matched-graph non-resonant control: a leaky diffusive resolvent."""
    laplacian = np.asarray(laplacian, dtype=np.float64)
    cue = np.asarray(cue, dtype=np.complex128)
    n = laplacian.shape[0]
    A = float(leak) * np.eye(n) + laplacian
    return np.linalg.solve(A.astype(np.complex128), cue)


def make_fragment_cue(
    world: SyntheticSplatWorld,
    target_object: int,
    *,
    start: int = 0,
    count: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.where(world.object_id == target_object)[0]
    order = idx[np.argsort(world.ring_index[idx])]
    count = min(int(count), len(order))
    local = (np.arange(count) + int(start)) % len(order)
    cue_idx = order[local]
    cue = np.zeros(world.n, dtype=np.complex128)
    cue[cue_idx] = 1.0
    return cue, cue_idx


def completion_metrics(
    response: np.ndarray,
    object_id: np.ndarray,
    target_object: int,
    cue_idx: np.ndarray,
) -> dict[str, float]:
    """Score completion away from the explicitly stimulated splats."""
    response = np.asarray(response)
    object_id = np.asarray(object_id)
    cue_idx = np.asarray(cue_idx, dtype=np.int64)
    energy = np.abs(response) ** 2
    not_cue = np.ones(energy.size, dtype=bool)
    not_cue[cue_idx] = False
    target = object_id == target_object
    target_noncue = target & not_cue

    total = float(np.sum(energy))
    noncue_total = float(np.sum(energy[not_cue]))
    target_noncue_energy = float(np.sum(energy[target_noncue]))
    cue_energy = float(np.sum(energy[cue_idx]))

    target_amp = np.abs(response[target])
    ones = np.ones_like(target_amp)
    denom = float(np.linalg.norm(target_amp) * np.linalg.norm(ones))
    uniformity = float(np.dot(target_amp, ones) / denom) if denom > 0.0 else 0.0

    return {
        "target_total_purity": float(np.sum(energy[target]) / total) if total else 0.0,
        "completion_purity": target_noncue_energy / noncue_total if noncue_total else 0.0,
        "completion_gain": target_noncue_energy / cue_energy if cue_energy else 0.0,
        "target_uniformity_cosine": uniformity,
        "response_norm": float(np.linalg.norm(response)),
    }


def response_rgb(response: np.ndarray, cue_idx: np.ndarray | None = None) -> np.ndarray:
    """Map response magnitude to a simple blue->red diagnostic color."""
    amp = np.abs(np.asarray(response, dtype=np.complex128))
    scale = np.percentile(amp, 98.0)
    if scale <= 0.0:
        scale = 1.0
    x = np.clip(amp / scale, 0.0, 1.0)
    rgb = np.column_stack([255.0 * x, 40.0 + 120.0 * (1.0 - x), 255.0 * (1.0 - x)])
    if cue_idx is not None:
        rgb[np.asarray(cue_idx, dtype=np.int64)] = np.array([255.0, 255.0, 255.0])
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def write_ascii_ply(path: str | Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if xyz.shape != (rgb.shape[0], 3) or rgb.shape[1] != 3:
        raise ValueError("xyz and rgb must both be Nx3")

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz, rgb, strict=True):
            f.write(
                f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HypercubeSplatWorld:
    """Explicit splats carrying several crossing binary relation coordinates."""

    xyz: np.ndarray
    bits: np.ndarray

    @property
    def n(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def dimensions(self) -> int:
        return int(self.bits.shape[1])


def make_hypercube_world(dimensions: int = 6) -> HypercubeSplatWorld:
    if dimensions < 3:
        raise ValueError("dimensions must be >= 3")
    if dimensions > 10:
        raise ValueError("this dense toy intentionally stays small")

    n = 1 << dimensions
    bits = np.array(
        [[(i >> k) & 1 for k in range(dimensions)] for i in range(n)],
        dtype=np.int8,
    )

    # Deterministic 3-D projection for diagnostics only. Relation structure is
    # carried by the sparse material graph, not inferred from XYZ.
    angles = np.linspace(0.0, 2.0 * np.pi, dimensions, endpoint=False)
    basis = np.column_stack(
        [np.cos(angles), np.sin(angles), np.linspace(-0.7, 0.7, dimensions)]
    )
    centered = 2.0 * bits.astype(np.float64) - 1.0
    xyz = centered @ basis / np.sqrt(float(dimensions))
    return HypercubeSplatWorld(xyz=xyz, bits=bits)


def build_anisotropic_hypercube_laplacian(
    dimensions: int,
    edge_weights: np.ndarray,
) -> np.ndarray:
    weights = np.asarray(edge_weights, dtype=np.float64)
    if weights.shape != (dimensions,):
        raise ValueError("one edge weight is required per hypercube dimension")
    if np.any(weights <= 0.0):
        raise ValueError("edge weights must be positive")

    n = 1 << dimensions
    adjacency = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for dim, weight in enumerate(weights.tolist()):
            j = i ^ (1 << dim)
            adjacency[i, j] = float(weight)
    degree = adjacency.sum(axis=1)
    return np.diag(degree) - adjacency


def build_polysemantic_stiffness(
    dimensions: int,
    edge_weights: np.ndarray,
    *,
    base_stiffness: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    laplacian = build_anisotropic_hypercube_laplacian(dimensions, edge_weights)
    stiffness = laplacian + float(base_stiffness) * np.eye(laplacian.shape[0])
    return stiffness, laplacian


def relation_address_frequency(
    edge_weight: float,
    *,
    base_stiffness: float = 0.35,
) -> float:
    """Frequency of the single-bit Walsh mode for one hypercube dimension."""
    value = float(base_stiffness) + 2.0 * float(edge_weight)
    if value <= 0.0:
        raise ValueError("frequency requires positive stiffness")
    return float(np.sqrt(value))


def make_shared_fragment_cue(
    world: HypercubeSplatWorld,
    relation_dimensions: tuple[int, ...] = (0, 1, 2),
    *,
    count: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose one fragment lying on the same side of every tested relation."""
    mask = np.ones(world.n, dtype=bool)
    for dim in relation_dimensions:
        mask &= world.bits[:, int(dim)] == 0
    candidates = np.where(mask)[0]
    if candidates.size < count:
        raise ValueError("not enough splats share the requested relation intersection")

    positions = np.linspace(0, candidates.size, count, endpoint=False, dtype=int)
    cue_idx = candidates[positions]
    cue = np.zeros(world.n, dtype=np.complex128)
    cue[cue_idx] = 1.0
    return cue, cue_idx


def align_to_cue_phase(response: np.ndarray, cue_idx: np.ndarray) -> np.ndarray:
    response = np.asarray(response, dtype=np.complex128)
    cue_idx = np.asarray(cue_idx, dtype=np.int64)
    reference = np.mean(response[cue_idx])
    if abs(reference) == 0.0:
        return response.copy()
    return response * np.exp(-1j * np.angle(reference))


def relation_metrics(
    response: np.ndarray,
    bits: np.ndarray,
    relation_dimension: int,
    cue_idx: np.ndarray,
) -> dict[str, float]:
    """Evaluate one crossing relation using phase and magnitude separately."""
    response = np.asarray(response, dtype=np.complex128)
    bits = np.asarray(bits, dtype=np.int8)
    cue_idx = np.asarray(cue_idx, dtype=np.int64)
    dim = int(relation_dimension)

    aligned = align_to_cue_phase(response, cue_idx)
    signed = np.real(aligned)
    same_side = bits[:, dim] == bits[int(cue_idx[0]), dim]
    not_cue = np.ones(response.size, dtype=bool)
    not_cue[cue_idx] = False
    target = same_side & not_cue

    predicted_same = signed >= 0.0
    phase_accuracy = float(np.mean(predicted_same[not_cue] == same_side[not_cue]))

    positive_energy = np.maximum(signed, 0.0) ** 2
    positive_total = float(np.sum(positive_energy[not_cue]))
    phase_completion_purity = (
        float(np.sum(positive_energy[target])) / positive_total if positive_total else 0.0
    )

    magnitude_energy = np.abs(response) ** 2
    magnitude_total = float(np.sum(magnitude_energy[not_cue]))
    magnitude_only_purity = (
        float(np.sum(magnitude_energy[target])) / magnitude_total if magnitude_total else 0.0
    )

    target_signed = signed[target]
    distractor_signed = signed[(~same_side) & not_cue]
    return {
        "phase_relation_accuracy": phase_accuracy,
        "phase_completion_purity": phase_completion_purity,
        "magnitude_only_purity": magnitude_only_purity,
        "target_signed_mean": float(np.mean(target_signed)),
        "distractor_signed_mean": float(np.mean(distractor_signed)),
        "phase_margin": float(np.mean(target_signed) - np.mean(distractor_signed)),
        "response_norm": float(np.linalg.norm(response)),
    }


def phase_relation_rgb(response: np.ndarray, cue_idx: np.ndarray) -> np.ndarray:
    aligned = align_to_cue_phase(response, cue_idx)
    signed = np.real(aligned)
    scale = float(np.percentile(np.abs(signed), 98.0))
    if scale <= 0.0:
        scale = 1.0
    x = np.clip(signed / scale, -1.0, 1.0)
    red = 40.0 + 215.0 * np.maximum(x, 0.0)
    blue = 40.0 + 215.0 * np.maximum(-x, 0.0)
    green = 45.0 + 90.0 * (1.0 - np.abs(x))
    rgb = np.column_stack([red, green, blue])
    rgb[np.asarray(cue_idx, dtype=np.int64)] = np.array([255.0, 255.0, 255.0])
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)

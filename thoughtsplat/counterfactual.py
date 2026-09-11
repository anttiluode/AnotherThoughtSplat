from __future__ import annotations

import hashlib

import numpy as np

from thoughtsplat.polysemantic import align_to_cue_phase


def array_sha256(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(str(arr.shape).encode("utf-8"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def counterfactual_scalar_field(
    response: np.ndarray,
    cue_idx: np.ndarray,
) -> np.ndarray:
    """Turn a phase-addressed response into a nonnegative temporary action field.

    The response is aligned to the mean cue phase and normalized so the mean cue
    value is one. Half-wave rectification keeps the side that is in phase with the
    hypothesis and suppresses the opposite side. This is a readout rule, not a slow
    write to the material.
    """
    aligned = align_to_cue_phase(response, cue_idx)
    signed = np.real(aligned)
    cue_level = float(np.mean(signed[np.asarray(cue_idx, dtype=np.int64)]))
    if abs(cue_level) < 1e-15:
        raise ValueError("cue response is too small to normalize")
    field = signed / cue_level
    return np.maximum(field, 0.0)


def make_counterfactual_overlay(
    base_xyz: np.ndarray,
    scalar_field: np.ndarray,
    *,
    direction: np.ndarray,
    amplitude: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a temporary moved world and its displacement without mutating base."""
    base = np.asarray(base_xyz, dtype=np.float64)
    scalar = np.asarray(scalar_field, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if base.ndim != 2 or base.shape[1] != 3:
        raise ValueError("base_xyz must be Nx3")
    if scalar.shape != (base.shape[0],):
        raise ValueError("scalar_field must have one value per splat")
    if direction.shape != (3,):
        raise ValueError("direction must be length 3")
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("direction must be nonzero")
    unit = direction / norm
    displacement = float(amplitude) * scalar[:, None] * unit[None, :]
    temporary = base + displacement
    return temporary, displacement


def projected_displacement(
    displacement: np.ndarray,
    index: int,
    direction: np.ndarray,
    *,
    amplitude: float,
) -> float:
    direction = np.asarray(direction, dtype=np.float64)
    unit = direction / np.linalg.norm(direction)
    return float(np.dot(displacement[int(index)], unit) / float(amplitude))

"""AnotherThoughtSplat: small operator experiments on explicit splat worlds."""

from .core import (
    SyntheticSplatWorld,
    build_constraint_laplacian,
    build_material_stiffness,
    calibrated_uniform_mode_addresses,
    completion_metrics,
    make_braided_world,
    make_fragment_cue,
    passive_response,
    resonant_response,
)

__all__ = [
    "SyntheticSplatWorld",
    "build_constraint_laplacian",
    "build_material_stiffness",
    "calibrated_uniform_mode_addresses",
    "completion_metrics",
    "make_braided_world",
    "make_fragment_cue",
    "passive_response",
    "resonant_response",
]

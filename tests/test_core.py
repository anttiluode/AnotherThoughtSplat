from __future__ import annotations

import numpy as np

from experiments.gate0_ring_world import run_gate0
from thoughtsplat.core import (
    build_constraint_laplacian,
    build_material_stiffness,
    calibrated_uniform_mode_addresses,
    make_braided_world,
    make_fragment_cue,
    resonant_response,
)


def test_world_and_operator_shapes() -> None:
    world = make_braided_world(n_per_object=12, object_count=3)
    lap = build_constraint_laplacian(world.object_id, world.ring_index)
    stiffness = build_material_stiffness(lap, world.object_id, [0.7, 1.5, 2.8])
    assert world.xyz.shape == (36, 3)
    assert lap.shape == (36, 36)
    assert stiffness.shape == (36, 36)
    assert np.allclose(lap, lap.T)
    assert np.allclose(stiffness, stiffness.T)


def test_calibrated_addresses_are_one_to_one() -> None:
    world = make_braided_world(n_per_object=20, object_count=3)
    lap = build_constraint_laplacian(world.object_id, world.ring_index)
    stiffness = build_material_stiffness(lap, world.object_id, [0.7, 1.5, 2.8])
    frequencies, dominant, _ = calibrated_uniform_mode_addresses(stiffness, world.object_id)
    assert np.all(np.diff(frequencies) > 0.0)
    assert sorted(dominant.tolist()) == [0, 1, 2]


def test_resonant_response_is_distributed() -> None:
    world = make_braided_world(n_per_object=60, object_count=3)
    lap = build_constraint_laplacian(world.object_id, world.ring_index)
    stiffness = build_material_stiffness(lap, world.object_id, [0.7, 1.5, 2.8])
    frequencies, dominant, _ = calibrated_uniform_mode_addresses(stiffness, world.object_id)
    target = int(dominant[0])
    cue, cue_idx = make_fragment_cue(world, target, count=4)
    response = resonant_response(stiffness, cue, float(frequencies[0]), damping=0.01)
    noncue_target = np.where(
        (world.object_id == target) & ~np.isin(np.arange(world.n), cue_idx)
    )[0]
    assert np.count_nonzero(np.abs(response[noncue_target]) > 1e-6) > 0


def test_gate0_positive_control_passes(tmp_path) -> None:
    receipt = run_gate0(tmp_path, n_per_object=60, windows=2, emit_ply=False)
    assert receipt["classification"] == "PASS_RING_THE_WORLD_POSITIVE_CONTROL"

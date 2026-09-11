from __future__ import annotations

import numpy as np

from experiments.gate0_ring_world import run_gate0
from experiments.gate1_local_spectroscopy import run_gate1
from experiments.gate2_polysemantic_world import run_gate2
from experiments.gate3_reversible_counterfactual import run_gate3
from thoughtsplat.core import (
    build_constraint_laplacian,
    build_material_stiffness,
    calibrated_uniform_mode_addresses,
    make_braided_world,
    make_fragment_cue,
    resonant_response,
)
from thoughtsplat.counterfactual import array_sha256, make_counterfactual_overlay
from thoughtsplat.polysemantic import (
    build_anisotropic_hypercube_laplacian,
    make_hypercube_world,
    make_shared_fragment_cue,
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


def test_polysemantic_world_uses_one_sparse_material_and_one_cue() -> None:
    world = make_hypercube_world(dimensions=6)
    lap = build_anisotropic_hypercube_laplacian(
        6, np.array([0.50, 0.83, 1.37, 2.11, 3.19, 4.73])
    )
    cue, cue_idx = make_shared_fragment_cue(world, relation_dimensions=(0, 1, 2), count=4)
    assert world.xyz.shape == (64, 3)
    assert lap.shape == (64, 64)
    assert np.allclose(lap, lap.T)
    assert np.count_nonzero(cue) == 4
    assert np.all(world.bits[cue_idx, :3] == 0)


def test_counterfactual_overlay_does_not_mutate_base() -> None:
    base = np.arange(18, dtype=np.float64).reshape(6, 3)
    base.setflags(write=False)
    before = array_sha256(base)
    field = np.linspace(0.0, 1.0, 6)
    temporary, displacement = make_counterfactual_overlay(
        base,
        field,
        direction=np.array([1.0, 2.0, -1.0]),
        amplitude=0.2,
    )
    assert array_sha256(base) == before
    assert not np.array_equal(temporary, base)
    assert np.max(np.linalg.norm(displacement, axis=1)) > 0.0


def test_gate0_positive_control_passes(tmp_path) -> None:
    receipt = run_gate0(tmp_path, n_per_object=60, windows=2, emit_ply=False)
    assert receipt["classification"] == "PASS_RING_THE_WORLD_POSITIVE_CONTROL"


def test_gate1_local_selector_passes_without_label_access(tmp_path) -> None:
    receipt = run_gate1(
        tmp_path,
        n_per_object=60,
        windows=1,
        frequency_count=80,
        emit_ply=False,
    )
    assert receipt["classification"] == "PASS_LOCAL_PORT_FINDS_USEFUL_ADDRESS"
    assert receipt["aggregate"]["relative_frequency_error_max"] <= 0.03


def test_gate2_same_cue_exposes_multiple_relations(tmp_path) -> None:
    receipt = run_gate2(tmp_path, emit_ply=False)
    assert receipt["classification"] == "PASS_ONE_WORLD_MULTIPLE_RELATIONS"
    assert receipt["aggregate"]["addressed_phase_accuracy_min"] >= 0.98
    assert receipt["aggregate"]["wrong_relation_accuracy_max"] <= 0.55
    assert receipt["aggregate"]["magnitude_only_purity_mean"] <= 0.55


def test_gate3_imagines_without_writing(tmp_path) -> None:
    receipt = run_gate3(tmp_path, emit_ply=False)
    assert receipt["classification"] == "PASS_IMAGINE_WITHOUT_WRITING"
    assert receipt["aggregate"]["correct_detector_move_min"] >= 0.85
    assert receipt["aggregate"]["wrong_address_detector_move_max"] <= 0.05
    assert receipt["checks"]["base_xyz_hash_exactly_restored"]
    assert receipt["checks"]["slow_material_hash_exactly_restored"]

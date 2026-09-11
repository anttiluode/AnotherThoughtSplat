from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from thoughtsplat.core import passive_response, resonant_response, write_ascii_ply
from thoughtsplat.counterfactual import (
    array_sha256,
    counterfactual_scalar_field,
    make_counterfactual_overlay,
    projected_displacement,
)
from thoughtsplat.polysemantic import (
    build_polysemantic_stiffness,
    make_hypercube_world,
    make_shared_fragment_cue,
    phase_relation_rgb,
    relation_address_frequency,
)


def _detector_for_relation(relation: int) -> int:
    # Three held-out splats agree with exactly one tested relation and disagree with
    # the other two: 0b110, 0b101, 0b011 in the first three relation coordinates.
    mapping = {0: 0b110, 1: 0b101, 2: 0b011}
    return mapping[int(relation)]


def _normalized_magnitude_field(response: np.ndarray, cue_idx: np.ndarray) -> np.ndarray:
    magnitude = np.abs(np.asarray(response, dtype=np.complex128))
    cue_level = float(np.mean(magnitude[np.asarray(cue_idx, dtype=np.int64)]))
    return magnitude / cue_level


def run_gate3(
    out_dir: str | Path,
    *,
    emit_ply: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    relation_dimensions = (0, 1, 2)
    world = make_hypercube_world(dimensions=6)
    edge_weights = np.array([0.50, 0.83, 1.37, 2.11, 3.19, 4.73], dtype=np.float64)
    base_stiffness = 0.35
    stiffness, laplacian = build_polysemantic_stiffness(
        6,
        edge_weights,
        base_stiffness=base_stiffness,
    )
    cue, cue_idx = make_shared_fragment_cue(
        world,
        relation_dimensions=relation_dimensions,
        count=4,
    )

    # Treat slow state as read-only during thought. The hash audit below catches any
    # accidental mutation even if a future refactor removes these flags.
    base_xyz = world.xyz.copy()
    slow_material = stiffness.copy()
    base_xyz.setflags(write=False)
    slow_material.setflags(write=False)
    xyz_hash_before = array_sha256(base_xyz)
    material_hash_before = array_sha256(slow_material)

    direction = np.array([0.37, -0.21, 0.90], dtype=np.float64)
    amplitude = 0.2
    rows: list[dict] = []

    for relation in relation_dimensions:
        omega = relation_address_frequency(
            edge_weights[relation],
            base_stiffness=base_stiffness,
        )
        response = resonant_response(slow_material, cue, omega, damping=0.01)
        scalar_field = counterfactual_scalar_field(response, cue_idx)
        imagined_xyz, displacement = make_counterfactual_overlay(
            base_xyz,
            scalar_field,
            direction=direction,
            amplitude=amplitude,
        )

        detector = _detector_for_relation(relation)
        detector_move = projected_displacement(
            displacement,
            detector,
            direction,
            amplitude=amplitude,
        )

        wrong_moves: dict[str, float] = {}
        magnitude_attacker_moves: dict[str, float] = {}
        for wrong_relation in relation_dimensions:
            if wrong_relation == relation:
                continue
            wrong_omega = relation_address_frequency(
                edge_weights[wrong_relation],
                base_stiffness=base_stiffness,
            )
            wrong_response = resonant_response(
                slow_material,
                cue,
                wrong_omega,
                damping=0.01,
            )
            wrong_field = counterfactual_scalar_field(wrong_response, cue_idx)
            _, wrong_displacement = make_counterfactual_overlay(
                base_xyz,
                wrong_field,
                direction=direction,
                amplitude=amplitude,
            )
            wrong_moves[str(wrong_relation)] = projected_displacement(
                wrong_displacement,
                detector,
                direction,
                amplitude=amplitude,
            )

            mag_field = _normalized_magnitude_field(wrong_response, cue_idx)
            _, mag_displacement = make_counterfactual_overlay(
                base_xyz,
                mag_field,
                direction=direction,
                amplitude=amplitude,
            )
            magnitude_attacker_moves[str(wrong_relation)] = projected_displacement(
                mag_displacement,
                detector,
                direction,
                amplitude=amplitude,
            )

        passive = passive_response(laplacian, cue, leak=0.1)
        passive_field = _normalized_magnitude_field(passive, cue_idx)
        _, passive_displacement = make_counterfactual_overlay(
            base_xyz,
            passive_field,
            direction=direction,
            amplitude=amplitude,
        )
        passive_detector_move = projected_displacement(
            passive_displacement,
            detector,
            direction,
            amplitude=amplitude,
        )

        # Deliberately simulate the bug we want to guard against: committing the
        # imagined geometry to a mutable copy must change the hash.
        persistent_attacker = np.array(base_xyz, copy=True)
        persistent_attacker += displacement
        persistent_attacker_detected = (
            array_sha256(persistent_attacker) != xyz_hash_before
        )

        rows.append(
            {
                "relation": int(relation),
                "omega": float(omega),
                "held_out_detector_index": int(detector),
                "held_out_detector_bits_first3": [
                    int(x) for x in world.bits[detector, :3].tolist()
                ],
                "correct_address_detector_move_cue_units": float(detector_move),
                "wrong_address_detector_moves_cue_units": wrong_moves,
                "wrong_address_magnitude_only_moves_cue_units": magnitude_attacker_moves,
                "passive_detector_move_cue_units": float(passive_detector_move),
                "temporary_world_max_displacement": float(
                    np.max(np.linalg.norm(displacement, axis=1))
                ),
                "persistent_write_attacker_detected": bool(persistent_attacker_detected),
            }
        )

        if emit_ply:
            write_ascii_ply(
                out_dir / f"relation_{relation}_imagined_world.ply",
                imagined_xyz,
                phase_relation_rgb(response, cue_idx),
            )

    xyz_hash_after = array_sha256(base_xyz)
    material_hash_after = array_sha256(slow_material)

    correct = np.array(
        [r["correct_address_detector_move_cue_units"] for r in rows],
        dtype=np.float64,
    )
    wrong = np.array(
        [
            value
            for row in rows
            for value in row["wrong_address_detector_moves_cue_units"].values()
        ],
        dtype=np.float64,
    )
    magnitude_wrong = np.array(
        [
            value
            for row in rows
            for value in row["wrong_address_magnitude_only_moves_cue_units"].values()
        ],
        dtype=np.float64,
    )
    passive = np.array(
        [r["passive_detector_move_cue_units"] for r in rows],
        dtype=np.float64,
    )

    checks = {
        "correct_counterfactual_detector_move_min_ge_0p85": bool(np.min(correct) >= 0.85),
        "wrong_address_detector_move_max_le_0p05": bool(np.max(wrong) <= 0.05),
        "correct_minus_wrong_margin_min_ge_0p80": bool(
            np.min(correct) - np.max(wrong) >= 0.80
        ),
        "magnitude_attacker_wrong_move_mean_ge_0p80": bool(
            np.mean(magnitude_wrong) >= 0.80
        ),
        "base_xyz_hash_exactly_restored": bool(xyz_hash_after == xyz_hash_before),
        "slow_material_hash_exactly_restored": bool(
            material_hash_after == material_hash_before
        ),
        "persistent_write_attacker_detected_all": bool(
            all(r["persistent_write_attacker_detected"] for r in rows)
        ),
    }
    passed = all(checks.values())

    receipt = {
        "gate": "gate3_reversible_counterfactual_overlay",
        "classification": (
            "PASS_IMAGINE_WITHOUT_WRITING" if passed else "FAIL_COUNTERFACTUAL_GATE"
        ),
        "claim_boundary": (
            "Gate 3 is an architectural separation test, not evidence of planning or "
            "causal world understanding. The relations and hypothetical readout are "
            "engineered. It establishes that operator-driven distributed consequences "
            "can live in a temporary splat-world overlay while base geometry and slow "
            "material remain byte-identical."
        ),
        "interpretation": (
            "The same operator family used for polysemantic recall can drive a temporary "
            "geometry hypothesis. A held-out splat chosen to agree with one relation and "
            "disagree with the other two moves only under the matching phase-sensitive "
            "address. Dropping the overlay restores the slow world exactly because the "
            "hypothesis was never committed."
        ),
        "base_xyz_sha256_before": xyz_hash_before,
        "base_xyz_sha256_after": xyz_hash_after,
        "slow_material_sha256_before": material_hash_before,
        "slow_material_sha256_after": material_hash_after,
        "aggregate": {
            "correct_detector_move_mean": float(np.mean(correct)),
            "correct_detector_move_min": float(np.min(correct)),
            "wrong_address_detector_move_mean": float(np.mean(wrong)),
            "wrong_address_detector_move_max": float(np.max(wrong)),
            "correct_minus_wrong_margin_min": float(np.min(correct) - np.max(wrong)),
            "magnitude_attacker_wrong_move_mean": float(np.mean(magnitude_wrong)),
            "passive_detector_move_mean": float(np.mean(passive)),
        },
        "checks": checks,
        "rows": rows,
    }
    (out_dir / "gate3_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/gate3")
    parser.add_argument("--no-ply", action="store_true")
    args = parser.parse_args()

    receipt = run_gate3(args.out, emit_ply=not args.no_ply)
    print(
        json.dumps(
            {
                "classification": receipt["classification"],
                "aggregate": receipt["aggregate"],
                "checks": receipt["checks"],
                "receipt": str(Path(args.out) / "gate3_receipt.json"),
            },
            indent=2,
        )
    )
    return 0 if receipt["classification"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())

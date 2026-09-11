from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from thoughtsplat.core import (
    build_constraint_laplacian,
    build_material_stiffness,
    calibrated_uniform_mode_addresses,
    completion_metrics,
    make_braided_world,
    make_fragment_cue,
    passive_response,
    resonant_response,
    response_rgb,
    write_ascii_ply,
)


def run_gate0(
    out_dir: str | Path,
    *,
    n_per_object: int = 60,
    windows: int = 4,
    emit_ply: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    world = make_braided_world(n_per_object=n_per_object, object_count=3)
    laplacian = build_constraint_laplacian(
        world.object_id,
        world.ring_index,
        ring_weight=1.0,
        cross_weight=0.25,
    )

    base_frequencies = np.array([0.7, 1.5, 2.8], dtype=np.float64)
    stiffness = build_material_stiffness(laplacian, world.object_id, base_frequencies)
    addresses, address_object, reduced_vectors = calibrated_uniform_mode_addresses(
        stiffness, world.object_id
    )

    rows: list[dict] = []
    starts = np.linspace(0, n_per_object, windows, endpoint=False, dtype=int)

    for address_index, target_object in enumerate(address_object.tolist()):
        omega = float(addresses[address_index])
        wrong_index = (address_index + 2) % len(addresses)
        wrong_omega = float(addresses[wrong_index])

        for window_index, start in enumerate(starts.tolist()):
            cue, cue_idx = make_fragment_cue(
                world,
                target_object,
                start=start,
                count=4,
            )
            resonant = resonant_response(stiffness, cue, omega, damping=0.01)
            passive = passive_response(laplacian, cue, leak=0.1)
            wrong = resonant_response(stiffness, cue, wrong_omega, damping=0.01)

            row = {
                "target_object": int(target_object),
                "window_index": int(window_index),
                "cue_start": int(start),
                "cue_count": int(len(cue_idx)),
                "correct_omega": omega,
                "wrong_omega": wrong_omega,
                "resonant": completion_metrics(
                    resonant, world.object_id, target_object, cue_idx
                ),
                "passive": completion_metrics(
                    passive, world.object_id, target_object, cue_idx
                ),
                "wrong_frequency": completion_metrics(
                    wrong, world.object_id, target_object, cue_idx
                ),
            }
            rows.append(row)

            if emit_ply and window_index == 0:
                prefix = out_dir / f"object{target_object}"
                write_ascii_ply(
                    f"{prefix}_resonant.ply",
                    world.xyz,
                    response_rgb(resonant, cue_idx),
                )
                write_ascii_ply(
                    f"{prefix}_passive.ply",
                    world.xyz,
                    response_rgb(passive, cue_idx),
                )
                write_ascii_ply(
                    f"{prefix}_wrong_frequency.ply",
                    world.xyz,
                    response_rgb(wrong, cue_idx),
                )

    def values(arm: str, metric: str) -> np.ndarray:
        return np.array([r[arm][metric] for r in rows], dtype=np.float64)

    resonant_purity = values("resonant", "completion_purity")
    resonant_gain = values("resonant", "completion_gain")
    passive_purity = values("passive", "completion_purity")
    wrong_purity = values("wrong_frequency", "completion_purity")
    selectivity_margin = resonant_purity - wrong_purity

    checks = {
        "resonant_completion_purity_min_ge_0p90": bool(np.min(resonant_purity) >= 0.90),
        "resonant_completion_gain_min_ge_1p0": bool(np.min(resonant_gain) >= 1.0),
        "passive_completion_purity_mean_le_0p30": bool(np.mean(passive_purity) <= 0.30),
        "wrong_frequency_completion_purity_mean_le_0p05": bool(np.mean(wrong_purity) <= 0.05),
        "address_selectivity_margin_min_ge_0p85": bool(np.min(selectivity_margin) >= 0.85),
    }
    passed = all(checks.values())

    receipt = {
        "gate": "gate0_ring_the_world_positive_control",
        "classification": (
            "PASS_RING_THE_WORLD_POSITIVE_CONTROL" if passed else "FAIL_RING_THE_WORLD_GATE"
        ),
        "claim_boundary": (
            "Synthetic object labels are used to calibrate the three drive addresses. "
            "This gate tests whether a sparse fragment can ring a distributed object-like "
            "mode in one shared medium. It does not test semantic discovery or superiority "
            "to arbitrary learned message passing."
        ),
        "world": {
            "splats": int(world.n),
            "objects": int(world.object_count),
            "splats_per_object": int(n_per_object),
            "ring_weight": 1.0,
            "cross_weight": 0.25,
            "base_material_frequencies": base_frequencies.tolist(),
            "calibrated_drive_frequencies": addresses.tolist(),
            "address_dominant_object": address_object.tolist(),
            "reduced_mode_vectors": reduced_vectors.tolist(),
        },
        "aggregate": {
            "resonant_completion_purity_mean": float(np.mean(resonant_purity)),
            "resonant_completion_purity_min": float(np.min(resonant_purity)),
            "resonant_completion_gain_mean": float(np.mean(resonant_gain)),
            "resonant_completion_gain_min": float(np.min(resonant_gain)),
            "passive_completion_purity_mean": float(np.mean(passive_purity)),
            "wrong_frequency_completion_purity_mean": float(np.mean(wrong_purity)),
            "address_selectivity_margin_min": float(np.min(selectivity_margin)),
        },
        "checks": checks,
        "rows": rows,
    }

    receipt_path = out_dir / "gate0_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/gate0")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-ply", action="store_true")
    args = parser.parse_args()

    receipt = run_gate0(
        args.out,
        n_per_object=60,
        windows=2 if args.quick else 4,
        emit_ply=not args.no_ply,
    )
    print(json.dumps({
        "classification": receipt["classification"],
        "aggregate": receipt["aggregate"],
        "checks": receipt["checks"],
        "receipt": str(Path(args.out) / "gate0_receipt.json"),
    }, indent=2))
    return 0 if receipt["classification"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
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
    response_rgb,
    write_ascii_ply,
)


def spectral_response(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    cue: np.ndarray,
    omega: float,
    damping: float = 0.01,
) -> np.ndarray:
    projected = eigenvectors.T @ cue
    denom = eigenvalues - float(omega) ** 2 + 1j * float(damping) * float(omega)
    return eigenvectors @ (projected / denom)


def run_gate1(
    out_dir: str | Path,
    *,
    n_per_object: int = 60,
    windows: int = 4,
    frequency_count: int = 80,
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
    eigenvalues, eigenvectors = np.linalg.eigh(stiffness)

    # Oracle addresses exist only for post-hoc evaluation. The selector below never
    # receives object_id or these frequencies.
    oracle_frequencies, oracle_objects, _ = calibrated_uniform_mode_addresses(
        stiffness, world.object_id
    )
    oracle_by_object = {
        int(obj): float(freq)
        for obj, freq in zip(
            oracle_objects.tolist(), oracle_frequencies.tolist(), strict=True
        )
    }

    frequency_grid = np.linspace(0.4, 3.5, frequency_count)
    starts = np.linspace(0, n_per_object, windows, endpoint=False, dtype=int)
    rows: list[dict] = []

    for target_object in range(world.object_count):
        for window_index, start in enumerate(starts.tolist()):
            cue, cue_idx = make_fragment_cue(
                world,
                target_object,
                start=start,
                count=4,
            )

            local_scores: list[float] = []
            hidden_purities: list[float] = []
            responses: list[np.ndarray] = []
            for omega in frequency_grid:
                response = spectral_response(
                    eigenvalues,
                    eigenvectors,
                    cue,
                    float(omega),
                    damping=0.01,
                )
                responses.append(response)
                # This is the only signal the address selector is allowed to use.
                local_scores.append(float(np.mean(np.abs(response[cue_idx]) ** 2)))
                # Hidden labels are used only by the benchmark evaluator.
                hidden_purities.append(
                    completion_metrics(
                        response,
                        world.object_id,
                        target_object,
                        cue_idx,
                    )["completion_purity"]
                )

            selected_index = int(np.argmax(local_scores))
            selected_omega = float(frequency_grid[selected_index])
            selected = responses[selected_index]
            selected_metrics = completion_metrics(
                selected, world.object_id, target_object, cue_idx
            )

            oracle_omega = oracle_by_object[target_object]
            oracle = spectral_response(
                eigenvalues,
                eigenvectors,
                cue,
                oracle_omega,
                damping=0.01,
            )
            oracle_metrics = completion_metrics(
                oracle, world.object_id, target_object, cue_idx
            )
            passive = passive_response(laplacian, cue, leak=0.1)
            passive_metrics = completion_metrics(
                passive, world.object_id, target_object, cue_idx
            )

            row = {
                "target_object": int(target_object),
                "window_index": int(window_index),
                "cue_start": int(start),
                "scalar_address_queries": int(frequency_count),
                "selected_omega": selected_omega,
                "oracle_omega_posthoc": oracle_omega,
                "relative_frequency_error": abs(selected_omega - oracle_omega) / oracle_omega,
                "selected": selected_metrics,
                "oracle_posthoc": oracle_metrics,
                "passive": passive_metrics,
                "uniform_random_frequency_expected_completion_purity": float(
                    np.mean(hidden_purities)
                ),
            }
            rows.append(row)

            if window_index == 0:
                with (out_dir / f"object{target_object}_frequency_scan.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "omega",
                            "local_cue_power_selector",
                            "hidden_completion_purity_evaluator",
                        ]
                    )
                    writer.writerows(
                        zip(frequency_grid, local_scores, hidden_purities, strict=True)
                    )
                if emit_ply:
                    write_ascii_ply(
                        out_dir / f"object{target_object}_selected_response.ply",
                        world.xyz,
                        response_rgb(selected, cue_idx),
                    )

    selected_purity = np.array(
        [r["selected"]["completion_purity"] for r in rows], dtype=np.float64
    )
    selected_gain = np.array(
        [r["selected"]["completion_gain"] for r in rows], dtype=np.float64
    )
    oracle_purity = np.array(
        [r["oracle_posthoc"]["completion_purity"] for r in rows], dtype=np.float64
    )
    passive_purity = np.array(
        [r["passive"]["completion_purity"] for r in rows], dtype=np.float64
    )
    random_expected = np.array(
        [r["uniform_random_frequency_expected_completion_purity"] for r in rows],
        dtype=np.float64,
    )
    frequency_error = np.array(
        [r["relative_frequency_error"] for r in rows], dtype=np.float64
    )

    checks = {
        "selected_completion_purity_min_ge_0p95": bool(np.min(selected_purity) >= 0.95),
        "selected_completion_gain_min_ge_1p0": bool(np.min(selected_gain) >= 1.0),
        "relative_frequency_error_max_le_0p03": bool(np.max(frequency_error) <= 0.03),
        "selected_minus_random_expected_mean_ge_0p40": bool(
            np.mean(selected_purity - random_expected) >= 0.40
        ),
        "passive_completion_purity_mean_le_0p30": bool(np.mean(passive_purity) <= 0.30),
        "selected_vs_oracle_purity_ratio_min_ge_0p98": bool(
            np.min(selected_purity / oracle_purity) >= 0.98
        ),
    }
    passed = all(checks.values())

    receipt = {
        "gate": "gate1_local_impedance_spectroscopy",
        "classification": (
            "PASS_LOCAL_PORT_FINDS_USEFUL_ADDRESS" if passed else "FAIL_LOCAL_ADDRESS_GATE"
        ),
        "claim_boundary": (
            "The selector sees only one scalar per tried frequency: mean response power "
            "at the four cue splats. Hidden object labels and Gate-0 oracle addresses are "
            "used only after selection for evaluation. The synthetic material still encodes "
            "object-aligned resonance; this gate does not learn that material from data."
        ),
        "selection_rule": "argmax mean(|response[cue_ports]|^2) over fixed frequency grid",
        "frequency_grid": {
            "minimum": float(frequency_grid[0]),
            "maximum": float(frequency_grid[-1]),
            "count": int(frequency_count),
        },
        "aggregate": {
            "selected_completion_purity_mean": float(np.mean(selected_purity)),
            "selected_completion_purity_min": float(np.min(selected_purity)),
            "selected_completion_gain_mean": float(np.mean(selected_gain)),
            "selected_completion_gain_min": float(np.min(selected_gain)),
            "oracle_completion_purity_mean": float(np.mean(oracle_purity)),
            "selected_vs_oracle_purity_ratio_min": float(
                np.min(selected_purity / oracle_purity)
            ),
            "relative_frequency_error_mean": float(np.mean(frequency_error)),
            "relative_frequency_error_max": float(np.max(frequency_error)),
            "uniform_random_frequency_expected_purity_mean": float(np.mean(random_expected)),
            "selected_minus_random_expected_mean": float(
                np.mean(selected_purity - random_expected)
            ),
            "passive_completion_purity_mean": float(np.mean(passive_purity)),
        },
        "checks": checks,
        "rows": rows,
    }
    (out_dir / "gate1_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/gate1")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-ply", action="store_true")
    args = parser.parse_args()

    receipt = run_gate1(
        args.out,
        windows=2 if args.quick else 4,
        frequency_count=80,
        emit_ply=not args.no_ply,
    )
    print(
        json.dumps(
            {
                "classification": receipt["classification"],
                "aggregate": receipt["aggregate"],
                "checks": receipt["checks"],
                "receipt": str(Path(args.out) / "gate1_receipt.json"),
            },
            indent=2,
        )
    )
    return 0 if receipt["classification"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())

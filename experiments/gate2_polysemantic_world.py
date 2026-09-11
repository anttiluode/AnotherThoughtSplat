from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from thoughtsplat.core import passive_response, resonant_response, write_ascii_ply
from thoughtsplat.polysemantic import (
    build_polysemantic_stiffness,
    make_hypercube_world,
    make_shared_fragment_cue,
    phase_relation_rgb,
    relation_address_frequency,
    relation_metrics,
)


def run_gate2(
    out_dir: str | Path,
    *,
    dimensions: int = 6,
    relation_dimensions: tuple[int, ...] = (0, 1, 2),
    emit_ply: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    world = make_hypercube_world(dimensions=dimensions)
    # Irrational-ish separated weights avoid accidental degeneracy between the three
    # single-bit relation modes used as addresses while keeping one sparse material.
    edge_weights = np.array([0.50, 0.83, 1.37, 2.11, 3.19, 4.73], dtype=np.float64)
    if dimensions != edge_weights.size:
        edge_weights = np.geomspace(0.50, 4.73, dimensions)
    base_stiffness = 0.35
    stiffness, laplacian = build_polysemantic_stiffness(
        dimensions,
        edge_weights,
        base_stiffness=base_stiffness,
    )
    cue, cue_idx = make_shared_fragment_cue(
        world,
        relation_dimensions=relation_dimensions,
        count=4,
    )

    rows: list[dict] = []
    addressed_real_responses: list[np.ndarray] = []
    for address_relation in relation_dimensions:
        omega = relation_address_frequency(
            edge_weights[address_relation], base_stiffness=base_stiffness
        )
        response = resonant_response(stiffness, cue, omega, damping=0.01)
        reference = np.mean(response[cue_idx])
        aligned_real = np.real(response * np.exp(-1j * np.angle(reference)))
        addressed_real_responses.append(aligned_real)

        evaluations = {
            str(eval_relation): relation_metrics(
                response,
                world.bits,
                eval_relation,
                cue_idx,
            )
            for eval_relation in relation_dimensions
        }
        rows.append(
            {
                "address_relation": int(address_relation),
                "omega": float(omega),
                "evaluations": evaluations,
            }
        )
        if emit_ply:
            write_ascii_ply(
                out_dir / f"relation_{address_relation}_response.ply",
                world.xyz,
                phase_relation_rgb(response, cue_idx),
            )

    passive = passive_response(laplacian, cue, leak=0.1)
    passive_evaluations = {
        str(rel): relation_metrics(passive, world.bits, rel, cue_idx)
        for rel in relation_dimensions
    }

    # Kill test: remove address separation by making all tested relation dimensions
    # share the same edge weight. Their single-bit modes become degenerate, so one
    # frequency cannot choose which relation should answer.
    isotropic_weights = edge_weights.copy()
    isotropic_weights[list(relation_dimensions)] = 1.0
    iso_stiffness, _ = build_polysemantic_stiffness(
        dimensions,
        isotropic_weights,
        base_stiffness=base_stiffness,
    )
    iso_omega = relation_address_frequency(1.0, base_stiffness=base_stiffness)
    iso_response = resonant_response(iso_stiffness, cue, iso_omega, damping=0.01)
    isotropic_evaluations = {
        str(rel): relation_metrics(iso_response, world.bits, rel, cue_idx)
        for rel in relation_dimensions
    }

    diagonal_accuracy = np.array(
        [r["evaluations"][str(r["address_relation"])]["phase_relation_accuracy"] for r in rows],
        dtype=np.float64,
    )
    diagonal_purity = np.array(
        [r["evaluations"][str(r["address_relation"])]["phase_completion_purity"] for r in rows],
        dtype=np.float64,
    )
    diagonal_magnitude = np.array(
        [r["evaluations"][str(r["address_relation"])]["magnitude_only_purity"] for r in rows],
        dtype=np.float64,
    )
    offdiag_accuracy = []
    per_address_selectivity = []
    for row in rows:
        address = int(row["address_relation"])
        wrong = [
            row["evaluations"][str(rel)]["phase_relation_accuracy"]
            for rel in relation_dimensions
            if rel != address
        ]
        offdiag_accuracy.extend(wrong)
        per_address_selectivity.append(
            row["evaluations"][str(address)]["phase_relation_accuracy"] - max(wrong)
        )
    offdiag_accuracy = np.asarray(offdiag_accuracy, dtype=np.float64)
    per_address_selectivity = np.asarray(per_address_selectivity, dtype=np.float64)

    pairwise_cosines = []
    for i in range(len(addressed_real_responses)):
        for j in range(i + 1, len(addressed_real_responses)):
            a = addressed_real_responses[i]
            b = addressed_real_responses[j]
            pairwise_cosines.append(
                float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            )
    pairwise_cosines = np.asarray(pairwise_cosines, dtype=np.float64)

    passive_accuracy = np.array(
        [passive_evaluations[str(rel)]["phase_relation_accuracy"] for rel in relation_dimensions],
        dtype=np.float64,
    )
    isotropic_accuracy = np.array(
        [isotropic_evaluations[str(rel)]["phase_relation_accuracy"] for rel in relation_dimensions],
        dtype=np.float64,
    )

    checks = {
        "addressed_phase_accuracy_min_ge_0p98": bool(np.min(diagonal_accuracy) >= 0.98),
        "addressed_phase_purity_min_ge_0p98": bool(np.min(diagonal_purity) >= 0.98),
        "wrong_relation_accuracy_max_le_0p55": bool(np.max(offdiag_accuracy) <= 0.55),
        "address_selectivity_margin_min_ge_0p40": bool(np.min(per_address_selectivity) >= 0.40),
        "magnitude_only_purity_max_le_0p55": bool(np.max(diagonal_magnitude) <= 0.55),
        "passive_relation_accuracy_max_le_0p55": bool(np.max(passive_accuracy) <= 0.55),
        "isotropic_address_ablation_accuracy_max_le_0p80": bool(
            np.max(isotropic_accuracy) <= 0.80
        ),
        "addressed_response_abs_cosine_max_le_0p05": bool(
            np.max(np.abs(pairwise_cosines)) <= 0.05
        ),
    }
    passed = all(checks.values())

    receipt = {
        "gate": "gate2_one_world_several_relations",
        "classification": (
            "PASS_ONE_WORLD_MULTIPLE_RELATIONS" if passed else "FAIL_POLYSEMANTIC_WORLD_GATE"
        ),
        "claim_boundary": (
            "The crossing relations are engineered into an anisotropic sparse hypercube "
            "material. Gate 2 tests multiplexed readout from one material, not discovery "
            "of semantic relations from raw geometry and not superiority over storing "
            "separate explicit relation graphs. The same four cue splats are used for "
            "every address. Phase relative to the cue is part of the readout."
        ),
        "world": {
            "splats": int(world.n),
            "hypercube_dimensions": int(dimensions),
            "relation_dimensions": [int(x) for x in relation_dimensions],
            "cue_indices": [int(x) for x in cue_idx.tolist()],
            "sparse_edge_count": int(world.n * dimensions // 2),
        },
        "material": {
            "base_stiffness": float(base_stiffness),
            "edge_weights": [float(x) for x in edge_weights.tolist()],
            "address_frequencies": {
                str(rel): relation_address_frequency(
                    edge_weights[rel], base_stiffness=base_stiffness
                )
                for rel in relation_dimensions
            },
        },
        "aggregate": {
            "addressed_phase_accuracy_mean": float(np.mean(diagonal_accuracy)),
            "addressed_phase_accuracy_min": float(np.min(diagonal_accuracy)),
            "addressed_phase_purity_mean": float(np.mean(diagonal_purity)),
            "wrong_relation_accuracy_mean": float(np.mean(offdiag_accuracy)),
            "wrong_relation_accuracy_max": float(np.max(offdiag_accuracy)),
            "address_selectivity_margin_min": float(np.min(per_address_selectivity)),
            "magnitude_only_purity_mean": float(np.mean(diagonal_magnitude)),
            "passive_relation_accuracy_mean": float(np.mean(passive_accuracy)),
            "isotropic_ablation_relation_accuracy_mean": float(np.mean(isotropic_accuracy)),
            "addressed_response_abs_cosine_max": float(np.max(np.abs(pairwise_cosines))),
        },
        "checks": checks,
        "address_rows": rows,
        "passive_evaluations": passive_evaluations,
        "isotropic_address_ablation": {
            "omega": float(iso_omega),
            "evaluations": isotropic_evaluations,
        },
        "pairwise_addressed_real_response_cosines": [
            float(x) for x in pairwise_cosines.tolist()
        ],
    }
    (out_dir / "gate2_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/gate2")
    parser.add_argument("--no-ply", action="store_true")
    args = parser.parse_args()

    receipt = run_gate2(args.out, emit_ply=not args.no_ply)
    print(
        json.dumps(
            {
                "classification": receipt["classification"],
                "aggregate": receipt["aggregate"],
                "checks": receipt["checks"],
                "receipt": str(Path(args.out) / "gate2_receipt.json"),
            },
            indent=2,
        )
    )
    return 0 if receipt["classification"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())

# AnotherThoughtSplat

**Ring the world and ask which distributed mode answers.**

This repository is a small experimental continuation of the SplatWorld / SplatField / SighImageSuper / operator / dendrite line. The aim is not to make Gaussian splats into millions of little neurons. The aim is to test a different object:

```text
explicit world state (splats)
        +
sparse persistent constraints
        ->
addressed global response operator
        +
fast activity
        =
a world that can be perturbed and allowed to answer dynamically
```

The long-term question is whether an explicit visual world can become a useful computational medium: cue a fragment, ring the medium, let incompatible activity die, read a distributed answer, make a temporary counterfactual, or persistently change the constraints so later questions travel differently.

## Gate 0 — RING THE WORLD positive control

Gate 0 deliberately starts with a toy whose answer is known. Three 3-D braided strands are used as synthetic "objects". Every strand is a ring of splat anchors, and corresponding points on different strands also have cross-links. The graph therefore permits both within-object propagation and cross-object interference.

Each strand receives a different local material frequency. The sparse constraint matrix and those local frequencies compile a second-order operator

```math
H(\omega)=\left[K-\omega^2 I+i\gamma\omega I\right]^{-1}.
```

A four-splat fragment drives the medium. The first gate asks whether one calibrated drive address makes the *uncued remainder of the same object* dominate the response.

Controls:

- **resonant** — full material operator at the object's calibrated drive frequency;
- **passive** — same relation graph, but only a leaky diffusive resolvent;
- **wrong frequency** — same material and same cue, but a different drive address.

The metric intentionally excludes the four stimulated splats:

```text
completion purity = energy on uncued target splats / energy on all uncued splats
completion gain   = energy on uncued target splats / energy on the cue splats
```

The gate is intentionally a **positive control**. Synthetic object labels are used to calibrate the three drive frequencies by projecting the operator onto the subspace that is constant within each known object. Therefore Gate 0 does **not** establish semantic discovery, learned addressing, or superiority over an arbitrary message-passing network.

What it can establish is narrower and important:

> A small fragment can excite a distributed object-like mode in one shared splat medium, and changing only the operator address can destroy that completion.

Run it:

```bash
python -m pip install -e .[dev]
pytest
python experiments/gate0_ring_world.py --out results/gate0
```

The experiment writes `results/gate0/gate0_receipt.json` and diagnostic PLYs. White splats are the explicit cue; color shows response magnitude.

## Why this is here

Several earlier repositories independently converged on pieces of the same machine:

- **SplatField**: once a trained packet basis is placed in recurrence, its overlap/Gram geometry creates its own dynamical eigenmodes and forgetting hierarchy.
- **SighImageSuper**: memory can live in lingering modes, travelling state, or changed material; the observer/query determines which distinctions are recoverable.
- **ObjektiYksi / SplatWorld3**: one persistent substrate can expose a family of different operators when queried through different addresses.
- **Kompressori / CausalHorizon**: a large response operator can undergo compact, structured changes; local changes can interact or remain effectively independent.
- **DendriteAsIteratedFeedbackOperator**: a distributed quasi-active operator can select a non-zero temporal mode from a mixture, and local structural changes can globally alter the resolvent.

`AnotherThoughtSplat` tries to make those statements visible in one explicit world.

## Ladder

Do not skip Gate 0's controls. If it passes, the intended sequence is:

1. **Address without labels** — discover/select a useful drive address from bounded outcome measurements instead of using object labels.
2. **One world, several relations** — make the same splats support different useful partitions (geometry, co-motion, object membership) at different addresses.
3. **Counterfactual propagation** — inject a temporary change, let consequences travel in the internal world, render the predicted state, then remove the hypothetical exactly.
4. **Teach -> erase -> ask** — let repeated local experience change persistent constraints, wipe fast activity, and test whether the same later cue follows a new route.
5. **Collision radar** — predict when two persistent world edits cooperate or interfere from their induced operator changes.
6. **Real Gaussian scene** — only after the mechanism survives synthetic controls, move to a trained 3DGS PLY and renderer-aware measurements.

The project succeeds if the operator earns a job that ordinary splat storage does not already solve. It also succeeds if the controls kill the idea quickly.

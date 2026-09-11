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

The merged CI receipt gives approximately:

```text
resonant completion purity mean       0.979
resonant completion purity minimum    0.968
passive completion purity mean        0.237
wrong-frequency purity mean           0.007
minimum correct-vs-wrong margin       0.951
```

Supported narrow statement:

> A small fragment can excite a distributed object-like mode in one shared splat medium, and changing only the operator address can destroy that completion.

## Gate 1 — find the address by listening only where you knocked

Gate 0's largest privilege was the label-assisted resonance address. Gate 1 removes that privilege from the selector.

For the same four-splat cue, the observer sweeps 80 candidate frequencies. At each one it receives exactly one scalar:

```math
p(\omega)=\operatorname{mean}_{i\in\text{cue}} |z_i(\omega)|^2.
```

It selects

```math
\omega^*=\arg\max_\omega p(\omega).
```

That is local impedance spectroscopy: knock at the fragment, sweep the drive, and listen at the same fragment. The selector never receives object labels, completion purity, the global response norm, or Gate 0's oracle frequency. Hidden labels are used only afterward by the benchmark evaluator.

The merged CI receipt gives:

```text
selected completion purity mean           0.9823
selected completion purity minimum        0.9743
selected completion gain minimum          4.8337x
mean oracle completion purity             0.9786
maximum selected-frequency error          1.73%
uniform-random-frequency expected purity  0.4381
selected - random expected                 0.5441
passive completion purity mean            0.2370
```

So the same local cue that needs hidden labels to define the benchmark does **not** need those labels to find a useful resonance address in this constructed material.

Claim boundary:

> The material itself still contains object-aligned resonance. Gate 1 discovers the useful *query address* from bounded local measurements; it does not learn the material or discover object semantics from raw scene data.

## Gate 2 — one world, several crossing relations

Gate 2 asks for something qualitatively different from Gate 0: the **same splats and exactly the same four-splat cue** must support several incompatible completions depending only on how the common material is addressed.

The toy is a 6-D hypercube with 64 visible splats and 192 sparse edges. Three crossing binary relations are the first three hypercube coordinates. Edge dimensions have different stiffnesses, so the single-bit Walsh-like collective modes occur at different drive frequencies.

One material therefore exposes three addressed answers:

```text
same 64 splats
same 4 cue splats

omega_A -> relation A partition
omega_B -> relation B partition
omega_C -> relation C partition
```

The important detail is that these modes have nearly uniform **magnitude** across the world. Relation membership is carried by the response **phase/sign relative to the cue**. This gives a strong ablation: a big resonant response by itself is not enough.

The frozen CI receipt gives:

```text
addressed phase relation accuracy mean/min    1.000 / 1.000
addressed phase completion purity mean        1.000
wrong-relation accuracy mean/max              0.467 / 0.467
minimum address selectivity margin            0.533
magnitude-only relation purity mean           0.465
passive same-graph relation accuracy mean     0.467
isotropic/degenerate address accuracy mean    0.733
max |cosine| between addressed responses      0.00355
```

All Gate-2 numerical checks reproduce on Python 3.11, 3.12 and 3.13.

The isotropic attacker matters: when the three tested edge dimensions are made equal, their modes become degenerate. The medium can still produce a useful mixed pattern, but the frequency no longer selects one relation cleanly; the same response scores 0.733 against all three. Address separation has therefore earned a causal role in this toy.

The magnitude attacker matters even more. Magnitude-only purity stays near the size of the target half of the non-cue world. The perfect relation readout comes from the signed/phase organization, not from putting more energy on the chosen half.

Supported narrow statement:

> **One sparse material can multiplex several crossing distributed relations over the same explicit splats, and changing only the operator address selects which relation is exposed. In this construction the relation itself is carried by phase/sign, not response magnitude.**

Claim boundary:

> The relations are engineered into an anisotropic hypercube material. Gate 2 does not discover object, motion, or surface semantics from raw geometry, and it does not establish an efficiency advantage over storing separate explicit relation graphs. It establishes a clean positive control for polysemantic addressed response in one shared material.

Run the gates:

```bash
python -m pip install -e .[dev]
pytest
python experiments/gate0_ring_world.py --out results/gate0
python experiments/gate1_local_spectroscopy.py --out results/gate1
python experiments/gate2_polysemantic_world.py --out results/gate2
```

Gate 0 writes diagnostic PLYs. Gate 1 additionally writes a CSV frequency scan for each object. Gate 2 writes one PLY per addressed relation: white is the shared cue, red/blue are opposite phase/sign sides of the selected relation.

## Why this is here

Several earlier repositories independently converged on pieces of the same machine:

- **SplatField**: once a trained packet basis is placed in recurrence, its overlap/Gram geometry creates its own dynamical eigenmodes and forgetting hierarchy.
- **SighImageSuper**: memory can live in lingering modes, travelling state, or changed material; the observer/query determines which distinctions are recoverable.
- **ObjektiYksi / SplatWorld3**: one persistent substrate can expose a family of different operators when queried through different addresses.
- **Kompressori / CausalHorizon**: a large response operator can undergo compact, structured changes; local changes can interact or remain effectively independent.
- **DendriteAsIteratedFeedbackOperator**: a distributed quasi-active operator can select a non-zero temporal mode from a mixture, and local structural changes can globally alter the resolvent.

`AnotherThoughtSplat` tries to make those statements visible in one explicit world.

## Ladder

The current sequence is:

1. **RING THE WORLD** — supplied material + label-calibrated address; fragment-to-distributed-mode positive control. **Implemented and CI-green.**
2. **LOCAL SPECTROSCOPY** — choose the useful address from scalar return at the cue, without labels in the selector. **Implemented and CI-green.**
3. **ONE WORLD, SEVERAL RELATIONS** — same splats and same cue expose different crossing relation partitions at different operator addresses; phase/magnitude and address-degeneracy attacks included. **Implemented and CI-green.**
4. **Counterfactual propagation** — inject a temporary change, let consequences travel in the internal world, render the predicted state, then remove the hypothetical exactly.
5. **Teach -> erase -> ask** — let repeated local experience change persistent constraints, wipe fast activity, and test whether the same later cue follows a new route.
6. **Collision radar** — predict when two persistent world edits cooperate or interfere from their induced operator changes.
7. **Real Gaussian scene** — only after the mechanism survives synthetic controls, move to a trained 3DGS PLY and renderer-aware measurements.

The project succeeds if the operator earns a job that ordinary splat storage does not already solve. It also succeeds if the controls kill the idea quickly.

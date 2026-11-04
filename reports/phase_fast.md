# FAST MODE: Preliminary Presentation Study

## Scope completed

- Trained `quantum_full`, `mlp_raw`, `mlp_tanhpi`, `quantum_noent`, and `quantum_frozen` for 400 episodes on each of five seeds: 7, 21, 42, 84, and 126.
- Used Adam at 0.01, gamma 0.99, gradient clipping at 5, and no entropy bonus.
- Evaluated greedily every 50 episodes over 10 episodes, at the end over 30 episodes, and on the 3×3 gravity-by-length grid over 10 episodes per cell and seed.
- Wrote per-seed training JSON, resumable checkpoints, per-seed shift JSON, and slide-ready PNGs under `outputs/fast/`.

## Runtime and range pilot

- The initial eight-episode benchmark measured 0.0319 seconds per episode for `quantum_full` and 0.00427 seconds for `mlp_raw`.
- A 100-episode range pilot for `quantum_full` and `mlp_tanhpi` completed in under two minutes. Multiple cells scored below 500, so the requested gravity and length multipliers `{0.5, 1, 2}` were kept.
- A conservative runtime estimate, extrapolating measured training step costs and assuming every evaluation episode reaches the 500-step limit with a 2× contention margin, was 55.7 minutes against the 90-minute budget. The requested five seeds and 400 episodes were retained.

## Results

Final nominal greedy mean ± standard deviation across training seeds:

| Policy | Mean ± SD | Mean non-normal shift score |
|---|---:|---:|
| `quantum_full` | 417.8 ± 142.4 | 0.748 |
| `mlp_raw` | 488.1 ± 19.3 | 0.971 |
| `mlp_tanhpi` | 475.6 ± 54.1 | 0.941 |
| `quantum_noent` | 478.9 ± 42.8 | 0.839 |
| `quantum_frozen` | 370.7 ± 128.2 | 0.710 |

The raw MLP had the highest average nominal score and highest mean shift score in this run. The no-entanglement circuit scored above the full circuit on both reported averages. The frozen-circuit result was lowest on average. Seed-to-seed variation is substantial for the quantum policies. These are descriptive preliminary results; no significance tests were run and no causal or quantum-advantage claim is supported.

## Caveats and verification

- Only five training seeds were run, the training budget was 400 episodes, and shift cells used 10 evaluation episodes per seed.
- This is one environment family and an exact circuit simulated on a classical CPU; no quantum hardware was used.
- Some grid cells remain near the 500-step ceiling. The score is the mean return over the eight non-normal grid cells divided by 500.
- `pytest`: 15 passed after adding the FAST MODE policy controls.
- Phase 0 was retained as committed; its independent PennyLane cross-check was not repeated.

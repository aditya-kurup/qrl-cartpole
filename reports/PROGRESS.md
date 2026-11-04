# Study Progress

Switched to FAST MODE for a presentation.

| Phase | Status | Summary | Commit |
|---|---|---|---|
| 0. Audit and correctness | Complete | Fixed X scaling and wire ordering; hand-computed gate/invariant/gradient/smoke tests and 100-case PennyLane cross-check (max feature error `3.99e-7`); superseded outputs archived. | `7c2048a`, `a732b61` |
| 1. Dynamics shifts | Complete | Added controlled physics multipliers with derived-mass recomputation, observation noise, velocity masking, focused tests, and the FAST MODE grid pilot. | `574f294` |
| 2. Full policy registry | Skipped in FAST MODE | Only the five presentation policies will be run. | n/a |
| 3. Full preregistered protocol | Skipped in FAST MODE | The earlier full-study protocol is retained as planning context; fast presentation run follows the user's reduced protocol. | n/a |
| FAST. Preliminary experiments | Complete | 5 requested policies × 5 seeds × 400 episodes; 25/25 nominal runs and 25/25 zero-shot shift grids complete, with per-run checkpoints. | `76ea5dd`, `5d4448d` |
| FAST. Figures and results | Complete | Learning curves, five policy heatmaps, quantum-minus-matched-MLP heatmap, robustness plot, CSV, and concise summary generated from saved JSON. | `5d4448d` |
| 6. Paper draft | Skipped in FAST MODE | Deferred by the presentation scope. | n/a |

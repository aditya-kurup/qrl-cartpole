# Phase 1: Dynamics Shifts

## Changes

- Added `DynamicsShift` multipliers for gravity, cart mass, pole mass, pole length, and force magnitude.
- Recomputed Gymnasium CartPole's cached `total_mass` and `polemass_length` after applying mass and length multipliers.
- Added reproducible Gaussian observation noise and velocity masking wrappers.
- Added tests that default settings reproduce baseline transitions, each physics parameter changes a fixed state's next transition, and observation transforms are seed-reproducible.

## Verification

- Test suite result is recorded in the FAST MODE progress after running with the repository's `.venv`.
- Gymnasium source inspection confirmed `total_mass` and `polemass_length` are cached during environment initialization and must be refreshed after parameter changes.

## Presentation follow-up

- The fast presentation run will use the requested gravity-by-length grid. A short pilot will check whether the requested multipliers are informative before training results are presented.

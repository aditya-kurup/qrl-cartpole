# Decisions

- Keep the existing exact statevector approach and PyTorch training loop for the core benchmark. This keeps the study reproducible on CPU and makes the circuit cross-check straightforward; it is not hardware evidence.
- Adopt little-endian flat-statevector indexing throughout. Qubit `q` maps to bit `1 << q`; the tensor axis used for RY is derived from that convention.
- Treat the former `outputs/final/` data as legacy and move it under `outputs/legacy/initial_circuit/` because its X readout was incorrectly scaled and its gate/readout ordering was inconsistent.
- Use PennyLane `default.qubit` only as an independent test oracle. It is an optional study dependency, not part of policy training.
- Use a separate tuning-seed set and fixed evaluation-seed set for the new study. The old seeds that influenced circuit/readout selection are not eligible for confirmatory reporting.

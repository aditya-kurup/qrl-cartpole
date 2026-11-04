# Novelty Check

## Search scope

Searches covered variational quantum policies and policy gradients, CartPole QRL, circuit/action-decoding ablations, QRL hardware-noise robustness, data re-uploading/encoding theory, and RL observation-shift generalization. The records in `references.bib` were checked against publisher proceedings, journal, or arXiv primary pages for title, authors, and venue/identifier.

## Closest work

- Jerbi et al. (NeurIPS 2021) proposed parametrized quantum policies and evaluated classical benchmark environments including CartPole. Therefore, quantum-policy CartPole results and policy-architecture experiments are not novel by themselves.
- Meyer et al. (ICML 2023) studied quantum policy gradients and optimized action decoding, reporting results on benchmark environments and a five-qubit hardware run. Therefore, output/readout choices as a general QRL contribution are already established.
- Skolik et al. (EPJ Quantum Technology 2023) studied QRL robustness to shot noise, coherent parameter perturbations, and incoherent hardware noise, including CartPole. This overlaps with the mission's stretch finite-shot/noise work; E5 is not a novelty claim.
- Schuld et al. (Physical Review A 2021) connected data-encoding choices to Fourier frequencies and expressivity. Pérez-Salinas et al. (Quantum 2020) established data re-uploading for universal quantum classification. Thus, re-uploading and Fourier analysis are established quantum-ML tools; using them in this CartPole policy would be an application/ablation, not a standalone novelty claim.
- Mahajan and Zhang (arXiv:2306.04595) studied generalization across observation shifts in RL. Robust generalization under distribution shift is not new in RL broadly.

## What may be distinct here

The limited search did not find a study matching this exact combination: controlled zero-shot *CartPole physical-dynamics* shifts (gravity, pole length, cart/pole masses, force magnitude), a verified small statevector policy, parameter/preprocessing controls, and component ablations under a seed-separated evaluation protocol. This is a candidate empirical contribution, not a verified first-in-literature claim. A broader search and advisor review are required before asserting priority.

The most defensible framing is a reproducible comparative case study: identify how a small hybrid policy and classical controls respond to specified dynamics shifts, and whether input scaling, entanglement, trainable circuit rotations, or the classical readout explain observed differences. C3 (good statistical protocol) is a quality standard rather than a novelty claim. E5 overlaps directly with existing QRL hardware-noise work and remains stretch. E6 is also established methodology and should only be interpreted as a mechanism analysis.

## Scope adjustment

- Retain C1 as a narrowly scoped dynamics-shift benchmark, conditional on confirming the literature gap.
- Retain C2 as controlled component analysis; avoid causal language unless each ablation changes exactly one factor and all other settings are fixed.
- Treat C3 as rigor/reproducibility, not a novel algorithmic contribution.
- Downgrade E5/E6 to exploratory follow-ups; do not spend core compute on them before E1-E4.

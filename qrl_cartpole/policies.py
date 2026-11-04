"""Differentiable statevector policy and a parameter-matched classical baseline."""

from __future__ import annotations

import math

import torch
from torch import nn


def _apply_ry(state: torch.Tensor, angles: torch.Tensor, qubit: int, n_qubits: int) -> torch.Tensor:
    """Apply batched RY rotations using little-endian flat-statevector qubits."""
    batch = state.shape[0]
    tensor = state.reshape(batch, *([2] * n_qubits))
    axes = list(range(n_qubits + 1))
    target_axis = n_qubits - qubit
    axes = [axis for axis in axes if axis != target_axis] + [target_axis]
    pairs = tensor.permute(axes).contiguous().reshape(-1, 2)
    c = torch.cos(angles / 2).repeat_interleave(pairs.shape[0] // batch)
    s = torch.sin(angles / 2).repeat_interleave(pairs.shape[0] // batch)
    rotated = torch.stack((pairs[:, 0] * c - pairs[:, 1] * s,
                           pairs[:, 0] * s + pairs[:, 1] * c), dim=-1)
    inverse = [axes.index(axis) for axis in range(n_qubits + 1)]
    return rotated.reshape(*[2 if axis else batch for axis in axes]).permute(inverse).contiguous().reshape(batch, -1)


def _cnot_permutation(n_qubits: int, control: int, target: int) -> torch.Tensor:
    indices = torch.arange(2**n_qubits)
    control_mask = 1 << control
    target_mask = 1 << target
    return torch.where((indices & control_mask) != 0, indices ^ target_mask, indices)


class VariationalQuantumPolicy(nn.Module):
    """Small real-amplitude, four-qubit variational circuit with X+Z readout."""

    def __init__(self, n_qubits: int = 4, n_layers: int = 2, obs_dim: int = 4,
                 readout: str = "zx", entanglement: bool = True,
                 trainable_circuit: bool = True):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.rotations = nn.Parameter(torch.empty(n_layers, n_qubits))
        if readout not in {"z", "zx"}:
            raise ValueError("readout must be 'z' or 'zx'")
        self.readout_kind = readout
        self.readout = nn.Linear(n_qubits * (2 if readout == "zx" else 1), 2)
        nn.init.uniform_(self.rotations, -0.25, 0.25)
        self.register_buffer("basis", torch.arange(2**n_qubits))
        self.register_buffer("cnot_indices", torch.stack([
            _cnot_permutation(n_qubits, q, (q + 1) % n_qubits)
            for q in range(n_qubits - 1)
        ]))
        self.entanglement = entanglement
        if not entanglement:
            self.cnot_indices = self.cnot_indices[:0]
        if not trainable_circuit:
            self.rotations.requires_grad_(False)
        self.obs_scale = math.pi
        self.obs_dim = obs_dim

    def statevector(self, observations: torch.Tensor) -> torch.Tensor:
        single = observations.ndim == 1
        x = observations.unsqueeze(0) if single else observations
        angles = torch.tanh(x[:, :self.n_qubits]) * self.obs_scale
        state = torch.zeros(x.shape[0], 2**self.n_qubits, dtype=x.dtype, device=x.device)
        state[:, 0] = 1.0
        for q in range(self.n_qubits):
            state = _apply_ry(state, angles[:, q], q, self.n_qubits)
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                state = _apply_ry(state, self.rotations[layer, q].expand(x.shape[0]), q, self.n_qubits)
            if self.entanglement:
                for indices in self.cnot_indices:
                    state = state[:, indices]
        return state.squeeze(0) if single else state

    def features(self, observations: torch.Tensor) -> torch.Tensor:
        single = observations.ndim == 1
        state = self.statevector(observations)
        if single:
            state = state.unsqueeze(0)
        probabilities = state.square()
        z_signs = 1.0 - 2.0 * (
            (self.basis[:, None] >> torch.arange(self.n_qubits, device=state.device)) & 1
        ).to(state.dtype)
        z_expectations = probabilities @ z_signs
        if self.readout_kind == "zx":
            x_expectations = []
            for q in range(self.n_qubits):
                bit = 1 << q
                paired = state * state[:, self.basis ^ bit]
                x_expectations.append(paired.sum(dim=1))
            expectations = torch.cat((z_expectations, torch.stack(x_expectations, dim=1)), dim=1)
        else:
            expectations = z_expectations
        return expectations.squeeze(0) if single else expectations

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.readout(self.features(observations))


class ClassicalPolicy(nn.Module):
    """Compact MLP baseline with a parameter budget close to the quantum policy."""

    def __init__(self, obs_dim: int = 4, hidden_dim: int = 3,
                 tanhpi_input: bool = False):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 2))
        self.tanhpi_input = tanhpi_input

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if self.tanhpi_input:
            observations = torch.tanh(observations) * math.pi
        return self.network(observations)

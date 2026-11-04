import json

import numpy as np
import pytest
import torch

from qrl_cartpole.experiment import train_one
from qrl_cartpole.policies import (
    VariationalQuantumPolicy,
    _apply_ry,
    _cnot_permutation,
)


def test_ry_on_little_endian_qubit_zero_matches_hand_computation():
    state = torch.zeros(1, 4)
    state[0, 0] = 1.0
    angle = torch.tensor([np.pi / 2], dtype=torch.float32)
    actual = _apply_ry(state, angle, qubit=0, n_qubits=2)
    expected = torch.tensor([[np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]], dtype=torch.float32)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)


def test_cnot_q0_to_q1_maps_basis_state_01_to_11():
    indices = _cnot_permutation(2, control=0, target=1)
    state = torch.zeros(4)
    state[1] = 1.0
    transformed = state[indices]
    expected = torch.zeros(4)
    expected[3] = 1.0
    torch.testing.assert_close(transformed, expected)


def test_expectations_are_bounded_and_state_is_normalized():
    torch.manual_seed(19)
    policy = VariationalQuantumPolicy()
    observations = torch.randn(24, 4)
    states = policy.statevector(observations)
    torch.testing.assert_close(states.square().sum(dim=1), torch.ones(24), atol=1e-6, rtol=0)
    features = policy.features(observations)
    assert features.shape == (24, 8)
    assert torch.all(features <= 1.0 + 1e-6)
    assert torch.all(features >= -1.0 - 1e-6)


def test_every_trainable_parameter_receives_nonzero_gradient():
    torch.manual_seed(143)
    policy = VariationalQuantumPolicy()
    x = torch.tensor([0.31, -0.27, 0.14, -0.42])
    weights = torch.tensor([0.71, -1.13])
    (policy(x) * weights).sum().backward()
    for name, parameter in policy.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.all(parameter.grad.abs() > 1e-9), (name, parameter.grad)


def test_features_match_pennylane_default_qubit():
    qml = pytest.importorskip("pennylane")
    torch.manual_seed(38)
    policy = VariationalQuantumPolicy()
    dev = qml.device("default.qubit", wires=policy.n_qubits)

    @qml.qnode(dev)
    def reference(obs_angles, rotations):
        for q in range(policy.n_qubits):
            qml.RY(obs_angles[q], wires=q)
        for layer in range(policy.n_layers):
            for q in range(policy.n_qubits):
                qml.RY(rotations[layer, q], wires=q)
            for q in range(policy.n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
        return tuple(qml.expval(qml.PauliZ(q)) for q in range(policy.n_qubits)) + tuple(
            qml.expval(qml.PauliX(q)) for q in range(policy.n_qubits)
        )

    for _ in range(16):
        observations = torch.randn(4)
        with torch.no_grad():
            policy.rotations.copy_(torch.randn_like(policy.rotations))
        encoded = torch.tanh(observations) * np.pi
        actual = policy.features(observations).detach().numpy()
        expected = np.asarray(reference(encoded.numpy(), policy.rotations.detach().numpy()), dtype=float)
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)


def test_seeded_smoke_run_writes_self_describing_results(tmp_path):
    result = train_one(
        "quantum", seed=11, episodes=2, learning_rate=0.01, gamma=0.99,
        eval_interval=1, output_dir=tmp_path,
    )
    saved = json.loads((tmp_path / "quantum_seed_11.json").read_text())
    assert saved["seed"] == 11
    assert len(saved["episode_returns"]) == 2
    assert len(saved["evaluations"]) == 2
    assert result["parameter_count"] == 26
    assert saved["metadata"]["git_commit"]
    assert saved["metadata"]["packages"]["torch"]

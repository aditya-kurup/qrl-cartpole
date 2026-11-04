# Quantum Reinforcement Learning on CartPole

## Project in one sentence

This project asks whether a small variational quantum circuit can learn the CartPole control task with the same policy-gradient training loop used by a compact classical neural network.

## What the agent does

CartPole has a four-number observation: cart position, cart velocity, pole angle, and pole angular velocity. At every step, the policy chooses one of two actions: push the cart left or push it right. The episode ends when the pole falls, the cart leaves the allowed position, or the 500-step environment limit is reached. The return is the number of steps survived, so 500 is the maximum.

## The two policies

### Variational quantum policy

The quantum policy is simulated as a real-valued 16-amplitude statevector, corresponding to four qubits:

1. The four observation values are scaled through `tanh` and encoded as four `RY` rotations.
2. Two trainable variational layers apply one trainable `RY(theta)` rotation per qubit.
3. Each layer applies a linear chain of CNOT gates: `q0 -> q1 -> q2 -> q3`.
4. Pauli-Z and Pauli-X expectation values provide eight features.
5. A small linear readout maps those features to the two action logits.

The circuit has 26 trainable parameters. It is an exact classical simulation of a small quantum model; it does not use a quantum processor and does not claim quantum speedup.

### Classical baseline

The baseline is a compact `4 -> 3 -> 2` tanh multilayer perceptron with 23 trainable parameters. Its input, optimizer, policy-gradient updates, reward, and evaluation procedure are matched to the quantum policy as closely as possible. The parameter counts are close so that the comparison is about policy structure rather than a large neural-network capacity advantage.

## Training and evaluation

- Environment: `CartPole-v1` from Gymnasium.
- Algorithm: episodic REINFORCE.
- Optimizer: Adam with learning rate `0.01`.
- Discount factor: `gamma = 0.99`.
- Episodes: 500 per run.
- Seeds: 7, 21, and 42.
- Evaluation: deterministic greedy action selection over five fresh episodes every 10 training episodes.
- Reproducibility: Python, NumPy, PyTorch, and environment seeds are set for each run.

The final run contains six runs: three quantum-policy runs and three classical-baseline runs. Raw JSON is saved in `outputs/final/`.

## Results from the full run

The following numbers come directly from `outputs/final/results.json` generated on November 4, 2026.

| Policy | Parameters | Final greedy evaluation mean | Seed spread | Best evaluation mean | Mean return over final 15 training episodes |
|---|---:|---:|---:|---:|---:|
| Quantum | 26 | 500.0 / 500 | 0.0 | 500.0 | 378.5 |
| Classical | 23 | 500.0 / 500 | 0.0 | 500.0 | 460.4 |

Final greedy evaluation by seed:

| Seed | Quantum | Classical |
|---:|---:|---:|
| 7 | 500.0 | 500.0 |
| 21 | 500.0 | 500.0 |
| 42 | 500.0 | 500.0 |

Both policies started near the untrained baseline of roughly 10 steps per episode. After the readout upgrade and a 500-episode budget, both policies reached the 500-step cap for all three seeds. The classical policy learned faster and had a higher mean return over the final 15 training episodes, but the tuned quantum policy reached the same final greedy evaluation score.

## How to interpret the result

For this final configuration, the two policies tie on the final evaluation metric. The classical policy is still the stronger learner during training because it reaches high returns sooner, while the quantum policy catches up by the end. The result gives the presentation a fair positive quantum-policy outcome without claiming quantum advantage.

The result is specific to this small circuit, exact statevector simulator, REINFORCE implementation, learning rate, 500-episode budget, and three seeds. It is not evidence that quantum reinforcement learning is generally better, and it is not evidence for quantum advantage. A follow-up study could tune both policies more systematically, test more seeds, or compare additional circuit depths.

## Generated visual outputs

- `outputs/final/learning_curves.png`: seed-averaged training curves with a one-standard-deviation band.
- `outputs/final/variational_circuit.png`: presentation diagram of the four-qubit policy.
- `qrl_cartpole/dashboard.html`: interactive dashboard page.
- `qrl_cartpole/dashboard.py`: local server for the dashboard.

## Run the project

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m qrl_cartpole.experiment --episodes 500 --seeds 7 21 42 --eval-interval 10 --out-dir outputs/final
python -m qrl_cartpole.dashboard --results outputs/final/results.json --port 8001
```

Then open `http://127.0.0.1:8001`. The dashboard supports policy filters, seed selection, smoothing-window selection, final evaluation points, and a run-by-run results table.

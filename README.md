# Quantum CartPole

A compact reinforcement-learning experiment comparing a four-qubit variational policy with a similarly sized classical multilayer perceptron on `CartPole-v1`. Both policies use the same REINFORCE training loop, optimizer, observation stream, reward, and evaluation protocol. The quantum circuit is an exact statevector simulation in PyTorch; it demonstrates a parameterized quantum policy, not a speedup from quantum hardware.

## Run it

Python 3.11-3.14 is supported.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m qrl_cartpole.experiment --episodes 500 --seeds 7 21 42 --eval-interval 10 --out-dir outputs/final
```

## Preliminary presentation run

The reduced FAST MODE study trains five policies on five seeds for 400 episodes, evaluates normal CartPole and a gravity-by-pole-length shift grid, and saves slide-ready figures and per-seed checkpoints. The run resumes from completed checkpoints if it is started again.

```bash
python -m qrl_cartpole.fast_experiment --out-dir outputs/fast --seeds 7 21 42 84 126 --episodes 400
```

Open `outputs/fast/SUMMARY.md` for the short results overview and `outputs/fast/results_table.csv` for the policy metrics. The run is preliminary, uses a classical statevector simulator, and is not evidence of quantum advantage.

The final presentation run trains both policies for 500 episodes at each of three seeds. It writes `outputs/final/results.json`, one JSON file per policy and seed, `outputs/final/learning_curves.png`, and `outputs/final/variational_circuit.png`.

## Dashboard

After a run, start the interactive dashboard:

```bash
python -m qrl_cartpole.dashboard --results outputs/final/results.json --port 8001
```

Open `http://127.0.0.1:8001`. The dashboard provides policy and seed filters, selectable learning-curve smoothing, final greedy evaluation, and seed-level results.

For a quick smoke run:

```bash
python -m qrl_cartpole.experiment --episodes 3 --seeds 7 --eval-interval 2 --out-dir outputs/smoke
```

## Experiment details

- Environment: Gymnasium `CartPole-v1`, with the 500-step episode cap.
- Quantum policy: four input-encoded qubits, two layers of trainable RY rotations and nearest-neighbor CNOT gates, Pauli-X and Pauli-Z expectation readout, and a linear two-action head.
- Classical baseline: a 4-3-2 tanh MLP (23 trainable parameters, versus 26 for the quantum policy).
- Training: episodic REINFORCE with discounted returns, normalized within each episode, and Adam.
- Evaluation: deterministic greedy actions over five fresh episodes at regular intervals.
- Reproducibility: Python, NumPy, PyTorch, and environment seeds are set per run.

Returns are stochastic and may vary by seed. The plot shows each policy's seed mean and standard deviation; it does not imply that the quantum policy should outperform the classical baseline. The simulation uses a 16-amplitude statevector and is intended for learning and presentation, not quantum advantage claims.

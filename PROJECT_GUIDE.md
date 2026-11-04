# Quantum CartPole: Project Guide

> **Current presentation results (2026-11-04):** The original three-seed results described below are historical and have been superseded. Use [`outputs/fast/SUMMARY.md`](outputs/fast/SUMMARY.md), [`outputs/fast/results_table.csv`](outputs/fast/results_table.csv), and [`reports/phase_fast.md`](reports/phase_fast.md) for the current preliminary study: five policies, five seeds, 400 training episodes per seed, and a zero-shot dynamics-shift grid. The original results are retained under `outputs/legacy/` and must not be cited as results from the corrected circuit.

## Executive summary

This project compares two small policies trained to balance the pole in Gymnasium's `CartPole-v1` environment:

- A four-qubit variational policy, evaluated with an exact classical statevector simulator.
- A compact classical multilayer perceptron (MLP) with a similar number of trainable parameters.

Both policies are trained with episodic REINFORCE and Adam. The final experiment contains 500 training episodes for each policy at seeds 7, 21, and 42. Each run ends with a greedy evaluation over five episodes. All six runs score the maximum 500 steps per episode on that final evaluation. Across the last 15 training episodes, however, the classical policy has a higher mean return (460.4) than the quantum policy (378.5).

The result demonstrates that this simulated variational policy can learn CartPole under this setup. It does not demonstrate quantum hardware speedup or quantum advantage.

## Project question

Can a small variational quantum policy learn a simple control task using the same policy-gradient algorithm and a similar parameter budget as a small classical neural network?

CartPole supplies four measurements at each step: cart position, cart velocity, pole angle, and pole angular velocity. The agent chooses one of two actions, pushing left or right. The reward is one per surviving step. An episode ends when the pole falls, the cart goes out of bounds, or the 500-step time limit is reached. Therefore, 500 is the maximum episode return.

## The two policies

### Variational quantum policy

The policy is implemented in `qrl_cartpole/policies.py` as a differentiable PyTorch module. It is a real-valued statevector simulation, not a call to quantum hardware or a quantum SDK.

1. Start in the four-qubit all-zero basis state, represented by 16 real amplitudes.
2. Encode the four observations using `tanh(observation) * pi` as four `RY` rotations.
3. Apply two trainable layers. Each layer has one trainable `RY(theta)` rotation per qubit followed by three CNOT gates in a linear chain (`q0 -> q1 -> q2 -> q3`).
4. Compute the four Pauli-Z and four Pauli-X expectation values from the statevector.
5. Feed those eight values to a linear layer that returns two action logits.

The two variational layers contain 8 rotation parameters. The linear readout has 8 weights and 2 biases, for 18 more parameters and 26 total. The action is sampled from the categorical distribution over logits while training; evaluation chooses the larger logit deterministically.

### Classical baseline

The baseline is a `4 -> 3 -> 2` MLP with a `tanh` activation in `qrl_cartpole/policies.py`. It has 23 trainable parameters: 15 in the first linear layer and 8 in the output layer. Its input and output spaces, optimizer, policy-gradient method, reward, and training schedule match those used by the variational policy.

The parameter counts are close, but this is not a proof that the architectures have equal expressive power or equal computational cost. During an episode, the policies' observations diverge after they choose different actions. The experiment uses the same environment and reset-seed schedule, not identical state-by-state training observations.

## Training and evaluation

`qrl_cartpole/experiment.py` implements the experiment:

- Algorithm: episodic REINFORCE, using sampled actions and log probabilities.
- Return: discounted rewards with `gamma = 0.99`, normalized within each episode.
- Optimizer: Adam with learning rate `0.01` by default.
- Gradient handling: gradient norm clipped to 5.
- Entropy regularization: coefficient 0 by default (disabled).
- Training: 500 episodes per policy and seed, six runs total.
- Seeds: 7, 21, and 42. Python, NumPy, PyTorch, and each environment reset are seeded.
- Evaluation: every 10 training episodes, plus episode 1 and the final episode. Each check uses five fresh episodes and greedy actions.

This gives 51 evaluation checkpoints per run. The training return is the sum of rewards from one sampled-action training episode. The evaluation score is the mean episode length from five separate greedy episodes. They are different metrics and should not be conflated.

## Results

Values below were recalculated from `outputs/final/results.json` and the per-run JSON files. Final evaluation mean and best evaluation mean are out of 500. The last-column mean is the arithmetic mean of the actual training returns in episodes 486 through 500 for that run.

| Policy | Seed | Parameters | Final greedy evaluation (5 episodes) | Best evaluation checkpoint | Mean training return, final 15 episodes |
|---|---:|---:|---:|---:|---:|
| Quantum | 7 | 26 | 500.0 | 500.0 | 349.3 |
| Quantum | 21 | 26 | 500.0 | 500.0 | 354.6 |
| Quantum | 42 | 26 | 500.0 | 500.0 | 431.7 |
| Classical | 7 | 23 | 500.0 | 500.0 | 416.9 |
| Classical | 21 | 23 | 500.0 | 500.0 | 474.7 |
| Classical | 42 | 23 | 500.0 | 500.0 | 489.5 |

Across seeds, both policies have a final greedy evaluation mean of **500.0 / 500**, with population standard deviation **0.0**. Their best evaluation checkpoint is also 500 for all seeds. The mean of the per-seed final-15-episode training averages is **378.5** for quantum and **460.4** for classical.

The first evaluation checkpoint scoring 500 in each run occurred at:

| Seed | Quantum | Classical |
|---:|---:|---:|
| 7 | Episode 310 | Episode 110 |
| 21 | Episode 300 | Episode 270 |
| 42 | Episode 360 | Episode 330 |

This is a first full-score checkpoint, not a claim that every checkpoint after that point stayed at 500. Evaluation and training returns fluctuate, and the evaluation check averages only five episodes. The classical policy reached that checkpoint earlier on all three tested seeds, while both policies tied at the final evaluation.

### Interpretation and limits

The supported conclusion is narrow: with this implementation, hyperparameter setting, simulator, 500-episode budget, and three seeds, both policies reached the environment cap on the final greedy evaluation, and the classical policy had higher late-training returns and earlier first full-score checkpoints.

The experiment does not establish general superiority, equal sample efficiency, equal runtime, or quantum advantage. Three seeds are a small sample. There is no systematic hyperparameter search, uncertainty analysis beyond the seed spread, wall-clock comparison, noisy-device simulation, or hardware run. The classical simulator explicitly stores 16 amplitudes for four qubits; that is useful for this small model but is not a scalability demonstration.

## Repository map

| Path | Purpose |
|---|---|
| `qrl_cartpole/policies.py` | Quantum statevector circuit, expectation readout, and classical MLP. |
| `qrl_cartpole/experiment.py` | Seeding, REINFORCE training, greedy evaluation, result saving, and CLI. |
| `qrl_cartpole/plotting.py` | Learning-curve and circuit-diagram figure generation. |
| `qrl_cartpole/dashboard.py` | Local HTTP server that serves results and the dashboard page. |
| `qrl_cartpole/dashboard.html` | Interactive results dashboard, including policy and seed filters and smoothing controls. |
| `outputs/final/results.json` | Combined source data for all six experiment runs. |
| `outputs/final/{quantum,classical}_seed_{7,21,42}.json` | Per-run episode returns, evaluation checkpoints, and run metadata. |
| `outputs/final/learning_curves.png` | Seed-averaged training curves with a one-standard-deviation band and final evaluations. |
| `outputs/final/variational_circuit.png` | Diagram of the four-qubit variational policy. |
| `README.md` | Installation, command-line usage, and a short experiment overview. |
| `easy_explanation.md` | Plain-language overview for a presentation audience. |
| `project_report.md` | Concise report of the experiment and results. |
| `pyproject.toml` | Python package metadata, dependencies, and console entry points. |

## Reproduce the final experiment

Python 3.11 through 3.14 is supported by the package metadata.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m qrl_cartpole.experiment \
  --episodes 500 \
  --seeds 7 21 42 \
  --eval-interval 10 \
  --out-dir outputs/final
```

The command writes one JSON file per policy and seed, combined `results.json`, and both figures. Re-running into `outputs/final` replaces those named result and image files with the new run's outputs.

To serve the dashboard after a run:

```bash
python -m qrl_cartpole.dashboard --results outputs/final/results.json --port 8001
```

Open `http://127.0.0.1:8001`. The dashboard lets you filter policies and seeds, choose the training-curve smoothing window, inspect final evaluations, and compare the per-run parameter and return summaries.

A short smoke run can check the basic command path:

```bash
python -m qrl_cartpole.experiment \
  --episodes 3 \
  --seeds 7 \
  --eval-interval 2 \
  --out-dir outputs/smoke
```

## Suggested next steps

1. **Strengthen the result estimate.** Run at least 20 to 30 independent seeds and report the mean, standard deviation, and a confidence interval for final evaluation and sample efficiency. Use more than five evaluation episodes for the final score.
2. **Define sample efficiency before tuning.** Choose a criterion such as the first checkpoint whose mean score exceeds a predeclared threshold, or the area under the evaluation curve. Report each seed and avoid treating a single five-episode score of 500 as sustained success.
3. **Compare tuned configurations fairly.** Run a small predeclared sweep over learning rate, entropy coefficient, circuit depth, and readout (`z` versus `zx`). Give both policy families comparable tuning budgets and keep final test seeds separate from tuning seeds.
4. **Add implementation checks.** Test the rotation and CNOT statevector operations against small hand-computed cases, verify expectation values stay in `[-1, 1]`, and confirm gradients reach all trainable parameters. Add a short seeded smoke test for the training and result-writing path.
5. **Measure cost explicitly.** Record wall-clock training time and evaluation time on the same machine. If the research question involves quantum hardware, add a clearly separated hardware or noise-aware study; the current exact simulator cannot answer that question.
6. **Make runs self-describing.** Save the effective CLI configuration, package versions, timestamp, and source revision with every result so future plots can be traced to the exact run that produced them.

## One-sentence presentation takeaway

> In this three-seed CartPole experiment, a four-qubit variational policy trained in an exact classical simulation reached the same final 500-step greedy score as a similarly sized MLP, but the MLP learned faster by the measured training-return and first-full-score checkpoints; this is not evidence of quantum advantage.

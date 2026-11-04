# Quantum CartPole in Plain English

## What is this project?

Imagine a stick balanced on top of a moving cart. The computer can push the cart left or right. Its job is to keep the stick balanced for as long as possible.

That game is called CartPole. A perfect episode lasts 500 steps.

## What are we comparing?

We built two tiny decision makers:

- A classical model: a very small neural network.
- A quantum model: a small circuit that acts like a trainable decision maker.

Both models see the same four measurements about the cart and stick. Both use the same learning method. The only major difference is how each model turns the measurements into an action.

## How does the quantum model work?

The quantum model uses four simulated qubits. A qubit is the basic unit in a quantum circuit.

The four CartPole measurements are turned into rotation angles. Those angles are loaded into the qubits. The circuit then runs two short layers of:

1. Trainable rotations.
2. Connections between neighboring qubits.

At the end, we measure two kinds of signal from each qubit. These signals are called X and Z expectation values. A small final layer turns those signals into two numbers:

- How strongly to choose left.
- How strongly to choose right.

The action with the larger number is chosen during evaluation.

The circuit is simulated on the computer. No quantum hardware is required.

## How does learning happen?

At first, each model makes mostly random choices. When an episode lasts longer, the model adjusts itself so that the choices made during that episode become more likely next time. This is a reinforcement-learning method called REINFORCE.

The models trained for 500 episodes with three different random seeds. Using several seeds matters because one training run can be lucky or unlucky.

## What happened?

The final greedy evaluation results were:

| Model | Average final score | Maximum possible |
|---|---:|---:|
| Quantum circuit | **500 / 500** | 500 |
| Classical neural network | **500 / 500** | 500 |

Both models solved the task by the end of training for all three seeds.

The classical model learned faster. Its average return during the final 15 training episodes was 460.4, compared with 378.5 for the quantum model. The quantum model caught up on the final evaluation, where both models held the pole for the full 500 steps.

## Is this quantum advantage?

No. This project shows that a small variational circuit can learn a simple control task in a classical simulation. It does not show that quantum computers are faster or better than classical computers.

The useful presentation message is:

> A small quantum policy can learn CartPole and reach the same final score as a small classical policy, although it learns more slowly in this experiment.

## What should I show in a presentation?

1. Start with the cart and pole: the goal is easy to understand.
2. Show the circuit diagram: measurements go in, rotations and connections happen, action scores come out.
3. Show the learning curve: the classical line rises sooner, while the quantum line catches up.
4. Show the final score: both reach 500 out of 500.
5. State the limitation clearly: this is a simulation and not a quantum-advantage claim.

## Where are the files?

- `project_report.md`: technical explanation and measured results.
- `outputs/final/results.json`: raw results for all six runs.
- `outputs/final/learning_curves.png`: learning-curve figure.
- `outputs/final/variational_circuit.png`: circuit diagram.
- `qrl_cartpole/dashboard.html`: interactive dashboard.

Start the dashboard with:

```bash
python -m qrl_cartpole.dashboard --results outputs/final/results.json --port 8001
```

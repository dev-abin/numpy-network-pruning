# NumPy network with gradual pruning

A small **partial implementation** of the AQUA hiring challenge: a NumPy classifier with manual backpropagation, mini-batch momentum, gradual magnitude/Taylor pruning, and sparse inference checks. The mandatory general autodiff engine and composable layers are **not implemented**.

Learning uses only NumPy; Matplotlib saves plots. Scikit-learn and neural-network frameworks are not used. See [DESIGN.md](DESIGN.md) for derivations, decisions and limitations.

## Install and reproduce

Python 3.10+ is required; tested with Python 3.12. These commands use a Windows virtual environment. On macOS/Linux, substitute `.venv/bin/python` for `.venv\Scripts\python`.

~~~powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

# Gradient checks for this fixed network
.venv\Scripts\python -m unittest test_main.NetworkTests.test_manual_gradients_match_finite_differences -v

# All 12 correctness tests
.venv\Scripts\python -m unittest -v

# Part 2 training and Part 3 gradual-pruning demonstration, together
.venv\Scripts\python main.py

# Part 4: both criteria, five sparsity targets and three seeds
.venv\Scripts\python sweep.py
~~~

The model is 2 → 8 → 2 with 32 connection weights and 10 biases. It uses a fixed synthetic train/test split, 150 warmup epochs and 150 pruning epochs, batch size 64, learning rate 0.01 and momentum 0.9. Pruning removes weights; biases remain active.

## Main result

Mean final test accuracy across seeds 7, 8 and 9, on data seed 42:

| Method | Weight sparsity | Accuracy ± sample SD |
|---|---:|---:|
| Unpruned control | 0% | 97.33% ± 0.33 pp |
| Magnitude | 50% | 97.00% ± 2.33 pp |
| Taylor-style | 50% | 78.22% ± 13.21 pp |

**Claim:** magnitude pruning halves active connection weights with a 0.33 percentage-point mean accuracy reduction against the equally trained control. Taylor loses on all three seeds at this sparsity; three seeds do not establish general superiority.

The sparse path matches dense predictions and uses **16 instead of 32 connection multiplications per input**. This excludes other arithmetic and overhead; runtime/storage savings are not measured. Taylor uses three extra training-data gradient evaluations for scoring, so equal optimizer updates do not mean equal total compute.

`main.py` reports the single seed-7 example (99.33%); the table reports the three-seed averages. Full protocol, curves and per-seed comparisons are in [DESIGN.md](DESIGN.md#evidence-and-limitations).

## Saved evidence

- `results/metrics.json`, `learning_curve.csv`, `summary.png`: single-run settings, all 300 epochs and loss/accuracy plots.
- `results/cost.json`: sparse equivalence result and connection-multiplication counts.
- `results/comparison.csv`, `comparison_summary.json`, `sparsity_accuracy.png`: all 30 comparison runs, paired statistics and the sparsity–accuracy plot.

The core code is in `main.py`, `sparse_inference.py` and `sweep.py`; tests are in `test_main.py`. Numerical checks cover this fixed network, not a general autodiff operation suite. Momentum is the allowed optimizer alternative; Adam and automatic regrowth are not implemented.

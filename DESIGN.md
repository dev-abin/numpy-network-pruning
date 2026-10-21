# Design and technical reasoning

This document explains the implemented prototype and addresses the four design questions in the challenge. The network uses manually derived gradients. A general tensor/computation-graph autodiff engine and its composable-layer integration remain incomplete.

## Implementation and experiment protocol

Inputs are 1,200 independently generated pairs drawn uniformly from [-1, 1). Label 1 means the coordinate product is positive; otherwise the label is 0. Use the first 900 generated samples for training and the remaining 300 for testing, with data seed 42. This XOR-style classification needs a nonlinear boundary but avoids downloads and keeps runs small.

The MLP has one tanh hidden layer:

~~~text
X: (N, 2)     W1: (2, 8)     b1: (8,)
hidden = tanh(X @ (W1 * M1) + b1)
W2: (8, 2)   b2: (2,)
logits = hidden @ (W2 * M2) + b2
~~~

There are 32 connection weights and 10 biases. Eight hidden units are a fixed simple choice rather than a tuned architecture. Initial weights have mean zero and standard deviation `sqrt(1 / input_width)`; this fan-in scaling keeps initial weighted sums moderate for tanh. Biases and momentum start at zero.

Subtract each row's maximum logit before exponentiation and compute cross entropy from log probabilities. Manual backpropagation starts with `(probabilities - one_hot_labels) / batch_length`, then applies the chain rule through both linear layers and tanh's derivative `1 - hidden**2`. Bias gradients sum over samples because a bias is shared across the batch.

Each epoch shuffles all training indices with a seeded generator and visits every sample once: 14 batches of 64 and one batch of 4. The final batch's gradients use its actual size. Momentum follows:

~~~text
velocity = 0.9 * velocity + gradient
parameter -= 0.01 * velocity
~~~

Training has two phases: 150 dense warmup epochs, then 150 epochs with gradual pruning. Total: 300 epochs and 4,500 optimizer updates. Learning rate stays 0.01. The final mask is reached during the second phase, leaving further training for surviving weights to adjust; a separate fine-tuning phase is omitted for simplicity. Momentum is the permitted optimizer alternative; Adam is not implemented.

## Importance criterion and derivation

Magnitude pruning is the baseline: score each active connection by `abs(weight)` and remove the lowest-scoring ones. It is simple, but weight size alone does not establish loss sensitivity, and magnitudes from different layers need not be directly comparable.

For the alternative `taylor` criterion, consider mean training loss L and one active weight w. Hold other weights and the current mask fixed. A first-order Taylor expansion gives:

~~~text
L(w + delta_w) - L(w) ≈ (dL/dw) * delta_w
Removing the connection means delta_w = -w
Estimated signed loss change ≈ -w * (dL/dw)
Importance score = abs(w * (dL/dw))
~~~

For example, weight 0.2 and gradient -0.1 give estimated signed removal cost +0.02. The sign can also be negative. Absolute value measures estimated change size rather than guaranteeing that removal increases loss.

At each pruning checkpoint, calculate the gradient of mean cross entropy over all 900 training examples. No test inputs/labels enter importance scoring. Scoring changes neither parameters nor momentum and does not consume shuffle randomness.

The approximation has limits: removing an entire weight is a finite change; gradients near a stationary point can be small despite curvature; averaging can cancel sample-level gradients; removing several weights introduces interactions. Numerical tests validate local fractional-removal sensitivity, not reliable prediction of a full deletion's loss change. These limitations may explain poor results, but no ablation establishes the exact cause here.

For a final removal target T, prune at the start of phase epochs 30, 60 and 90 to `0.3*T`, `0.6*T` and `T`. Warmup precedes these steps, and training continues through phase epoch 150. This stepped schedule is easy to inspect; no controlled experiment establishes that it beats one-shot pruning.

At T=50%, requested targets are 15%, 30% and 50%. Remove `ceil(32 * target)` total weights: 5, 10 and 16. Rank currently active connections only, preserve all previous removals, reject a lower removal count, and use stable sorting to break ties. Biases are excluded. A 90% target becomes 29 removals, or 90.625% achieved sparsity.

## Gradient of a masked weight and optimizer state

There is no general autodiff engine in this submission. The fixed network's manual derivatives implement this policy: for a binary mask M,

~~~text
W_effective = W * M
dL/dW = (dL/dW_effective) * M
~~~

A masked stored weight therefore has zero gradient, with M treated as constant during that backward calculation. This is the derivative of the forward computation actually used. We do not differentiate through the discrete ranking decision or apply a straight-through estimator.

Zero gradients alone are insufficient with momentum. A nonzero old velocity could keep moving a weight after removal. Pruning explicitly zeros both removed weights and their velocities. After every update, both are multiplied by their masks again. Surviving connections keep their momentum; biases remain unmasked.

Normal training is monotonic: removed connections do not return. A dedicated test manually re-enables one mask entry to check revival handling. Its stored weight and old velocity start at zero, so its first update uses only its new gradient. This is an optimizer-correctness test, not an automatic-regrowth feature.

The checks establish the policy for this architecture. They do not replace the mandatory graph-based autodiff engine or operation-level broadcasting tests.

## Bottlenecks and optimization

Because no general autodiff engine is implemented, we cannot report measured graph-engine bottlenecks. The current code has these concrete potential costs:

- Training uses dense matrix multiplication even after masking; zero weights do not reduce the executed matrix dimensions.
- Python loops orchestrate mini-batches, and a full training-set evaluation runs after each epoch.
- Taylor scoring adds three full-training forward/backward evaluations per nonzero target, without optimizer updates.
- Sparse inference uses Python connection loops and small NumPy vector operations, which can have substantial overhead.
- Dense weights, masks and momentum remain allocated. A list of Python tuples does not automatically save memory.

These are code-level observations, not profiler measurements. Profile before optimizing. Possible changes include less frequent epoch evaluation, array-level operation fusion, and optimized sparse kernels for inference. Larger batches alter optimization behavior and would need a fresh comparison. A future autodiff implementation would also need profiling for graph construction, traversal and intermediate storage; none of those costs has been measured here.

## Serving in a multi-tenant inference service

This is a proposed extension, not implemented infrastructure.

1. Finish training offline, freeze masks, and export active connections and biases as an immutable model version. Verify sparse/dense prediction equivalence before publishing it. Re-export after any training change.
2. Load approved model versions into a bounded worker cache and warm them before routing traffic. Keep model versions fixed per request and retain a rollback version.
3. Authenticate requests and enforce authorization to the requested tenant/model, per-tenant quotas and concurrency limits. Do not allow one tenant's requests to access another tenant's private model or outputs.
4. Batch compatible requests for the same model version while preserving request/result mapping. Use optimized sparse execution only if measured latency, throughput and memory beat the dense path for realistic batch sizes.
5. Load-test before targeting thousands of requests per second, then scale workers from measured capacity. Monitor errors, tail latency, throughput, queue depth and per-tenant resource use.

Connection-count reduction alone does not establish a deployment speedup. The current snapshot is a correctness demonstration; kernel choice and replica sizing require actual service measurements.

## Evidence and limitations

For each initialization seed (7, 8 and 9), train one warmup model. Every method/target gets an independent deep copy of that model, momentum and shuffle-generator state. Both criteria receive identical batch orders, learning rates, pruning checkpoints and 4,500 optimizer updates, including the 0% control.

Taylor additionally uses three full-training gradient evaluations for scoring. Equal optimizer budgets therefore do not imply equal training compute or wall time. At 0% no connections are removed, no scoring is needed, and both controls agree exactly.

Mean final test accuracy ± sample standard deviation:

| Target removal | Achieved removal | Active weights | Magnitude | Taylor-style |
|---|---|---:|---|---|
| 0% | 0% | 32 | 97.33% ± 0.33 pp | 97.33% ± 0.33 pp |
| 25% | 25% | 24 | 97.11% ± 0.51 pp | 96.56% ± 0.84 pp |
| 50% | 50% | 16 | 97.00% ± 2.33 pp | 78.22% ± 13.21 pp |
| 75% | 75% | 8 | 55.78% ± 14.55 pp | 53.33% ± 2.03 pp |
| 90% | 90.625% | 3 | 49.89% ± 0.19 pp | 50.67% ± 4.41 pp |

At 50% sparsity, the paired comparison is:

| Seed | Magnitude accuracy | Taylor accuracy | Taylor minus magnitude |
|---|---:|---:|---:|
| 7 | 99.33% | 85.00% | -14.33 pp |
| 8 | 94.67% | 86.67% | -8.00 pp |
| 9 | 97.00% | 63.00% | -34.00 pp |

Mean paired difference is -18.78 percentage points, sample SD 13.56 pp. Taylor loses on all three seeds here. At 90.625% sparsity its mean advantage is 0.78 pp, with paired SD 4.60 pp and mixed signs; this is not convincing evidence of improvement.

**Falsifiable claim:** on this fixed dataset split and 300-epoch protocol, magnitude pruning at 50% sparsity averages 97.00% accuracy versus 97.33% for the dense control, a 0.33 percentage-point reduction while halving active connection weights. See `results/comparison.csv` for all 30 runs and `comparison_summary.json` for settings and statistics.

The single seed-7 demonstration ends at 99.33% accuracy, with loss 0.09276 and 16 active weights. It is distinct from the three-seed aggregate. `learning_curve.csv` contains all 300 epoch records; `summary.png` shows loss and accuracy. `sparsity_accuracy.png` compares both criteria at achieved sparsities.

For cost evidence, `SparseNetwork` exports active `(source, destination, weight)` tuples and copies biases. Its forward pass initializes outputs from biases, visits only exported connections, then applies tanh and log-softmax. `sparse_linear` increments a multiplication counter by batch length for each visited connection. At 50% sparsity that is 16 weight multiplications per input versus the dense matrix-shape count of 32.

`check_sparse_inference` compares the final example model's log probabilities within numerical tolerance and predicted classes exactly. `results/cost.json` records matching predictions, counts and reduction. Separate tests cover other sparsities, nonzero biases, masked stored values and the all-pruned case. The sweep itself handles training comparisons only.

Only connection multiplications are counted. Bias initialization, additions, activations, log-softmax, export work and runtime overhead are excluded. There is no total hardware-FLOP, latency or memory/storage claim.

Three seeds measure limited initialization/batch-order variation on one fixed split, not statistical significance or dataset uncertainty. This is an exploratory development comparison; the simplified protocol was chosen after earlier results had been reviewed. Further tuning needs validation data and independent confirmation. Current artifacts replace the earlier 330-epoch results. No controlled gradual-versus-one-shot comparison or automatic-regrowth experiment is included.

## Correctness checks and incomplete scope

The 12 tests cover numerical gradients with/without masks; persistent zero weights/momentum; a momentum update and manual revival; exact budgets; complete mini-batch coverage; pruning checkpoints and monotonic masks; seed repeatability; sparse prediction/count equivalence; snapshot independence; numerical Taylor sensitivity; sensitivity-versus-magnitude ranking; and scoring calls/identical unpruned controls.

The general tensor/graph engine, its differentiable-operation collection, topological backward/broadcasting accumulation and composable layers are not implemented. Numerical checks cover the current fixed network. Adam and automatic regrowth are absent; momentum is the allowed minimum optimizer alternative, and regrowth is bonus work. These scope limits remain explicit rather than treating a feature-count checklist as an estimated score.

"""Sparse inference with prediction checks and connection-multiplication counts."""

import numpy as np


def active_connections(weights, mask):
    """Return active connections as (source index, destination index, weight) tuples."""
    # Find active mask positions and copy their weights into a connection list.
    return [(int(i), int(j), float(weights[i, j])) for i, j in np.argwhere(mask != 0)]


def sparse_linear(x, connections, bias):
    """Apply active connections to a batch; return outputs and multiplication count."""
    # Start each sample's output values with the neuron biases.
    output = np.broadcast_to(bias, (len(x), len(bias))).copy()
    multiplications = 0
    # Visit surviving connections only; removed connections have no list entry.
    for source, destination, weight in connections:
        # Add this connection's contribution to its destination for every sample.
        output[:, destination] += x[:, source] * weight
        multiplications += len(x)  # One scalar weight multiplication per sample.
    return output, multiplications


class SparseNetwork:
    """An inference snapshot. Export again if training changes the model."""

    def __init__(self, model):
        """Copy a trained model's active weights and biases into a fixed snapshot."""
        # Export both layers; later model updates require a new snapshot.
        self.connections1 = active_connections(model.w1, model.mask1)
        self.connections2 = active_connections(model.w2, model.mask2)
        # Copy biases so this snapshot is independent of the training model.
        self.b1 = model.b1.copy()
        self.b2 = model.b2.copy()
        self.last_connection_multiplications = 0

    def forward(self, x):
        """Return hidden values (N, 8) and class log probabilities (N, 2)."""
        # Calculate the hidden layer using active connections, then apply tanh.
        hidden, count1 = sparse_linear(x, self.connections1, self.b1)
        hidden = np.tanh(hidden)
        # Convert hidden values into two raw class scores per sample.
        logits, count2 = sparse_linear(hidden, self.connections2, self.b2)
        # Record connection multiplications across both layers for the whole batch.
        self.last_connection_multiplications = count1 + count2
        # Calculate stable log-softmax, using the same formula as the dense model.
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        return hidden, log_probs


def check_sparse_inference(model, x):
    """Verify matching predictions and count connection multiplications per input."""
    # Export the model, then evaluate identical inputs through both forward paths.
    sparse = SparseNetwork(model)
    _, dense_log_probs = model.forward(x)
    _, sparse_log_probs = sparse.forward(x)

    # Small floating-point differences are allowed; class predictions must match.
    np.testing.assert_allclose(sparse_log_probs, dense_log_probs, atol=1e-10, rtol=1e-10)
    predictions_match = bool(np.array_equal(
        sparse_log_probs.argmax(axis=1), dense_log_probs.argmax(axis=1)
    ))
    if not predictions_match:
        raise ValueError("Sparse and dense predictions differ.")

    # Dense count comes from matrix sizes; sparse loops count their own operations.
    dense_count = int(model.w1.size + model.w2.size)
    # Divide the batch's total count by its sample count to get a per-input count.
    sparse_count = sparse.last_connection_multiplications // len(x)
    # Report arithmetic savings as a fraction; this does not measure runtime.
    return {
        "dense_connection_multiplications_per_input": dense_count,
        "sparse_connection_multiplications_per_input": sparse_count,
        "connection_multiplication_reduction": 1 - sparse_count / dense_count,
        "predictions_match": predictions_match,
        "count_scope": "Connection multiplications only; excludes other operations and overhead. No total FLOP, runtime or storage savings claimed.",
    }

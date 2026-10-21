"""Small NumPy classifier with magnitude and first-order sensitivity pruning."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save a plot without opening a window.
import matplotlib.pyplot as plt
import numpy as np
from sparse_inference import check_sparse_inference


def pruning_schedule(target):
    """Plan three gradual pruning steps toward a final removal fraction.

    Args:
        target: Final fraction of connection weights to remove, such as 0.5.

    Returns:
        A dictionary mapping pruning-phase epochs 30, 60 and 90 to cumulative
        removal targets. For target=0.5, these targets are 0.15, 0.30 and 0.50.
        Epoch numbers start at 1 within the training call using this schedule.
    """
    # Three easy-to-explain steps, followed by time to adjust at the final mask.
    return {30: 0.3 * target, 60: 0.6 * target, 90: target}


def make_data(seed=42):
    """Create a repeatable two-class dataset and separate training/test samples.

    Each sample contains two numbers drawn uniformly between -1 and 1.
    Its label is 1 if their product is positive, and 0 otherwise.

    Args:
        seed: Random seed controlling the generated numbers.

    Returns:
        x_train, y_train, x_test, y_test: NumPy arrays with shapes (900, 2),
        (900,), (300, 2) and (300,). Labels are integers 0 or 1.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(1200, 2))
    # XOR: matching signs belong to class 1; opposite signs to class 0.
    y = (x[:, 0] * x[:, 1] > 0).astype(int)
    # Samples are randomly generated, so taking a slice gives a fixed split.
    return x[:900], y[:900], x[900:], y[900:]


class SmallNetwork:
    def __init__(self, seed=7):
        """Initialize a network with 2 inputs, 8 hidden units and 2 outputs.

        Create random weights, zero biases, all-one connection masks and zero
        momentum velocities. A separate seeded generator controls batch shuffles.
        Initialization creates the parameters; learning happens in train().

        Args:
            seed: Random seed for initial weights and reproducible shuffling.

        Returns:
            None. Stores the network parameters and training state on self.
        """
        rng = np.random.default_rng(seed)
        # Scale initial weights by input count to help tanh avoid saturation.
        self.w1 = rng.normal(0, np.sqrt(1 / 2), size=(2, 8))
        # Creates 16 weights connecting the 2 inputs to the 8 hidden neurons.
        self.b1 = np.zeros(8)
        # Gives each hidden neuron a bias, initially zero.
        self.w2 = rng.normal(0, np.sqrt(1 / 8), size=(8, 2))
        # Creates 16 weights connecting the 8 hidden neurons to the 2 outputs.
        self.b2 = np.zeros(2)
        # Gives each output a bias, initially zero.
        self.mask1 = np.ones_like(self.w1)
        # Creates a mask matching w1. All ones means every connection is initially active.
        self.mask2 = np.ones_like(self.w2)
        self.velocities = [np.zeros_like(p) for p in (self.w1, self.b1, self.w2, self.b2)]
        # Creates zero-filled arrays that will store momentum during training.
        self.shuffle_rng = np.random.default_rng(seed + 1000)
        # Creates a separate generator for shuffling training samples.

    def forward(self, x):
        """Calculate hidden values and class log probabilities for input samples.
        Apply masked weights, a tanh hidden layer and a stable log-softmax.

        Args:
            x: Input array with shape (N, 2), where N is the sample count.

        Returns:
            hidden: Hidden-layer values with shape (N, 8).
            log_probs: Log probabilities for the two classes, shape (N, 2).

        """
        # Apply connection masks, calculate hidden weighted sums, then apply tanh.
        hidden = np.tanh(x @ (self.w1 * self.mask1) + self.b1)
        # Convert the 8 hidden values into 2 raw class scores per sample.
        logits = hidden @ (self.w2 * self.mask2) + self.b2
        # Subtract the row maximum to avoid overflow in exp.
        shifted = logits - logits.max(axis=1, keepdims=True)
        # Calculate log probabilities: score minus log(sum of exponentiated scores).
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        # Return hidden values for backpropagation and log probabilities for the loss.
        return hidden, log_probs

    def evaluate(self, x, y):
        """Measure average prediction loss and accuracy without training.

        Args:
            x: Input samples with shape (N, 2).
            y: Correct class labels, 0 or 1, with shape (N,).

        Returns:
            A tuple (loss, accuracy). Loss is mean cross entropy; accuracy is
            the fraction predicted correctly, such as 0.97 for 97 percent.
        """
        # Get class log probabilities;
        _, log_probs = self.forward(x)
        # Select each sample's correct-class log probability, average, then negate.
        loss = -log_probs[np.arange(len(y)), y].mean()
        # Pick the highest-scoring class per sample and calculate the fraction correct.
        accuracy = (log_probs.argmax(axis=1) == y).mean()
        # return both metrics.
        return float(loss), float(accuracy)

    def gradients(self, x, y):
        """Compute parameter gradients using handwritten backpropagation.

        Differentiate mean cross-entropy loss through the output layer, tanh
        and input layer. Masked weights receive zero gradients. This only
        calculates gradients; it does not change weights or momentum.

        Args:
            x: Input batch with shape (N, 2).
            y: Correct class labels with shape (N,). Gradients are averaged
                over this batch's actual sample count.

        Returns:
            d_w1, d_b1, d_w2, d_b2: Gradient arrays matching the corresponding
            parameter shapes (2, 8), (8,), (8, 2) and (2,).
        """
        hidden, log_probs = self.forward(x)
        # Convert log probabilities back into ordinary probabilities.
        d_logits = np.exp(log_probs)
        # Subtract 1 from each sample's correct-class probability.
        # This gives the softmax + cross-entropy gradient: probabilities - one-hot labels.
        d_logits[np.arange(len(y)), y] -= 1
        # Average gradients because our loss is averaged over the batch.
        d_logits /= len(y)

        # Calculate output-weight gradients; mask removed connections to zero.
        # hidden.T @ d_logits combines contributions from all samples.
        d_w2 = (hidden.T @ d_logits) * self.mask2
        # # Sum contributions from all samples for each shared output bias.
        d_b2 = d_logits.sum(axis=0)
        # # Pass the loss gradient backward through the output weights to hidden values.
        d_hidden = d_logits @ (self.w2 * self.mask2).T
        # Apply tanh's derivative to get gradients of the sums BEFORE tanh.
        d_pre_hidden = d_hidden * (1 - hidden**2)
        # Calculate input-weight gradients; mask removed connections to zero.
        d_w1 = (x.T @ d_pre_hidden) * self.mask1
        # Sum contributions from all samples for each shared hidden-layer bias.
        d_b1 = d_pre_hidden.sum(axis=0)
        # Return gradients in the same order as the network's parameters.
        return d_w1, d_b1, d_w2, d_b2

    def train(self, x, y, epochs=150, learning_rate=0.01, momentum=0.9,
              batch_size=64, schedule=None, criterion="magnitude"):
        """Update the network using shuffled mini-batches and momentum.

        Visit every sample once per epoch, including a final short batch.
        Optional pruning happens before batch updates at scheduled epochs.
        Weights, momentum and shuffle state persist between calls.

        Raises:
            ValueError: If batch_size is not positive or training loss becomes
                non-finite. Invalid pruning requests can also raise ValueError.
        """
        # Reject zero or negative batch sizes.
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")

        # Store loss, accuracy and sparsity after each epoch.
        history = []

        # One epoch means one complete pass through the training data.
        for epoch in range(epochs):

            # If this epoch has a scheduled pruning step, remove connections first.
            if schedule and epoch + 1 in schedule:
                self.prune(schedule[epoch + 1], criterion=criterion, x=x, y=y)

            # Shuffle sample indices so batches have a different order each epoch.
            order = self.shuffle_rng.permutation(len(y))

            # Move through the shuffled indices one batch at a time.
            for start in range(0, len(y), batch_size):
                indices = order[start:start + batch_size]
                # Calculate gradients using this batch's inputs and matching labels.
                gradients = self.gradients(x[indices], y[indices])
                # Momentum remembers previous gradients between batch updates.

                # Match each parameter with its gradient, momentum velocity and mask.
                for parameter, gradient, velocity, mask in zip(
                    (self.w1, self.b1, self.w2, self.b2), gradients,
                    self.velocities, (self.mask1, 1, self.mask2, 1)
                ):
                    # Retain a fraction of the previous momentum.
                    velocity *= momentum

                    # Add the current batch's gradient to that momentum.
                    velocity += gradient

                    # Clear momentum for removed connections.
                    velocity *= mask

                    # Update the parameter in the direction opposite to the momentum.
                    parameter -= learning_rate * velocity

                    # Keep removed weights exactly zero.
                    parameter *= mask

            # After all batches, measure performance on the full training dataset.
            loss, accuracy = self.evaluate(x, y)

            # Stop if loss becomes NaN or infinity.
            if not np.isfinite(loss):
                raise ValueError("Training produced a non-finite loss.")

            # Calculate the fraction of connection weights removed; exclude biases.
            sparsity = 1 - (self.mask1.sum() + self.mask2.sum()) / (self.w1.size + self.w2.size)

            # Save this epoch's measurements.
            history.append({"epoch": epoch + 1, "loss": loss, "accuracy": accuracy,
                            "sparsity": float(sparsity)})
        return history

    def importance(self, criterion="magnitude", x=None, y=None):
        """Assign each connection a score used to decide which weights survive.

        Magnitude uses abs(weight). Taylor uses abs(weight * gradient), a
        first-order estimate of loss-change size when a connection is removed.
        The Taylor score does not guarantee the actual loss will increase.
        Computing scores does not update parameters or momentum.

        Args:
            criterion: "magnitude" or "taylor".
            x: Training inputs with shape (N, 2), required for Taylor scores.
            y: Training labels with shape (N,), required for Taylor scores.

        Returns:
            Two score arrays with shapes (2, 8) and (8, 2), corresponding to
            w1 and w2. Lower scores are candidates for removal.

        Raises:
            ValueError: If the criterion is unknown or Taylor data is missing.
        """
        # Magnitude pruning: use each weight's absolute value as its importance score.
        if criterion == "magnitude":
            return np.abs(self.w1), np.abs(self.w2)

        # Taylor pruning: also consider how sensitive the loss is to each weight.
        if criterion == "taylor":

            # Inputs and labels are needed to calculate gradients.
            if x is None or y is None:
                raise ValueError("Taylor importance needs training data and labels.")

            # Get weight gradients for both layers; discard the unused bias gradients.
            g1, _, g2, _ = self.gradients(x, y)

            # Estimate loss-change size using abs(weight × gradient).
            # Lower scores are removal candidates; this is only a local approximation.
            return np.abs(self.w1 * g1), np.abs(self.w2 * g2)

        # Reject any unsupported scoring rule.
        raise ValueError("Criterion must be 'magnitude' or 'taylor'.")

    def prune(self, sparsity=0.5, criterion="magnitude", x=None, y=None):
        """Remove low-scoring connections until a cumulative budget is reached.

        Rank currently active weights across both layers; keep biases and all
        previous removals. Zero newly removed weights and their momentum.
        Round the removal count up: 90 percent of 32 weights means removing 29.

        Args:
            sparsity: Desired total fraction removed, between 0 and 1. This is
                a cumulative target, not an extra fraction to remove each call.
            criterion: "magnitude" or "taylor" for connection ranking.
            x: Training inputs, needed when Taylor scores require new removals.
            y: Training labels, needed when Taylor scores require new removals.

        Returns:
            None. Changes masks, weights and momentum on this network in place.
            If the requested removal count is already reached, does nothing.

        Raises:
            ValueError: If sparsity is outside [0, 1], the requested count would
                revive connections, or the required importance inputs are invalid.
        """

        # Check that the requested removal fraction is between 0% and 100%.
        if not 0 <= sparsity <= 1:
            raise ValueError("Sparsity must be between 0 and 1.")

        # Flatten both layer masks, join them into one array, and make a separate copy.
        mask = np.concatenate((self.mask1.ravel(), self.mask2.ravel())).copy()

        # Calculate the TOTAL number of weights to remove, rounding up.
        count = int(np.ceil(len(mask) * sparsity))

        # Count connections already removed: their mask value is zero.
        already_removed = int((mask == 0).sum())

        # Reject a target that would require bringing removed connections back.
        if count < already_removed:
            raise ValueError("Pruning cannot lower sparsity or revive connections.")

        # Stop if the requested number of removals has already been reached.
        if count == already_removed:
            return

        # Calculate importance scores for both weight matrices.
        score1, score2 = self.importance(criterion, x, y)

        # Flatten and join scores in the same order as the combined mask.
        scores = np.concatenate((score1.ravel(), score2.ravel()))

        # Get indices of connections that are still active.
        active = np.flatnonzero(mask)

        # Order active connections from lowest to highest importance.
        # Stable sorting preserves the existing order when scores are equal.
        ranked = active[np.argsort(scores[active], kind="stable")]

        # Disable only the additional connections needed to reach the total target.
        mask[ranked[:count - already_removed]] = 0

        # Split the combined mask back into the first layer's original matrix shape.
        self.mask1 = mask[:self.w1.size].reshape(self.w1.shape)

        # Restore the remaining mask entries to the second layer's matrix shape.
        self.mask2 = mask[self.w1.size:].reshape(self.w2.shape)

        # Set removed weights to exactly zero in both layers.
        self.w1 *= self.mask1
        self.w2 *= self.mask2

        # Clear momentum for removed weights so old updates cannot move them.
        # Velocity entries 0 and 2 belong to w1 and w2; biases are not pruned.
        self.velocities[0] *= self.mask1
        self.velocities[2] *= self.mask2


def save_report(results, cost, phases, schedule):
    """Save JSON metrics, a CSV learning curve and plots, then print a summary.

    results contains settings and measured loss/accuracy; cost contains sparse
    prediction checks and multiplication counts. phases is an ordered list of
    (phase name, epoch records) pairs. schedule supplies pruning checkpoints.
    Files are written to the results folder beside this script.
    """
    output = Path(__file__).resolve().parent / "results"
    output.mkdir(exist_ok=True)

    (output / "cost.json").write_text(json.dumps(cost, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")

    # Save all phases in one CSV using ordinary loops and continuous epoch numbers.
    with (output / "learning_curve.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "phase", "loss", "accuracy", "sparsity"])
        writer.writeheader()
        epoch = 0
        for phase, records in phases:
            for row in records:
                epoch += 1
                writer.writerow({
                    "epoch": epoch,
                    "phase": phase,
                    "loss": row["loss"],
                    "accuracy": row["accuracy"],
                    "sparsity": row["sparsity"],
                })

    # Keep only the required training loss and accuracy learning curves.
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # Join both phases so learning curves cover the entire experiment.
    combined_history = []
    for phase, records in phases:
        combined_history.extend(records)

    # First plot: training loss after each epoch.
    axes[0].plot([row["loss"] for row in combined_history])

    warmup_epochs = results["warmup_epochs"]

    # Mark each pruning checkpoint on both learning-curve plots.
    # Add the warmup length to convert phase epochs into overall plot positions.
    for step in schedule:
        axes[0].axvline(warmup_epochs + step - 1, color="gray", linestyle="--")
        axes[1].axvline(warmup_epochs + step - 1, color="gray", linestyle="--")

    # Set the first plot's title and axis labels.
    axes[0].set(
        title="Training loss",
        xlabel="Epoch (zero-based)",
        ylabel="Cross entropy"
    )

    # Second plot: training accuracy after each epoch.
    axes[1].plot([row["accuracy"] for row in combined_history])
    axes[1].set(
        title="Training accuracy",
        xlabel="Epoch (zero-based)",
        ylabel="Accuracy",
        ylim=(0, 1)
    )

    # Adjust spacing, save the figure and release its resources.
    fig.tight_layout()
    fig.savefig(output / "summary.png", dpi=150)
    plt.close(fig)

    # Print a short summary; detailed measurements remain in metrics.json.
    print(f"Final test accuracy: {results['test_accuracy_after']:.2%}; weight sparsity: {results['weight_sparsity']:.0%}")
    print(
        f"Connection multiplications per input: "
        f"{cost['dense_connection_multiplications_per_input']} dense, "
        f"{cost['sparse_connection_multiplications_per_input']} sparse."
    )
    print(f"Results saved to {output}")

def main():
    """Run warmup and gradual pruning, then hand results to save_report."""

    # Create training inputs/labels and separate test inputs/labels.
    x_train, y_train, x_test, y_test = make_data()

    # Initialize the small network with random weights.
    model = SmallNetwork()

    # Train for 150 warmup epochs with all connections active.
    history = model.train(x_train, y_train)

    # Measure test loss and accuracy before any pruning.
    before_loss, before_accuracy = model.evaluate(x_test, y_test)

    # Plan three pruning steps toward 50% total weight removal.
    schedule = pruning_schedule(0.5)

    # Continue training for 150 epochs, pruning at the scheduled checkpoints.
    # Magnitude is the default importance criterion.
    pruning_history = model.train(x_train, y_train, schedule=schedule)
    # Measure test performance after the entire gradual-pruning phase.
    after_loss, after_accuracy = model.evaluate(x_test, y_test)

    # Check that every removed weight is still exactly zero.
    for weights, mask in ((model.w1, model.mask1), (model.w2, model.mask2)):
        assert np.all(weights[mask == 0] == 0), "Pruned weights must remain zero."

    # Collect experiment settings and measured results for the saved report.
    results = {
        # Seeds make the generated data and starting weights reproducible.
        "data_seed": 42,
        "model_seed": 7,

        # Record how many examples were used for training and testing.
        "train_samples": len(y_train),
        "test_samples": len(y_test),

        # Record the training settings used in each phase.
        "warmup_epochs": 150,
        "pruning_epochs": 150,
        "batch_size": 64,
        "pruning_schedule": schedule,
        "learning_rate": 0.01,
        "momentum": 0.9,

        # Count all connection weights and those whose masks remain active.
        "total_weights": int(model.w1.size + model.w2.size),
        "active_weights_after_pruning": int(
            model.mask1.sum() + model.mask2.sum()
        ),

        # Biases are kept and excluded from the weight-sparsity calculation.
        "unpruned_biases": int(model.b1.size + model.b2.size),
        "weight_sparsity": 0.5,

        # Save test measurements before and after gradual pruning.
        "test_loss_before": before_loss,
        "test_loss_after": after_loss,
        "test_accuracy_before": before_accuracy,
        "test_accuracy_after": after_accuracy,

        # Multiply accuracy differences by 100 to express percentage points.
        "accuracy_change_percentage_points":
            100 * (after_accuracy - before_accuracy),

        # Explain that sparse inference costs are measured separately.
        "cost_note": (
            "Training remains dense; results/cost.json measures a separate "
            "sparse inference path. Counts cover connection multiplications "
            "only; no runtime speedup is claimed."
        ),
    }

    # Verify sparse inference, then hand all reporting data to one function.
    cost = check_sparse_inference(model, x_test)
    phases = [("warmup", history), ("pruning", pruning_history)]
    save_report(results, cost, phases, schedule)


if __name__ == "__main__":
    main()

"""Focused network and optimizer checks, not a general autodiff suite."""

import unittest
import copy
import numpy as np
from main import SmallNetwork, pruning_schedule
from sparse_inference import SparseNetwork


class NetworkTests(unittest.TestCase):
    def test_manual_gradients_match_finite_differences(self):
        """Compare handwritten gradients with numerical slopes, with and without masks."""
        x = np.array([[0.2, -0.4], [0.6, 0.3], [-0.5, -0.7]])
        y = np.array([0, 1, 1])
        for masked in (False, True):
            model = SmallNetwork()
            if masked:
                model.prune()
            gradients = model.gradients(x, y)
            for parameter, analytic in zip((model.w1, model.b1, model.w2, model.b2), gradients):
                numeric = np.zeros_like(parameter)
                for index in np.ndindex(parameter.shape):
                    original = parameter[index]
                    parameter[index] = original + 1e-5
                    plus, _ = model.evaluate(x, y)
                    parameter[index] = original - 1e-5
                    minus, _ = model.evaluate(x, y)
                    parameter[index] = original
                    numeric[index] = (plus - minus) / 2e-5
                np.testing.assert_allclose(analytic, numeric, atol=1e-7, rtol=1e-5)

    def test_pruned_connections_stay_zero_after_training(self):
        """Check removed weights and their old momentum stay zero during training."""
        model = SmallNetwork()
        model.velocities[0].fill(2)
        model.velocities[2].fill(2)
        model.prune()
        self.assertEqual(int(model.mask1.sum() + model.mask2.sum()), 16)
        x = np.array([[0.2, -0.4], [0.6, 0.3]])
        y = np.array([0, 1])
        model.train(x, y, epochs=5)
        for weights, mask in ((model.w1, model.mask1), (model.w2, model.mask2)):
            np.testing.assert_array_equal(weights[mask == 0], 0)
        np.testing.assert_array_equal(model.velocities[0][model.mask1 == 0], 0)
        np.testing.assert_array_equal(model.velocities[2][model.mask2 == 0], 0)

    def test_momentum_update_and_manual_revival(self):
        """Check the momentum update and ensure revived weights use no stale velocity."""
        model = SmallNetwork()
        model.velocities[0].fill(2)
        model.prune()
        index = tuple(np.argwhere(model.mask1 == 0)[0])
        self.assertEqual(model.velocities[0][index], 0)
        model.mask1[index] = 1  # Deliberate revival for a correctness test only.
        x = np.array([[0.2, -0.4], [0.6, 0.3]])
        y = np.array([0, 1])
        params = (model.w1, model.b1, model.w2, model.b2)
        masks = (model.mask1, 1, model.mask2, 1)
        gradients = model.gradients(x, y)
        expected_v = [(0.9 * v + g) * m for v, g, m in zip(model.velocities, gradients, masks)]
        expected_p = [(p - 0.01 * v) * m for p, v, m in zip(params, expected_v, masks)]
        # Both samples form one batch, so its average gradient is unchanged.
        model.train(x, y, epochs=1)
        for actual, expected in zip(params, expected_p):
            np.testing.assert_allclose(actual, expected)
        # The revived weight starts at zero and moves only with its new gradient.
        self.assertAlmostEqual(model.w1[index], -0.01 * gradients[0][index])

    def test_sparsity_budgets(self):
        """Check each target removes the correct rounded-up number of weights."""
        for target in (0, 0.25, 0.5, 0.75, 0.9, 1):
            model = SmallNetwork()
            model.prune(target)
            active = int(model.mask1.sum() + model.mask2.sum())
            self.assertEqual(active, 32 - int(np.ceil(32 * target)))

    def test_minibatches_visit_every_sample_including_last_short_batch(self):
        """Check each epoch visits every sample once, including the short final batch."""
        model = SmallNetwork()
        x = np.arange(10).reshape(5, 2) / 10
        y = np.array([0, 1, 0, 1, 0])
        batches = []
        original_gradients = model.gradients

        def record_batch(batch_x, batch_y):
            """Record batch samples while calculating their gradients."""
            batches.append(batch_x.copy())
            return original_gradients(batch_x, batch_y)

        model.gradients = record_batch
        model.train(x, y, epochs=2, batch_size=2)
        self.assertEqual([len(batch) for batch in batches], [2, 2, 1, 2, 2, 1])
        for epoch in range(2):
            seen = np.concatenate(batches[epoch * 3:(epoch + 1) * 3])
            np.testing.assert_array_equal(np.sort(seen[:, 0]), x[:, 0])

    def test_gradual_pruning_preserves_masks_and_reaches_budget(self):
        """Verify pruning checkpoints reach their budgets without reviving weights."""
        model = SmallNetwork()
        x = np.array([[0.2, -0.4], [0.6, 0.3]])
        y = np.array([0, 1])
        old_mask = np.ones(32)
        for target in pruning_schedule(0.5).values():
            model.prune(target)
            model.train(x, y, epochs=2)
            mask = np.concatenate((model.mask1.ravel(), model.mask2.ravel()))
            self.assertTrue(np.all(mask <= old_mask))
            self.assertEqual(int((mask == 0).sum()), int(np.ceil(32 * target)))
            old_mask = mask.copy()
        with self.assertRaises(ValueError):
            model.prune(0.25)
        # Also check that train invokes pruning on the specified epochs.
        scheduled = SmallNetwork()
        history = scheduled.train(x, y, epochs=4, schedule={2: 0.25, 3: 0.5})
        self.assertEqual([row["sparsity"] for row in history], [0, 0.25, 0.5, 0.5])

    def test_same_seed_repeats_minibatch_training(self):
        """Check the same seed reproduces learning history and first-layer weights."""
        x = np.array([[0.2, -0.4], [0.6, 0.3], [-0.2, 0.1]])
        y = np.array([0, 1, 0])
        first, second = SmallNetwork(), SmallNetwork()
        h1 = first.train(x, y, epochs=3, batch_size=2)
        h2 = second.train(x, y, epochs=3, batch_size=2)
        self.assertEqual(h1, h2)
        np.testing.assert_array_equal(first.w1, second.w1)

    def test_sparse_forward_matches_dense_including_bias_only_network(self):
        """Compare sparse/dense outputs and operation counts from 0% to 100% pruning."""
        x = np.random.default_rng(5).normal(size=(9, 2))
        for target in (0, 0.25, 0.5, 0.75, 0.9, 1):
            model = SmallNetwork()
            model.b1[:] = np.linspace(-0.2, 0.2, 8)
            model.b2[:] = [-0.1, 0.3]
            model.prune(target)
            # Masks, not stored zero values, must decide which connections exist.
            model.w1[model.mask1 == 0] = 123
            model.w2[model.mask2 == 0] = -456
            sparse = SparseNetwork(model)
            for batch in (x[:1], x):
                expected_hidden, expected_log_probs = model.forward(batch)
                hidden, log_probs = sparse.forward(batch)
                np.testing.assert_allclose(hidden, expected_hidden, atol=1e-12)
                np.testing.assert_allclose(log_probs, expected_log_probs, atol=1e-12)
                np.testing.assert_array_equal(log_probs.argmax(axis=1), expected_log_probs.argmax(axis=1))
                self.assertEqual(sparse.last_connection_multiplications,
                                 len(batch) * (32 - int(np.ceil(32 * target))))

    def test_sparse_export_is_a_fixed_snapshot(self):
        """Check sparse predictions stay fixed when the original model changes."""
        model = SmallNetwork()
        sparse = SparseNetwork(model)
        x = np.array([[0.2, -0.4]])
        _, before = sparse.forward(x)
        model.w1.fill(100)
        model.b2[:] = [-50, 50]
        _, after = sparse.forward(x)
        np.testing.assert_array_equal(before, after)

    def test_taylor_importance_matches_numerical_removal_sensitivity(self):
        """Check Taylor scores against numerical sensitivity to tiny weight removals."""
        model = SmallNetwork()
        x = np.array([[0.2, -0.4], [0.6, 0.3], [-0.5, -0.7]])
        y = np.array([0, 1, 1])
        scores = model.importance("taylor", x, y)
        for weights, score in zip((model.w1, model.w2), scores):
            index = np.unravel_index(np.argmax(score), score.shape)
            original = weights[index]
            epsilon = 1e-5
            # Differentiate loss with respect to a tiny fractional removal of w.
            weights[index] = original * (1 - epsilon)
            loss_removed, _ = model.evaluate(x, y)
            weights[index] = original * (1 + epsilon)
            loss_added, _ = model.evaluate(x, y)
            weights[index] = original
            numerical_sensitivity = abs((loss_removed - loss_added) / (2 * epsilon))
            self.assertAlmostEqual(score[index], numerical_sensitivity, places=7)
        with self.assertRaises(ValueError):
            model.importance("taylor")

    def test_taylor_ranks_by_sensitivity_instead_of_magnitude(self):
        """Check Taylor can rank differently from magnitude and clears pruned momentum."""
        model = SmallNetwork()
        model.w1[:] = np.arange(1, 17).reshape(2, 8)
        model.w2[:] = np.arange(17, 33).reshape(8, 2)
        g1, g2 = np.ones_like(model.w1), np.ones_like(model.w2)
        g1[0, 0] = 100  # Smallest weight is sensitive.
        g2[-1, -1] = 0  # Largest weight is locally insensitive.
        model.gradients = lambda x, y: (g1, np.zeros(8), g2, np.zeros(2))
        magnitude = copy.deepcopy(model)
        magnitude.prune(1 / 32)
        model.velocities[2].fill(10)
        model.prune(1 / 32, criterion="taylor", x=np.zeros((1, 2)), y=np.array([0]))
        self.assertEqual(magnitude.mask1[0, 0], 0)
        self.assertEqual(model.mask1[0, 0], 1)
        self.assertEqual(model.mask2[-1, -1], 0)
        self.assertEqual(model.w2[-1, -1], 0)
        self.assertEqual(model.velocities[2][-1, -1], 0)

    def test_taylor_scoring_adds_no_updates_and_zero_controls_agree(self):
        """Check scoring-call counts, scheduled budgets and matching unpruned controls."""
        x = np.array([[0.2, -0.4], [0.6, 0.3], [-0.2, 0.1]])
        y = np.array([0, 1, 0])
        model = SmallNetwork()
        sizes = []
        original_gradients = model.gradients

        def record_size(batch_x, batch_y):
            """Record call sizes to distinguish scoring from mini-batch updates."""
            sizes.append(len(batch_y))
            return original_gradients(batch_x, batch_y)

        model.gradients = record_size
        history = model.train(x, y, epochs=2, batch_size=2,
                              schedule={1: 0.25, 2: 0.5}, criterion="taylor")
        # Two full-data scoring calls and four mini-batch update calls.
        self.assertEqual(sizes, [3, 2, 1, 3, 2, 1])
        self.assertEqual([row["sparsity"] for row in history], [0.25, 0.5])
        magnitude, taylor = SmallNetwork(), SmallNetwork()
        h1 = magnitude.train(x, y, epochs=3, batch_size=2, schedule={1: 0, 2: 0})
        h2 = taylor.train(x, y, epochs=3, batch_size=2, schedule={1: 0, 2: 0},
                          criterion="taylor")
        self.assertEqual(h1, h2)
        for a, b in zip((magnitude.w1, magnitude.w2), (taylor.w1, taylor.w2)):
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()

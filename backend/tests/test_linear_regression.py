import numpy as np
import pandas as pd

from backend.ml.linear_regression import LinearRegressionScratch


def test_gradient_matches_finite_difference():
    features = np.array([[1.0, 0.2], [1.0, -0.4], [1.0, 1.2]])
    targets = np.array([1.4, -0.2, 2.1])
    weights = np.array([0.3, -0.7])
    _, gradient = LinearRegressionScratch.loss_and_gradient(features, targets, weights)
    epsilon = 1e-6
    numeric = []
    for index in range(len(weights)):
        plus, minus = weights.copy(), weights.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        loss_plus = LinearRegressionScratch.loss_and_gradient(features, targets, plus)[0]
        loss_minus = LinearRegressionScratch.loss_and_gradient(features, targets, minus)[0]
        numeric.append((loss_plus - loss_minus) / (2 * epsilon))
    np.testing.assert_allclose(gradient, numeric, rtol=1e-5, atol=1e-6)


def test_gradient_descent_converges_and_round_trips(tmp_path):
    frame = pd.DataFrame({"kind": ["a", "a", "b", "b", "a", "b"], "x": [0, 1, 0, 1, 2, 2]})
    targets = np.array([2.0, 5.0, 4.0, 7.0, 8.0, 10.0])
    model = LinearRegressionScratch(learning_rate=0.04, epochs=3000, patience=400).fit(frame, targets, categorical=["kind"], numeric=["x"])
    assert model.training_loss[-1] < model.training_loss[0]
    np.testing.assert_allclose(model.predict(frame), targets, atol=0.1)
    path = tmp_path / "model.json"
    model.save(path)
    restored = LinearRegressionScratch.load(path)
    np.testing.assert_allclose(restored.predict(frame), model.predict(frame))
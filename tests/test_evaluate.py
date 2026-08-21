from loan_default.evaluate import evaluate_probabilities


def test_metrics_use_requested_threshold():
    metrics = evaluate_probabilities([0, 1, 1], [0.1, 0.4, 0.9], threshold=0.3)

    assert metrics["threshold"] == 0.3
    assert metrics["confusion_matrix"] == [[1, 0], [0, 2]]

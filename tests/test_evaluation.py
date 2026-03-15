from src.evaluation.evaluate_model import evaluate


def test_evaluate_function():

    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0]

    results = evaluate(y_true, y_pred)

    assert "accuracy" in results
    assert 0 <= results["accuracy"] <= 1
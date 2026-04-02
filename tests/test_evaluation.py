from src.evaluation.evaluate_model import evaluate


def test_evaluate_function():

    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0]

    results = evaluate(y_true, y_pred)

    assert "accuracy" in results
    assert 0 <= results["accuracy"] <= 1


def test_evaluate_supports_multiclass():

    y_true = [0, 1, 2, 1, 2, 0]
    y_pred = [0, 2, 2, 1, 0, 0]

    results = evaluate(y_true, y_pred)

    assert results["metric_average"] == "macro"
    assert len(results["confusion_matrix"]) == 3
    assert results["dataset_stats"]["num_classes"] == 3
    assert "class_counts" in results["dataset_stats"]

from learning import retrieval_metrics


def test_retrieval_metrics_at_five():
    result = retrieval_metrics(["10", "20", "30", "40", "50"], ["20", "50", "60"])
    assert result["precision_at_5"] == 2 / 5
    assert result["recall_at_5"] == 2 / 3
    assert result["mrr"] == 1 / 2


def test_retrieval_metrics_no_relevant_hit():
    result = retrieval_metrics(["1", "2"], ["9"], k=5)
    assert result["precision_at_5"] == 0
    assert result["recall_at_5"] == 0
    assert result["mrr"] == 0


def test_retrieval_metrics_first_result_is_relevant():
    result = retrieval_metrics(["7", "8", "9"], ["7"], k=5)
    assert result["precision_at_5"] == 1 / 5
    assert result["recall_at_5"] == 1
    assert result["mrr"] == 1

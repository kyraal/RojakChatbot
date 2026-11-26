import sys
from types import SimpleNamespace

import pandas as pd

from nlp_engine import ReviewAssistant


def make_reviews():
    rows = []
    samples = {
        1: ["poor stitching and damaged seams", "bad quality and torn fabric"],
        3: ["average material and ordinary fitting", "delivery was okay but not special"],
        5: ["excellent quality and comfortable fabric", "beautiful design and fast delivery"],
    }
    for rating, texts in samples.items():
        for index in range(12):
            rows.append({
                "review": texts[index % len(texts)] + f" customer item {index}",
                "review_id": str(len(rows) + 1),
                "rating": rating,
                "author": "tester",
                "sentiment": "negative" if rating <= 2 else "neutral" if rating == 3 else "positive",
            })
    return pd.DataFrame(rows)


def test_sentiment_analysis_reports_rating_distribution_and_held_out_metrics():
    assistant = ReviewAssistant(make_reviews())
    result = assistant.sentiment_analysis()

    assert result["total_reviews"] == 36
    assert result["distribution"]["positive"] == 12
    assert result["distribution"]["neutral"] == 12
    assert result["distribution"]["negative"] == 12
    assert result["evaluation"]["available"] is True
    assert 0 <= result["evaluation"]["accuracy"] <= 1
    assert 0 <= result["evaluation"]["macro_f1"] <= 1
    assert result["evaluation"]["test_examples"] > 0
    assert "proxy" in result["caveat"].lower()


def test_lda_returns_topics_with_keywords_and_examples():
    assistant = ReviewAssistant(make_reviews())
    result = assistant.lda_topics(n_topics=3)

    assert result["document_count"] == 36
    assert len(result["topics"]) == 3
    assert all(topic["keywords"] for topic in result["topics"])
    assert all("review_count" in topic for topic in result["topics"])


def test_pos_tag_empty_input_returns_empty_list():
    assistant = ReviewAssistant(make_reviews())
    assert assistant.pos_tag("   ") == []


def test_pos_tag_returns_token_tag_and_description(monkeypatch):
    class Tokenizer:
        def tokenize(self, text):
            return ["The", "fabric", "feels", "soft"]

    fake_nltk = SimpleNamespace(
        data=SimpleNamespace(find=lambda _path: True),
        tokenize=SimpleNamespace(TreebankWordTokenizer=Tokenizer),
        pos_tag=lambda tokens, lang="eng": list(zip(tokens, ["DT", "NN", "VBZ", "JJ"])),
    )
    monkeypatch.setitem(sys.modules, "nltk", fake_nltk)

    assistant = ReviewAssistant(make_reviews())
    tagged = assistant.pos_tag("The fabric feels soft")

    assert tagged[1]["token"] == "fabric"
    assert tagged[1]["tag"] == "NN"
    assert tagged[3]["description"] == "adjective"

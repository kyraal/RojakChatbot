import learning


def test_negation_is_preserved():
    text = learning.normalize_rojak("Kain ni tak panas, comfy tak?")
    assert "tak" in text
    assert "fabric" in text


def test_seed_classifier_recognises_common_rojak_questions():
    assert learning.predict_aspect("Kain ni panas tak?")["aspect"] == "fabric"
    assert learning.predict_aspect("Saiz dia ngam ke?")["aspect"] == "fit"
    assert learning.predict_aspect("Jahitan nampak sloppy ke?")["aspect"] == "quality"


def test_feedback_requires_human_approval(tmp_path, monkeypatch):
    monkeypatch.setattr(learning, "DB_PATH", tmp_path / "learning.db")
    feedback_id = learning.submit_feedback("parcel aku stuck", "It is delayed", "delivery")
    assert len(learning.feedback_rows("pending")) == 1
    assert len([x for x in learning.feedback_rows("approved")]) == 0
    learning.review_feedback(feedback_id, "approved", "delivery", "stuck", "delivery")
    assert learning.feedback_rows("approved")[0]["id"] == feedback_id
    assert learning.vocabulary()["stuck"] == "delivery"


def test_unapproved_correction_does_not_enter_training(tmp_path, monkeypatch):
    monkeypatch.setattr(learning, "DB_PATH", tmp_path / "learning.db")
    before, _ = learning.training_data()
    learning.submit_feedback("colour very cun", "Looks beautiful", "colour")
    after, _ = learning.training_data()
    assert len(after) == len(before)

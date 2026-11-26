"""Human-in-the-loop Rojak vocabulary and aspect learning.

Only approved corrections are used for vocabulary normalization and classifier training.
Customer corrections are never inserted into the review evidence corpus.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3
import csv
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "learning.db"
ASPECTS = ("fit", "fabric", "quality", "colour", "delivery")
MIN_CONFIDENCE = 0.38

# Small, transparent seed set. It is a starting point, not a substitute for
# collecting and manually annotating a larger representative Rojak dataset.
SEED_EXAMPLES = [
    ("saiz dia ngam tak", "fit"), ("cutting ni muat badan ke", "fit"),
    ("size chart accurate ke", "fit"), ("baju ni besar sangat", "fit"),
    ("is the fitting true to size", "fit"), ("ukuran pinggang berapa", "fit"),
    ("does it run small", "fit"), ("potongan dia loose ke", "fit"),
    ("kain ni panas tak", "fabric"), ("material lembut ke", "fabric"),
    ("is the fabric breathable", "fabric"), ("selesa pakai seharian tak", "fabric"),
    ("does the cloth feel hot", "fabric"), ("kain jenis apa", "fabric"),
    ("material ni itchy ke", "fabric"), ("tak panas bila outdoor", "fabric"),
    ("jahitan kemas ke", "quality"), ("stitching nampak sloppy", "quality"),
    ("ada benang terkeluar tak", "quality"), ("quality tahan lama ke", "quality"),
    ("baju senang koyak tak", "quality"), ("finishing dia neat ke", "quality"),
    ("ada defect atau lubang", "quality"), ("zipper kuat tak", "quality"),
    ("warna sama macam gambar", "colour"), ("colour dia lawa tak", "colour"),
    ("is the colour accurate", "colour"), ("warna luntur tak", "colour"),
    ("design ni cantik ke", "colour"), ("shade nampak macam photo tak", "colour"),
    ("warna dia terang sangat ke", "colour"), ("rupa sebenar macam mana", "colour"),
    ("parcel sampai bila", "delivery"), ("shipping cepat tak", "delivery"),
    ("courier lambat ke", "delivery"), ("berapa hari penghantaran", "delivery"),
    ("when will my parcel arrive", "delivery"), ("barang dah pos ke", "delivery"),
    ("delivery experience macam mana", "delivery"), ("order sampai lambat tak", "delivery"),
]

DEFAULT_VOCAB = {
    "ngam": "fit", "muat": "fit", "oversize": "fit", "ketat": "fit",
    "longgar": "fit", "true to size": "fit", "saiz": "fit",
    "bahang": "fabric", "berbahang": "fabric", "rimas": "fabric",
    "lembut": "fabric", "breathable": "fabric", "comfy": "fabric",
    "selesa": "fabric", "panas": "fabric",
    "sloppy": "quality", "kemas": "quality",
    "benang terkeluar": "quality", "rabak": "quality", "koyak": "quality",
    "lawa": "colour", "cun": "colour", "cantik": "colour",
    "pudar": "colour", "luntur": "colour",
    "lambat": "delivery", "slow": "delivery", "take ages": "delivery",
    "sampai": "delivery", "pos": "delivery",
}

def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with connect() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            correction TEXT NOT NULL,
            predicted_aspect TEXT,
            corrected_aspect TEXT,
            vocab_term TEXT,
            vocab_aspect TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            reviewed_at TEXT
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS vocabulary (
            term TEXT PRIMARY KEY COLLATE NOCASE,
            canonical_aspect TEXT NOT NULL,
            source_feedback_id INTEGER,
            approved_at TEXT NOT NULL
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS evaluation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            accuracy REAL NOT NULL,
            macro_f1 REAL NOT NULL,
            training_examples INTEGER NOT NULL,
            approved_corrections INTEGER NOT NULL,
            baseline_accuracy REAL,
            baseline_macro_f1 REAL,
            macro_precision REAL,
            macro_recall REAL
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS retrieval_evaluation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            precision_at_5 REAL NOT NULL,
            recall_at_5 REAL NOT NULL,
            mrr REAL NOT NULL,
            query_count INTEGER NOT NULL,
            results_json TEXT NOT NULL
        )""")
        # Migrate databases created by earlier versions of the learning loop.
        columns = {row[1] for row in db.execute("PRAGMA table_info(evaluation_runs)")}
        for name in ("baseline_accuracy", "baseline_macro_f1", "macro_precision", "macro_recall"):
            if name not in columns:
                db.execute(f"ALTER TABLE evaluation_runs ADD COLUMN {name} REAL")
        for term, aspect in DEFAULT_VOCAB.items():
            db.execute("INSERT OR IGNORE INTO vocabulary(term,canonical_aspect,approved_at) VALUES(?,?,?)",
                       (term, aspect, "seed"))

def vocabulary():
    init_db()
    with connect() as db:
        rows = db.execute("SELECT term,canonical_aspect FROM vocabulary ORDER BY length(term) DESC").fetchall()
    return {r["term"]: r["canonical_aspect"] for r in rows}

def _normalize_with_vocab(text, vocab):
    """Canonicalize phrases with one explicit vocabulary snapshot."""
    value = str(text or "").lower()
    for term, aspect in sorted(vocab.items(), key=lambda item: len(item[0]), reverse=True):
        value = re.sub(r"(?<!\\w)" + re.escape(term.lower()) + r"(?!\\w)",
                       " " + aspect + " ", value, flags=re.IGNORECASE)
    return re.sub(r"\\s+", " ", value).strip()


def normalize_rojak(text):
    """Canonicalize approved slang phrases while retaining negation words."""
    return _normalize_with_vocab(text, vocabulary())

def submit_feedback(question, correction, predicted_aspect=None):
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with connect() as db:
        cur = db.execute(
            "INSERT INTO feedback(question,correction,predicted_aspect,created_at) VALUES(?,?,?,?)",
            (question[:1000], correction[:2000], predicted_aspect if predicted_aspect in ASPECTS else None, now))
        return cur.lastrowid

def review_feedback(feedback_id, decision, corrected_aspect=None, vocab_term=None, vocab_aspect=None):
    init_db()
    if decision not in ("approved", "rejected"):
        raise ValueError("Decision must be approved or rejected")
    if decision == "approved" and corrected_aspect not in ASPECTS:
        raise ValueError("Choose a valid corrected aspect before approval")
    if vocab_term and vocab_aspect not in ASPECTS:
        raise ValueError("Choose a valid vocabulary aspect")
    now = datetime.now(timezone.utc).isoformat()
    with connect() as db:
        row = db.execute("SELECT * FROM feedback WHERE id=?", (int(feedback_id),)).fetchone()
        if row is None:
            raise ValueError("Feedback item not found")
        db.execute("""UPDATE feedback SET status=?,corrected_aspect=?,vocab_term=?,vocab_aspect=?,reviewed_at=?
                      WHERE id=?""",
                   (decision, corrected_aspect if decision == "approved" else None,
                    vocab_term.strip().lower() if vocab_term and decision == "approved" else None,
                    vocab_aspect if vocab_term and decision == "approved" else None, now, int(feedback_id)))
        if decision == "approved" and vocab_term:
            db.execute("""INSERT INTO vocabulary(term,canonical_aspect,source_feedback_id,approved_at)
                          VALUES(?,?,?,?) ON CONFLICT(term) DO UPDATE SET
                          canonical_aspect=excluded.canonical_aspect,
                          source_feedback_id=excluded.source_feedback_id,
                          approved_at=excluded.approved_at""",
                       (vocab_term.strip().lower(), vocab_aspect, int(feedback_id), now))
    return True

def feedback_rows(status=None):
    init_db()
    with connect() as db:
        if status in ("pending", "approved", "rejected"):
            rows = db.execute("SELECT * FROM feedback WHERE status=? ORDER BY id DESC", (status,)).fetchall()
        else:
            rows = db.execute("SELECT * FROM feedback ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]

def training_data():
    """Build training examples using one vocabulary/database read per batch."""
    init_db()
    vocab = vocabulary()
    X = [_normalize_with_vocab(question, vocab) for question, _ in SEED_EXAMPLES]
    y = [label for _, label in SEED_EXAMPLES]
    approved = [r for r in feedback_rows("approved") if r.get("corrected_aspect") in ASPECTS]
    X.extend(_normalize_with_vocab(r["question"], vocab) for r in approved)
    y.extend(r["corrected_aspect"] for r in approved)
    return X, y

def _new_model():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1,
                                 max_features=30000, sublinear_tf=True)),
    ])

def train_aspect_classifier():
    X, y = training_data()
    model = LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42)
    # Store the fitted feature extractor alongside the classifier for inference.
    features = _new_model()
    matrix = features.fit_transform(X)
    model.fit(matrix, y)
    return features, model, len(X)

def predict_aspect(question):
    features, model, count = train_aspect_classifier()
    vector = features.transform([normalize_rojak(question)])
    probabilities = model.predict_proba(vector)[0]
    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])
    label = str(model.classes_[index])
    ranked = np.argsort(probabilities)[::-1][:3]
    alternatives = [
        {"aspect": str(model.classes_[i]), "confidence": float(probabilities[i])}
        for i in ranked if int(i) != index
    ]
    if confidence < MIN_CONFIDENCE:
        return {"aspect": None, "confidence": confidence, "alternatives": [
            {"aspect": str(model.classes_[i]), "confidence": float(probabilities[i])} for i in ranked
        ], "training_examples": count}
    return {"aspect": label, "confidence": confidence, "alternatives": alternatives,
            "training_examples": count}

def load_test_examples():
    """Load the fixed, independently authored aspect test set."""
    path = ROOT / "data" / "aspect_test_set.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    examples = [(row["question"].strip(), row["aspect"].strip())
                for row in rows if row.get("question") and row.get("aspect")]
    if not examples or any(label not in ASPECTS for _, label in examples):
        raise ValueError("The aspect test set is empty or contains an invalid aspect label")
    return examples


def _score_model(X_train, y_train, X_test, y_test):
    features = _new_model()
    train_matrix = features.fit_transform(X_train)
    test_matrix = features.transform(X_test)
    model = LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42)
    model.fit(train_matrix, y_train)
    predictions = model.predict(test_matrix)
    labels = list(ASPECTS)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_precision": float(precision_score(y_test, predictions, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, predictions, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, predictions, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "report": classification_report(y_test, predictions, labels=labels, zero_division=0, output_dict=True),
    }


def _rule_based_aspect(question):
    """Transparent keyword baseline for the three-way model comparison."""
    text = normalize_rojak(question)
    rules = {
        "fit": ["fit", "size", "sizing", "saiz", "ukuran", "muat", "ngam", "ketat", "longgar", "besar", "kecil"],
        "fabric": ["fabric", "material", "kain", "panas", "bahang", "rimas", "lembut", "breathable", "selesa", "comfy"],
        "quality": ["quality", "kualiti", "jahitan", "stitch", "sewing", "sloppy", "kemas", "koyak", "rosak", "benang"],
        "colour": ["colour", "color", "warna", "cantik", "lawa", "cun", "pudar", "luntur", "design"],
        "delivery": ["delivery", "shipping", "penghantaran", "pos", "courier", "lambat", "sampai", "parcel"]
    }
    scores = {aspect: sum(1 for term in terms if term in text) for aspect, terms in rules.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] else None


def _classification_metrics(y_true, predictions):
    labels = list(ASPECTS)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_precision": float(precision_score(y_true, predictions, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, predictions, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, predictions, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=labels).tolist()
    }


def evaluate_classifier():
    """Compare seed-only baseline with approved-feedback training on one fixed test set."""
    test_examples = load_test_examples()
    y_test = [label for _, label in test_examples]

    # Each stage uses only the vocabulary available at that stage.
    baseline_vocab = DEFAULT_VOCAB.copy()
    baseline_X = [_normalize_with_vocab(q, baseline_vocab) for q, _ in SEED_EXAMPLES]
    baseline_test = [_normalize_with_vocab(q, baseline_vocab) for q, _ in test_examples]
    baseline_y = [label for _, label in SEED_EXAMPLES]
    baseline = _score_model(baseline_X, baseline_y, baseline_test, y_test)

    learned_vocab = vocabulary()
    trained_X = [_normalize_with_vocab(q, learned_vocab) for q, _ in SEED_EXAMPLES]
    trained_y = [label for _, label in SEED_EXAMPLES]
    approved = [row for row in feedback_rows("approved") if row.get("corrected_aspect") in ASPECTS]
    trained_X.extend(_normalize_with_vocab(row["question"], learned_vocab) for row in approved)
    trained_y.extend(row["corrected_aspect"] for row in approved)
    learned_test = [_normalize_with_vocab(q, learned_vocab) for q, _ in test_examples]
    learned = _score_model(trained_X, trained_y, learned_test, y_test)

    # Compare three approaches on the same held-out examples:
    # transparent rules, supervised TF-IDF + Logistic Regression, and a hybrid
    # that falls back to rules only when the classifier is below confidence threshold.
    feature_model = _new_model()
    train_matrix = feature_model.fit_transform(trained_X)
    test_matrix = feature_model.transform(learned_test)
    classifier = LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42)
    classifier.fit(train_matrix, trained_y)
    probabilities = classifier.predict_proba(test_matrix)
    classes = list(classifier.classes_)
    supervised_predictions = [classes[int(np.argmax(row))] for row in probabilities]
    rule_predictions = [_rule_based_aspect(q) or "quality" for q, _ in test_examples]
    hybrid_predictions = []
    for (question, _), probs, supervised in zip(test_examples, probabilities, supervised_predictions):
        confidence = float(np.max(probs))
        hybrid_predictions.append(supervised if confidence >= MIN_CONFIDENCE else (_rule_based_aspect(question) or supervised))
    three_models = {
        "rule_based": _classification_metrics(y_test, rule_predictions),
        "supervised": _classification_metrics(y_test, supervised_predictions),
        "hybrid": _classification_metrics(y_test, hybrid_predictions)
    }
    approved_count = sum(1 for row in feedback_rows("approved")
                         if row.get("corrected_aspect") in ASPECTS)

    result = {
        "three_models": three_models,
        "baseline": baseline,
        "learned": learned,
        "accuracy": learned["accuracy"],
        "macro_precision": learned["macro_precision"],
        "macro_recall": learned["macro_recall"],
        "macro_f1": learned["macro_f1"],
        "labels": list(ASPECTS),
        "confusion_matrix": learned["confusion_matrix"],
        "report": learned["report"],
        "training_examples": len(trained_X),
        "approved_corrections": approved_count,
        "test_examples": len(test_examples),
        "test_set": "data/aspect_test_set.csv",
    }
    with connect() as db:
        db.execute("""INSERT INTO evaluation_runs(
            created_at,accuracy,macro_f1,training_examples,approved_corrections,
            baseline_accuracy,baseline_macro_f1,macro_precision,macro_recall
        ) VALUES(?,?,?,?,?,?,?,?,?)""",
        (datetime.now(timezone.utc).isoformat(), learned["accuracy"], learned["macro_f1"],
         len(trained_X), approved_count, baseline["accuracy"], baseline["macro_f1"],
         learned["macro_precision"], learned["macro_recall"]))
    return result


def load_retrieval_test_examples():
    """Load manually labelled query-to-review relevance judgements."""
    path = ROOT / "data" / "retrieval_test_set.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    examples = []
    for row in rows:
        query = (row.get("query") or "").strip()
        relevant = [value.strip() for value in (row.get("relevant_review_ids") or "").split(";") if value.strip()]
        if query and relevant:
            examples.append({"query": query, "relevant_review_ids": sorted(set(relevant))})
    if not examples:
        raise ValueError("The retrieval test set is empty or has no relevance labels")
    return examples


def retrieval_metrics(retrieved_ids, relevant_ids, k=5):
    """Calculate fixed-k Precision, Recall and reciprocal rank for one query."""
    relevant = set(str(value) for value in relevant_ids)
    retrieved = [str(value) for value in retrieved_ids[:k]]
    hits = sum(1 for value in retrieved if value in relevant)
    precision = hits / float(k)  # Missing results count as non-relevant.
    recall = hits / float(len(relevant)) if relevant else 0.0
    reciprocal_rank = next((1.0 / rank for rank, value in enumerate(retrieved, 1)
                            if value in relevant), 0.0)
    return {"precision_at_5": precision, "recall_at_5": recall, "mrr": reciprocal_rank,
            "relevant_found": hits, "retrieved_count": len(retrieved)}


def evaluate_retrieval(assistant, top_k=5):
    """Evaluate top-k retrieval against a human-labelled, fixed relevance set."""
    if top_k != 5:
        raise ValueError("This dashboard currently reports fixed Precision@5 and Recall@5")
    examples = load_retrieval_test_examples()
    details = []
    for example in examples:
        hits = assistant.retrieve(example["query"], top_k=top_k)
        retrieved_ids = [str(hit.get("review_id", "")) for hit in hits]
        scores = retrieval_metrics(retrieved_ids, example["relevant_review_ids"], top_k)
        details.append({
            "query": example["query"],
            "relevant_review_ids": example["relevant_review_ids"],
            "retrieved_review_ids": retrieved_ids,
            **scores
        })
    count = len(details)
    averages = {
        key: float(sum(row[key] for row in details) / count)
        for key in ("precision_at_5", "recall_at_5", "mrr")
    }
    result = {
        **averages,
        "query_count": count,
        "top_k": top_k,
        "test_set": "data/retrieval_test_set.csv",
        "results": details,
        "note": "Relevance labels are manually assigned. Review IDs refer to the No. column in data/dataset.csv."
    }
    with connect() as db:
        db.execute("""INSERT INTO retrieval_evaluation_runs(
            created_at,precision_at_5,recall_at_5,mrr,query_count,results_json
        ) VALUES(?,?,?,?,?,?)""",
        (datetime.now(timezone.utc).isoformat(), averages["precision_at_5"],
         averages["recall_at_5"], averages["mrr"], count,
         json.dumps(details, ensure_ascii=False)))
    return result


def retrieval_evaluation_history(limit=10):
    init_db()
    with connect() as db:
        rows = db.execute(
            "SELECT id,created_at,precision_at_5,recall_at_5,mrr,query_count "
            "FROM retrieval_evaluation_runs ORDER BY id DESC LIMIT ?", (int(limit),)
        ).fetchall()
    return [dict(row) for row in rows]

def summary():
    """Return dashboard counters with aggregate SQL; do not rebuild training data."""
    init_db()
    with connect() as db:
        count_rows = db.execute(
            "SELECT status, COUNT(*) AS n FROM feedback GROUP BY status"
        ).fetchall()
        counts = {status: 0 for status in ("pending", "approved", "rejected")}
        counts.update({row["status"]: row["n"] for row in count_rows if row["status"] in counts})
        feedback_total = sum(counts.values())
        approved_training = db.execute(
            "SELECT COUNT(*) FROM feedback WHERE status='approved' AND corrected_aspect IN ("
            + ",".join("?" for _ in ASPECTS) + ")",
            tuple(ASPECTS)
        ).fetchone()[0]
        runs = [dict(r) for r in db.execute(
            "SELECT * FROM evaluation_runs ORDER BY id DESC LIMIT 20"
        )]
        vocab_count = db.execute("SELECT COUNT(*) FROM vocabulary").fetchone()[0]
    latest = runs[0] if runs else None
    return {"feedback_total": feedback_total, **counts, "vocabulary_size": vocab_count,
            "training_examples": len(SEED_EXAMPLES) + approved_training,
            "latest_evaluation": latest, "evaluation_history": runs}

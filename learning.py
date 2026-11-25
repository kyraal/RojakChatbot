"""Human-in-the-loop Rojak vocabulary and aspect learning.

Only approved corrections are used for vocabulary normalization and classifier training.
Customer corrections are never inserted into the review evidence corpus.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
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
            approved_corrections INTEGER NOT NULL
        )""")
        for term, aspect in DEFAULT_VOCAB.items():
            db.execute("INSERT OR IGNORE INTO vocabulary(term,canonical_aspect,approved_at) VALUES(?,?,?)",
                       (term, aspect, "seed"))

def vocabulary():
    init_db()
    with connect() as db:
        rows = db.execute("SELECT term,canonical_aspect FROM vocabulary ORDER BY length(term) DESC").fetchall()
    return {r["term"]: r["canonical_aspect"] for r in rows}

def normalize_rojak(text):
    """Canonicalize approved slang phrases while retaining negation words."""
    value = str(text or "").lower()
    for term, aspect in vocabulary().items():
        value = re.sub(r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)",
                       " " + aspect + " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip()
    return value

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
    init_db()
    X = [normalize_rojak(x) for x, _ in SEED_EXAMPLES]
    y = [label for _, label in SEED_EXAMPLES]
    approved = [r for r in feedback_rows("approved") if r.get("corrected_aspect") in ASPECTS]
    X.extend(normalize_rojak(r["question"]) for r in approved)
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
    if confidence < MIN_CONFIDENCE:
        return {"aspect": None, "confidence": confidence, "training_examples": count}
    return {"aspect": label, "confidence": confidence, "training_examples": count}

def evaluate_classifier():
    """Fixed seed holdout; approved examples are training-only and never enter test."""
    X = [normalize_rojak(x) for x, _ in SEED_EXAMPLES]
    y = [label for _, label in SEED_EXAMPLES]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, random_state=42, stratify=y)
    approved = [r for r in feedback_rows("approved") if r.get("corrected_aspect") in ASPECTS]
    X_train += [normalize_rojak(r["question"]) for r in approved]
    y_train += [r["corrected_aspect"] for r in approved]
    features = _new_model()
    train_matrix = features.fit_transform(X_train)
    test_matrix = features.transform(X_test)
    model = LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42)
    model.fit(train_matrix, y_train)
    pred = model.predict(test_matrix)
    labels = list(ASPECTS)
    accuracy = float(accuracy_score(y_test, pred))
    macro = float(f1_score(y_test, pred, labels=labels, average="macro", zero_division=0))
    report = classification_report(y_test, pred, labels=labels, zero_division=0, output_dict=True)
    result = {"accuracy": accuracy, "macro_f1": macro, "labels": labels,
              "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
              "report": report, "training_examples": len(X_train),
              "approved_corrections": len(approved), "test_examples": len(y_test)}
    with connect() as db:
        db.execute("""INSERT INTO evaluation_runs(created_at,accuracy,macro_f1,training_examples,approved_corrections)
                      VALUES(?,?,?,?,?)""",
                   (datetime.now(timezone.utc).isoformat(), accuracy, macro,
                    len(X_train), len(approved)))
    return result

def summary():
    init_db()
    rows = feedback_rows()
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in ("pending", "approved", "rejected")}
    X, _ = training_data()
    with connect() as db:
        runs = [dict(r) for r in db.execute("SELECT * FROM evaluation_runs ORDER BY id DESC LIMIT 20")]
        vocab_count = db.execute("SELECT COUNT(*) FROM vocabulary").fetchone()[0]
    latest = runs[0] if runs else None
    return {"feedback_total": len(rows), **counts, "vocabulary_size": vocab_count,
            "training_examples": len(X), "latest_evaluation": latest, "evaluation_history": runs}

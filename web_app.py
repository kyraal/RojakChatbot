"""Local HTML shop and review-grounded customer support API.

Run: python web_app.py
"""

from __future__ import annotations

from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from urllib.parse import urlparse

from nlp_engine import ReviewAssistant, load_reviews
import learning


ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "web"
DATA = ROOT / "data" / "dataset.csv"
FEEDBACK = ROOT / "web_feedback.jsonl"  # legacy; new submissions use SQLite

PRODUCTS = [
    {"id": "BM-101", "name": "Baju Kurung Classic", "colour": "Royal Blue & Gold", "price": 89, "tag": "BESTSELLER", "style": "navy"},
    {"id": "BM-102", "name": "Baju Kurung Modern", "colour": "Deep Burgundy", "price": 99, "tag": "NEW ARRIVAL", "style": "sage"},
    {"id": "BM-103", "name": "Women's Kurta Essential", "colour": "Crimson Red", "price": 79, "tag": "EVERYDAY", "style": "sand"},
]
ORDERS = [
    {"id": "RJ-261120-1042", "tracking": "MYX782641905", "product": "Baju Kurung Classic", "variant": "Royal Blue & Gold / M", "date": "20 Nov 2026", "status": "In transit", "eta": "25 Nov 2026", "total": 89, "steps": ["Order confirmed", "Packed", "With courier", "In transit"]},
    {"id": "RJ-241102-2078", "tracking": "MYX913578204", "product": "Women's Kurta Essential", "variant": "Crimson Red / L", "date": "02 Nov 2026", "status": "Delivered", "eta": "06 Nov 2026", "total": 79, "steps": ["Order confirmed", "Packed", "With courier", "Delivered"]},
]

ASPECTS = {
    "fit": {"label": "sizing and fit", "terms": ["size", "sizing", "saiz", "ukuran", "fit", "muat", "besar", "kecil", "ngam", "accurate"], "query": "saiz size ukuran fit muat besar kecil"},
    "fabric": {"label": "fabric and comfort", "terms": ["kain", "fabric", "material", "panas", "hot", "comfortable", "comfort", "selesa", "lembut", "cotton", "sejuk"], "query": "kain fabric material panas selesa lembut"},
    "quality": {"label": "stitching and quality", "terms": ["jahitan", "stitch", "sewing", "quality", "kualiti", "koyak", "rosak", "lubang", "hole", "kemas"], "query": "jahitan stitching quality kemas koyak"},
    "colour": {"label": "colour and appearance", "terms": ["colour", "color", "warna", "gambar", "picture", "cantik", "design", "rupa"], "query": "warna colour gambar cantik design"},
    "delivery": {"label": "delivery experience", "terms": ["delivery", "shipping", "penghantaran", "pos", "courier", "lambat", "cepat", "sampai"], "query": "delivery shipping penghantaran pos courier cepat lambat"},
}


def load_bot():
    if not DATA.exists():
        raise FileNotFoundError(f"Review dataset missing: {DATA}")
    return ReviewAssistant(load_reviews(DATA), normalizer=learning.normalize_rojak)


BOT = load_bot()


def matching_order(text):
    upper = text.upper()
    return next((o for o in ORDERS if o["id"] in upper or o["tracking"] in upper), None)


def detects_aspect(question):
    try:
        return learning.predict_aspect(question)["aspect"]
    except Exception:
        # Keep the app usable if model training fails; this is only a fallback.
        words = set(re.findall(r"\w+", question.lower()))
        for key, aspect in ASPECTS.items():
            if words.intersection(aspect["terms"]):
                return key
        return None

def product_answer(question, product_context=False):
    aspect_key = detects_aspect(question)
    if not aspect_key:
        return {"reply": "I can help with what customers said about fabric, stitching, colour and delivery. Which part would you like to know about?", "evidence": [], "type": "clarify"}
    aspect = ASPECTS[aspect_key]
    if product_context and aspect_key == "fit":
        return {"reply": "I don't have reliable sizing feedback for this specific women's cut yet. The available reviews are not tagged by style, so I wouldn't want to promise that its fit matches another garment. Please compare the garment measurements with your own before ordering.", "evidence": [], "type": "insufficient"}
    query = learning.normalize_rojak(question) + " " + aspect["query"]
    hits = BOT.retrieve(query, top_k=25)
    relevant = []
    seen = set()
    for hit in hits:
        review_words = set(re.findall(r"\w+", hit["review"].lower()))
        review_key = " ".join(hit["review"].lower().split())
        if review_key not in seen and review_words.intersection(aspect["terms"]) and hit["score"] >= 0.04:
            seen.add(review_key)
            relevant.append(hit)
        if len(relevant) == 5:
            break
    if not relevant:
        return {"reply": f"I couldn't find enough review evidence about {aspect['label']} to answer confidently. Try asking about another feature.", "evidence": [], "type": "insufficient"}
    ratings = Counter("positive" if h["rating"] and h["rating"] >= 4 else "negative" if h["rating"] and h["rating"] <= 2 else "mixed" for h in relevant)
    if ratings["positive"] and ratings["negative"]:
        overview = "The experiences are mixed."
    elif ratings["negative"] > ratings["positive"]:
        overview = "The retrieved reviews lean critical."
    elif ratings["positive"] > ratings["negative"]:
        overview = "The retrieved reviews lean positive."
    else:
        overview = "The retrieved reviews do not show a clear consensus."
    combined = " ".join(h["review"].lower() for h in relevant)
    if aspect_key == "fabric" and re.search(r"\b(panas|hot)\b", question.lower()):
        detail = "Some buyers specifically describe the fabric as comfortable or not hot." if re.search(r"tak panas|not hot|sejuk|serap peluh", combined) else "The reviews mention fabric and comfort, but don't establish clearly whether it feels hot."
    elif aspect_key == "fit":
        detail = "Some buyers say the size matches the chart, while fit may differ between wearers." if re.search(r"sama.*carta|size accurate|saiz.*betul", combined) else "Buyers' fit experiences vary, so check your measurements against a size chart."
    elif aspect_key == "quality":
        detail = "There are comments about neat stitching, and some report quality issues." if re.search(r"jahitan kemas|stitch.*neat", combined) and re.search(r"uneven|koyak|rosak|tak kemas", combined) else "Buyers comment on stitching and finish, with varying experiences."
    elif aspect_key == "delivery":
        detail = "These are past delivery experiences; I can't guarantee a future delivery time."
    else:
        detail = "The comments describe how the item looked to those buyers."
    quote = " ".join(relevant[0]["review"].split())[:170]
    reply = f"About {aspect['label']}: {detail} {overview} For example, one customer wrote: “{quote}”. You can open the original reviews below."
    return {"reply": reply, "evidence": [{"review": h["review"], "rating": h["rating"]} for h in relevant[:3]], "type": "reviews"}


def respond(text, context=None):
    question = " ".join(str(text).strip().split())[:1000]
    if not question:
        return {"reply": "Send me a question and I'll help.", "evidence": [], "type": "clarify"}
    wording = re.sub(r"^Order ID:\s*[A-Z0-9-]+\s*[—-]\s*", "", question, flags=re.I)
    lower = wording.lower()
    context = context if isinstance(context, dict) else {}
    order = matching_order(question)
    if not order and context.get("kind") == "order":
        order = matching_order(str(context.get("id", "")))
    order_terms = re.search(r"\b(order|parcel|tracking|track|pesanan|status)\b|barang.{0,16}(mana|sampai)", lower)
    if order and (order_terms or not wording or lower.strip().upper() in (order["id"], order["tracking"])):
        return {"reply": f"I found your order {order['id']} for {order['product']} ({order['variant']}). It is {order['status'].lower()}. Parcel code: {order['tracking']}. Estimated delivery/result: {order['eta']}. What else would you like to know?", "evidence": [], "type": "order"}
    if order_terms and not detects_aspect(wording):
        return {"reply": "Sure—I can check your order. Attach it from My Orders or paste its order ID or parcel code here, and tell me what you'd like to know.", "evidence": [], "type": "order_prompt"}
    if re.search(r"\b(refund|return|cancel|pulangan|pulang|batalkan|bayaran balik)\b", lower):
        return {"reply": "I do not have a verified store return policy in the review dataset, so I cannot confirm the return window or conditions. Please check the seller official policy or contact the store directly.", "evidence": [], "type": "insufficient"}
    if re.search(r"\b(hi|hello|hey|hai|helo|assalamualaikum)\b", lower) and len(lower.split()) <= 4:
        return {"reply": "Hi! Happy to help. Ask me about fabric, stitching, delivery reviews or an order. You can mix English and BM—no problem.", "evidence": [], "type": "greeting"}
    if re.search(r"\b(thank|thanks|terima kasih|tq)\b", lower):
        return {"reply": "You're welcome! Let me know if you'd like to check another product detail or an order.", "evidence": [], "type": "social"}
    return product_answer(wording, context.get("kind") == "product")


class Handler(BaseHTTPRequestHandler):
    def _json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/catalog":
            return self._json(200, {"products": PRODUCTS, "orders": ORDERS, "review_count": len(BOT.df)})
        if path == "/api/health":
            return self._json(200, {"ok": True, "reviews": len(BOT.df)})
        if path == "/api/admin/summary":
            return self._json(200, learning.summary())
        if path == "/api/admin/feedback":
            return self._json(200, {"feedback": learning.feedback_rows()})
        if path == "/admin":
            path = "/admin.html"
        target = (PUBLIC / (path.lstrip("/") or "index.html")).resolve()
        if PUBLIC.resolve() not in target.parents or not target.is_file():
            self.send_error(404)
            return
        mime = {".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".svg": "image/svg+xml"}.get(target.suffix, "application/octet-stream")
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{mime}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        if size > 100_000:
            return self._json(413, {"error": "Request too large"})
        try:
            payload = json.loads(self.rfile.read(size))
        except (ValueError, UnicodeDecodeError):
            return self._json(400, {"error": "Invalid JSON"})
        if self.path == "/api/chat":
            return self._json(200, respond(payload.get("message", ""), payload.get("context")))
        if self.path == "/api/feedback":
            question = str(payload.get("question", "")).strip()[:1000]
            correction = str(payload.get("correction", "")).strip()[:2000]
            if not question or not correction:
                return self._json(400, {"error": "Question and correction are required"})
            feedback_id = learning.submit_feedback(question, correction, detects_aspect(question))
            return self._json(200, {"saved": True, "id": feedback_id, "status": "pending", "note": "Saved for human review; not used for learning until approved"})
        if self.path == "/api/admin/review":
            try:
                learning.review_feedback(payload.get("id"), payload.get("decision"), payload.get("corrected_aspect"), payload.get("vocab_term"), payload.get("vocab_aspect"))
                if payload.get("decision") == "approved":
                    global BOT
                    BOT = load_bot()  # rebuild word + character TF-IDF matrices with approved vocabulary
                return self._json(200, {"saved": True, "summary": learning.summary()})
            except (ValueError, TypeError) as exc:
                return self._json(400, {"error": str(exc)})
        if self.path == "/api/admin/evaluate":
            try:
                return self._json(200, learning.evaluate_classifier())
            except Exception as exc:
                return self._json(500, {"error": str(exc)})
        return self._json(404, {"error": "Unknown endpoint"})


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("Rojak Wardrobe running at http://127.0.0.1:8765", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()

"""Local HTML shop and review-grounded customer support API.

Run: python web_app.py
"""

from __future__ import annotations

from collections import Counter
import base64
import binascii
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import io
import time
from pathlib import Path
import re
import threading
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
WHISPER_MODEL = None
WHISPER_PROCESSOR = None
WHISPER_PIPELINE = None
WHISPER_LOCK = threading.Lock()


def transcribe_with_whisper(audio_b64):
    """Transcribe Malaysian Malay, English and Manglish with Mesolitica Whisper."""
    global WHISPER_MODEL, WHISPER_PROCESSOR, WHISPER_PIPELINE
    if not isinstance(audio_b64, str) or not audio_b64:
        raise ValueError("No recorded audio was provided")
    try:
        audio_bytes = base64.b64decode(audio_b64, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("The audio upload was not valid base64 data")
    if len(audio_bytes) > 15 * 1024 * 1024:
        raise ValueError("Recording is too large. Please record a shorter question (under 15 MB).")
    if len(audio_bytes) < 100:
        raise ValueError("The recording is empty or too short")

    # PyAV decodes the browser's WebM/Opus recording to the 16 kHz waveform
    # expected by Whisper's feature extractor.
    import av
    import numpy as np
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    with WHISPER_LOCK:
        if WHISPER_PIPELINE is None:
            model_id = "mesolitica/malaysian-whisper-base"
            WHISPER_PROCESSOR = AutoProcessor.from_pretrained(model_id)
            WHISPER_MODEL = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            WHISPER_MODEL.eval()
            WHISPER_PIPELINE = pipeline(
                "automatic-speech-recognition",
                model=WHISPER_MODEL,
                tokenizer=WHISPER_PROCESSOR.tokenizer,
                feature_extractor=WHISPER_PROCESSOR.feature_extractor,
                device=-1,
                chunk_length_s=30,
                stride_length_s=5
            )

        container = av.open(io.BytesIO(audio_bytes))
        audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
        if audio_stream is None:
            raise ValueError("No audio track was found in the recording.")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        frames = []
        for frame in container.decode(audio_stream):
            converted = resampler.resample(frame)
            if converted:
                frames.extend(converted if isinstance(converted, list) else [converted])
        tail = resampler.resample(None)
        if tail:
            frames.extend(tail if isinstance(tail, list) else [tail])
        if not frames:
            raise ValueError("No speech audio could be decoded. Please record again.")
        waveform = np.concatenate([frame.to_ndarray().reshape(-1) for frame in frames]).astype(np.float32)
        if waveform.size < 1600:
            raise ValueError("The recording is too short. Please speak for at least one second.")

        # No language is forced: Whisper performs its own language detection.
        result = WHISPER_PIPELINE(waveform, generate_kwargs={"task": "transcribe"})
        transcript = (result.get("text") or "").strip()

    if not transcript:
        raise ValueError("Whisper could not detect clear speech. Please try again.")
    return {"text": transcript, "language": "auto-detected",
            "model": "mesolitica/malaysian-whisper-base"}


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


def _respond_base(text, context=None):
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



def detect_language(text):
    """Lightweight language-mode detection for English, BM and mixed Rojak."""
    words = set(re.findall(r"[a-z]+", str(text).lower()))
    bm_markers = {"saya","aku","awak","anda","dia","kita","kami","tak","tidak","ke","kah","lah","kan","ni","tu","dengan","untuk","boleh","berapa","macam","mana","kenapa","apa","bila","kain","saiz","warna","jahitan","penghantaran","panas","selesa","cantik","muat","ngam","lawa","sampai","lambat","takut"}
    en_markers = {"the","is","are","was","were","what","how","why","when","where","does","do","can","could","would","should","my","your","this","that","size","fabric","colour","color","stitching","delivery","comfortable","hot","good","bad","same","picture","order","please","i","you"}
    bm = len(words & bm_markers)
    en = len(words & en_markers)
    if bm and en:
        return "rojak"
    if bm:
        return "ms"
    return "en"


def localize_reply(reply, language):
    """Translate assistant-authored templates while retaining original review quotations."""
    if language == "en":
        return reply
    order_match = re.fullmatch(
        r"I found your order (.+?) for (.+?) \((.+?)\)\. It is (.+?)\. Parcel code: (.+?)\. Estimated delivery/result: (.+?)\. What else would you like to know\?",
        reply
    )
    if order_match:
        order_id, product, variant, status, parcel, eta = order_match.groups()
        if language == "ms":
            return f"Saya jumpa pesanan {order_id} untuk {product} ({variant}). Statusnya {status}. Kod bungkusan: {parcel}. Anggaran penghantaran: {eta}. Apa lagi yang anda ingin tahu?"
        return f"I found order {order_id} for {product} ({variant}). Status dia {status}. Parcel code: {parcel}. Estimated delivery: {eta}. Anything else you nak tahu?"
    translations = {
        "About sizing and fit:": ("Tentang saiz dan kesesuaian:", "Pasal sizing and fit:"),
        "About fabric and comfort:": ("Tentang kain dan keselesaan:", "Pasal fabric and comfort:"),
        "About stitching and quality:": ("Tentang jahitan dan kualiti:", "Pasal stitching and quality:"),
        "About colour and appearance:": ("Tentang warna dan penampilan:", "Pasal colour and appearance:"),
        "About delivery experience:": ("Tentang pengalaman penghantaran:", "Pasal delivery experience:"),
        "The experiences are mixed.": ("Pengalaman pembeli bercampur-campur.", "Buyer experience memang mixed."),
        "The retrieved reviews lean critical.": ("Ulasan yang ditemui lebih banyak menyatakan pengalaman kurang memuaskan.", "Reviews yang dijumpai lebih banyak negative."),
        "The retrieved reviews lean positive.": ("Ulasan yang ditemui lebih cenderung positif.", "Reviews yang dijumpai mostly positive."),
        "The retrieved reviews do not show a clear consensus.": ("Ulasan yang ditemui tidak menunjukkan satu kesimpulan yang jelas.", "Reviews yang dijumpai tak menunjukkan clear consensus."),
        "Some buyers specifically describe the fabric as comfortable or not hot.": ("Sesetengah pembeli menyatakan kain selesa dan tidak panas.", "Some buyers kata kain selesa and tak panas."),
        "The reviews mention fabric and comfort, but don't establish clearly whether it feels hot.": ("Ulasan menyebut tentang kain dan keselesaan, tetapi tidak mengesahkan sama ada kain terasa panas.", "Reviews mention fabric and comfort, tapi tak confirm sama ada kain panas."),
        "Some buyers say the size matches the chart, while fit may differ between wearers.": ("Sesetengah pembeli mengatakan saiz menepati carta, tetapi kesesuaian mungkin berbeza mengikut pemakai.", "Some buyers kata size ikut chart, tapi fit boleh berbeza ikut orang."),
        "Buyers' fit experiences vary, so check your measurements against a size chart.": ("Pengalaman tentang saiz berbeza, jadi bandingkan ukuran anda dengan carta saiz.", "Fit experience berbeza, so check measurements dengan size chart."),
        "There are comments about neat stitching, and some report quality issues.": ("Ada pembeli memuji jahitan yang kemas, manakala sesetengahnya melaporkan isu kualiti.", "Ada yang kata stitching kemas, tapi some buyers report quality issues."),
        "Buyers comment on stitching and finish, with varying experiences.": ("Pembeli memberikan pandangan berbeza tentang jahitan dan kemasan.", "Buyer feedback tentang stitching and finishing agak mixed."),
        "These are past delivery experiences; I can't guarantee a future delivery time.": ("Ini ialah pengalaman penghantaran terdahulu; saya tidak dapat menjamin tempoh penghantaran akan datang.", "Ini past delivery experience; I can't guarantee next delivery time."),
        "The comments describe how the item looked to those buyers.": ("Ulasan ini menerangkan rupa produk menurut pembeli.", "Comments ni describe how the item looked to buyers."),
        "For example, one customer wrote:": ("Sebagai contoh, seorang pelanggan menulis:", "For example, one customer wrote:"),
        "You can open the original reviews below.": ("Anda boleh melihat ulasan asal di bawah.", "You can check original reviews kat bawah."),
        "Hi! Happy to help. Ask me about fabric, stitching, delivery reviews or an order. You can mix English and BM—no problem.": ("Hai! Saya boleh bantu. Tanya saya tentang kain, jahitan, penghantaran atau pesanan anda.", "Hi! Boleh je tanya pasal fabric, stitching, delivery atau order. English and BM pun okay."),
        "You're welcome! Let me know if you'd like to check another product detail or an order.": ("Sama-sama! Beritahu saya jika anda ingin menyemak butiran produk atau pesanan lain.", "You're welcome! Kalau nak check product detail atau order lain, just tanya ya."),
        "Send me a question and I'll help.": ("Hantarkan soalan anda dan saya akan bantu.", "Send je your question, nanti saya bantu."),
        "I can help with what customers said about fabric, stitching, colour and delivery. Which part would you like to know about?": ("Saya boleh bantu tentang komen pelanggan mengenai kain, jahitan, warna dan penghantaran. Bahagian mana yang anda ingin tahu?", "I boleh bantu pasal fabric, stitching, colour and delivery. Which part you nak tahu?"),
        "I do not have a verified store return policy in the review dataset, so I cannot confirm the return window or conditions. Please check the seller official policy or contact the store directly.": ("Dataset ulasan tidak mengandungi polisi pemulangan yang disahkan. Sila semak polisi rasmi penjual atau hubungi kedai secara terus.", "Dataset reviews tak ada verified return policy, so please check seller's official policy atau contact kedai terus."),
        "I don't have reliable sizing feedback for this specific women's cut yet. The available reviews are not tagged by style, so I wouldn't want to promise that its fit matches another garment. Please compare the garment measurements with your own before ordering.": ("Saya belum mempunyai maklum balas saiz yang boleh dipercayai untuk potongan ini. Ulasan yang tersedia tidak diasingkan mengikut gaya, jadi sila bandingkan ukuran pakaian dengan ukuran anda sebelum membuat pesanan.", "I tak ada sizing feedback yang reliable untuk cut ni lagi. Reviews tak diasingkan ikut style, so please compare garment measurements dengan ukuran sendiri sebelum order."),
        "Sure—I can check your order. Attach it from My Orders or paste its order ID or parcel code here, and tell me what you'd like to know.": ("Boleh, saya boleh semak pesanan anda. Lampirkan pesanan melalui My Orders atau tampal ID pesanan atau kod bungkusan di sini.", "Sure, I boleh check order you. Attach dari My Orders atau paste order ID / parcel code kat sini."),
        "I found your order ": ("Saya jumpa pesanan anda ", "I found order you "),

        "Sorry, I couldn't connect to the shop assistant. Please try again.": ("Maaf, saya tidak dapat menyambung kepada pembantu kedai. Sila cuba lagi.", "Sorry, connection problem tadi. Please try again ya.")
    }
    for source, versions in translations.items():
        reply = reply.replace(source, versions[0] if language == "ms" else versions[1])
    if language == "ms":
        reply = re.sub(r"I couldn't find enough review evidence about (.*?) to answer confidently\. Try asking about another feature\.",
                       r"Saya tidak menemui ulasan yang mencukupi tentang \1 untuk menjawab dengan yakin. Cuba tanya tentang ciri lain.", reply)
    else:
        reply = re.sub(r"I couldn't find enough review evidence about (.*?) to answer confidently\. Try asking about another feature\.",
                       r"I tak jumpa cukup review evidence tentang \1 untuk jawab dengan yakin. Maybe ask about another feature?", reply)
    return reply


def respond(text, context=None):
    question = " ".join(str(text).strip().split())[:1000]
    language = detect_language(question)
    result = _respond_base(question, context)
    result["reply"] = localize_reply(result.get("reply", ""), language)
    result["language"] = language
    if question and result.get("type") == "reviews":
        try:
            prediction = learning.predict_aspect(question)
            alternatives = prediction.get("alternatives", [])
            result["suggestions"] = [
                {"aspect": item["aspect"], "label": ASPECTS[item["aspect"]]["label"]}
                for item in alternatives[:2] if item["aspect"] in ASPECTS
            ]
        except Exception:
            result["suggestions"] = []
    else:
        result["suggestions"] = []
    return result


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
        if path == "/api/admin/sentiment":
            return self._json(200, BOT.sentiment_analysis())
        if path == "/api/admin/topics":
            return self._json(200, BOT.lda_topics())
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
        limit = 20 * 1024 * 1024 if self.path == "/api/transcribe" else 100_000
        if size > limit:
            return self._json(413, {"error": "Request too large"})
        try:
            payload = json.loads(self.rfile.read(size))
        except (ValueError, UnicodeDecodeError):
            return self._json(400, {"error": "Invalid JSON"})
        if self.path == "/api/transcribe":
            try:
                started = time.perf_counter()
                result = transcribe_with_whisper(payload.get("audio", ""))
                result["processing_time_ms"] = round((time.perf_counter() - started) * 1000, 1)
                return self._json(200, result)
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})
            except ImportError:
                return self._json(503, {"error": "Whisper is not installed yet. Run: pip install -r requirements.txt"})
            except Exception as exc:
                return self._json(500, {"error": "Whisper transcription failed: " + str(exc)})
        if self.path == "/api/chat":
            started = time.perf_counter()
            result = respond(payload.get("message", ""), payload.get("context"))
            result["response_time_ms"] = round((time.perf_counter() - started) * 1000, 1)
            return self._json(200, result)
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
        if self.path == "/api/admin/evaluate-retrieval":
            try:
                return self._json(200, {
                    **learning.evaluate_retrieval(BOT),
                    "history": learning.retrieval_evaluation_history()
                })
            except Exception as exc:
                return self._json(500, {"error": str(exc)})
        if self.path == "/api/admin/evaluate":
            try:
                return self._json(200, learning.evaluate_classifier())
            except Exception as exc:
                return self._json(500, {"error": str(exc)})
        if self.path == "/api/admin/pos":
            sample = str(payload.get("text", "")).strip()[:500]
            if not sample:
                return self._json(400, {"error": "Enter a sentence to tag."})
            tagged = BOT.pos_tag(sample)
            if isinstance(tagged, dict) and tagged.get("error"):
                return self._json(503, tagged)
            return self._json(200, {"text": sample, "tokens": tagged,
                                    "note": "NLTK's pretrained POS tagger is English-centric; BM/Rojak tags are exploratory, not Malay-specific grammatical analysis."})
        return self._json(404, {"error": "Unknown endpoint"})


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("Rojak Wardrobe running at http://127.0.0.1:8765", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()

# Rojak — Malaysian Rojak Product Enquiry Assistant (Initial Build)

A Streamlit prototype for the NLP assignment. Users select the Baju Melayu collection represented by Pheiying's Shopee review dataset, ask questions in English/BM/Rojak, retrieve relevant reviews using word- and character-level TF-IDF + cosine similarity, and receive a concise English evidence-based summary.

## Dataset setup

Download `dataset.csv` from:

https://github.com/pheiying/Sentiment-Analysis-on-Out-Of-Vocabulary-OOV-Malaysia-Rojak-Language/blob/main/Dataset/dataset.csv

Save it as `data/dataset.csv` in this project. The app detects common column names; the Pheiying file uses `No.`, `Author`, `Rating`, `Review`.

**Dataset scope:** this source represents reviews for W Baju Melayu on Shopee. The CSV does not include a product ID/category field, so the first build intentionally offers one collection rather than inventing multiple product listings.

## Run locally (macOS / Windows)

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\\Scripts\\activate    # Windows PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501).

## Main user flow

1. Select **Baju Melayu Collection**.
2. Choose a suggested prompt or type a question in English, BM or Rojak.
3. Optional: record a voice question; the local Mesolitica Malaysian Whisper backend attempts automatic language detection and transcribes English, Bahasa Melayu and mixed speech. Code-switching accuracy should be tested with real recordings.
4. Preprocessing cleans text and retains useful informal spellings.
5. Reviews are ranked using a blend of word n-gram and character n-gram TF-IDF cosine similarity.
6. The assistant summarises the retrieved themes in English and shows the original review evidence.
7. Users can submit corrections in Feedback & Learning.

## NLP / assignment coverage

| Requirement / syllabus | Initial implementation | Notes |
|---|---|---|
| Speech-to-text | Optional microphone + Mesolitica Malaysian Whisper (`mesolitica/malaysian-whisper-base`) | Requires microphone permission and internet; code-switching can be imperfect. |
| Slang / informal language | Character n-gram TF-IDF retrieval + regex normalisation | No manually hardcoded slang dictionary. This handles spelling variation statistically, not guaranteed slang understanding. |
| Human correction | Feedback form writes `corrections.csv` | Saved feedback is not automatically a verified fact or automatically retrained in this build. |
| Knowledge-based answers | Review corpus retrieval + cosine similarity + evidence display | Answers are bounded by the customer reviews, not seller policy or live order data. |
| REGEX / preprocessing | URL, mention, punctuation, repeated-character and whitespace handling | Preserves word content and emoji-independent text. |
| Morphology | Optional Sastrawi stemming demo | Malay stemming only; not applied to retrieval by default to preserve original forms. |
| POS tagging | NLTK POS demo | Primarily English; code-mixed accuracy limited. |
| Semantic analysis / WSD | NLTK Lesk demo | Demonstration only; weak on Rojak and short contexts. |
| TF-IDF | Word unigrams/bigrams + character n-grams | Core retrieval method. |
| Sentiment analysis | Logistic Regression with word/char TF-IDF | Additional feature. Labels are derived from star ratings (1–2 negative, 3 neutral, 4–5 positive), so they are weak/proxy labels. |
| Evaluation | Stratified hold-out split, accuracy, precision, recall, F1, confusion matrix | Run from Feedback & Learning. Small dataset means results may vary by split. |
| LDA | Topic modelling tab | Exploratory review themes. |
| Word2Vec | Train-on-click demo | Exploratory embeddings; small corpus. |
| Markov NLG | Corpus-based demo | Demonstration only; never used as factual answer generation. |

## Important methodological limitations

- The dataset is a single product-collection review corpus, not a complete e-commerce FAQ database.
- The app cannot verify specifications, stock, prices, return policy, refunds or order tracking. It should not invent these answers.
- The English summary in this prototype is an evidence-grounded template based on retrieved review themes, not a large language model. Original reviews are shown so users can inspect the source.
- Sentiment labels are inferred from star ratings and may disagree with the actual sentiment expressed in a review.
- The correction form stores user feedback locally. A future version should add a controlled correction-review and retraining workflow, keeping a fixed untouched test set to prevent leakage.
- POS, Lesk WSD, LDA, Word2Vec and Markov are supporting syllabus demonstrations, not all part of the live answering path.

## Files

- `app.py` — Streamlit interface and navigation
- `nlp_engine.py` — preprocessing, retrieval, sentiment evaluation and syllabus demonstrations
- `data/dataset.csv` — place the original dataset here
- `corrections.csv` — created locally when users submit feedback (gitignored)

## Testing suggestions

Test questions such as:
- `Kain dia panas tak?`
- `Is the fabric comfortable for kids?`
- `Saiz dia accurate ke?`
- `Any complaints about jahitan?`
- `Warna sama macam gambar tak?`
- `Delivery cepat tak?`

Also test out-of-scope questions such as `Can I get a refund?` or `Where is my parcel?`. The assistant should return insufficient evidence rather than inventing seller policies or order information.


## Rojak learning-loop upgrade (working branch)

The HTML storefront is the interactive demo for the human-in-the-loop workflow:

```bash
python web_app.py
```

Open `http://127.0.0.1:8765` for the shop and `http://127.0.0.1:8765/admin` for the local learning dashboard.

### What is implemented

- **Rojak normalization:** a transparent vocabulary table maps approved informal words and phrases to aspect concepts. The normalizer preserves negation words such as `tak` and `not`.
- **Supervised aspect classification:** word- and character-level TF-IDF features feed a Logistic Regression classifier for fit, fabric, quality, colour and delivery. Low-confidence predictions fall back to clarification / the existing aspect fallback.
- **Human approval:** chat corrections enter a SQLite queue as pending. Only an explicit admin approval can add a corrected aspect example or vocabulary mapping.
- **Retrieval integration:** incoming questions and the review index use the same approved vocabulary normalization. After an approval, the word and character TF-IDF matrices are rebuilt from the original review corpus. The original review text remains the evidence shown to users.
- **Learning dashboard:** pending / approved / rejected counts, vocabulary size, training-example count and classifier evaluation history.
- **Sentiment analysis:** review ratings are converted to proxy labels (1–2 negative, 3 neutral, 4–5 positive). A word + character TF-IDF Logistic Regression model is evaluated on a held-out split; the dashboard shows the rating distribution and classification metrics. Overall-sentiment questions can be answered from the dataset-wide rating distribution.
- **LDA topic modelling:** Latent Dirichlet Allocation discovers recurring word co-occurrence patterns in the review corpus. The dashboard displays topic keywords, the number of assigned reviews and sample excerpts. Topics require human interpretation and are not verified product categories.
- **POS tagging:** an NLTK Penn Treebank tagger demo identifies grammatical roles in a user-entered sentence. Its pretrained model is English-centric, so BM/Rojak tags are exploratory and may be inaccurate.
- **Speech input:** the browser records audio with MediaRecorder and sends it to a local faster-whisper model; language is detected automatically, so users do not select BM/English/Rojak. Users can review/edit the transcript before sending. First transcription downloads the selected Whisper model; default is `small` on CPU with INT8. Set `WHISPER_MODEL_SIZE=tiny` for a faster, less accurate option.

### Important data and evaluation safeguards

- A correction is not a review and is never added to the customer-review evidence corpus.
- Approval of a correction is a human annotation for the aspect classifier; an optional vocabulary phrase is a separate explicit mapping.
- The classifier starts with a small hand-written seed set. The dashboard compares a seed-only baseline with a model trained on approved corrections using the same separately labelled test set (`data/aspect_test_set.csv`). Test examples are never used for training. This small manually authored set is not representative of all Malaysian Rojak; expand and independently review it before making broad performance claims.
- The learning database is `data/learning.db` and is local to the machine running the Python server. This workflow is intended for a local classroom demonstration; the admin endpoint has no authentication and should not be exposed publicly as-is.
- The HTML storefront's products and orders are simulated demo records. The review dataset does not establish those catalogue facts. Reviews are not tagged to the three individual storefront products, so review insights apply to the available collection rather than a verified individual style. The assistant must not present a simulated return policy as verified seller policy.
- The Streamlit app remains available as the original syllabus/demo interface; the new SQLite approval workflow is currently connected to the HTML/Python server.

### Suggested evaluation before submission

1. Review and expand `data/aspect_test_set.csv` with independently labelled English, BM, mixed-language, spelling-variant and negation examples.
2. Keep the test set frozen and outside training. Record the seed-only baseline and the model after approved corrections using accuracy, macro precision/recall/F1 and confusion matrix.
3. Collect classmate corrections, approve only valid annotations, then rerun the exact same test set.
4. For retrieval, manually mark relevant source reviews for each query and report Precision@5, Recall@5 and MRR. Do not use classifier accuracy as a substitute for retrieval quality.
5. For voice, test the same recordings in BM and English, manually transcribe references, and compare Word Error Rate (WER), latency and user preference across supported browsers.
6. Report actual sample sizes and scores from your test runs. Do not invent improvement if the metrics do not improve.



### Multilingual speech-to-text setup

Install the updated dependencies:

```bash
pip install -r requirements.txt
```

The HTML chat uses the locally loaded `mesolitica/malaysian-whisper-base` Transformers model. It was fine-tuned for Malaysian Malay (including local Malay) and English (including Manglish). The first voice transcription downloads the model; later recordings reuse it while the Python server remains running. Audio is resampled to 16 kHz and chunked for recordings longer than 30 seconds. Language is not forced, so Whisper attempts automatic detection. No browser language selector or Whisper API key is required. Mixed BM-English code-switching still needs validation with real recordings.

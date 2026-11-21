# 🏨 International Hotel Booking Analytics & Graph-RAG Travel Assistant

A university team project (German University in Cairo, ACL course, team of 4, Oct–Dec 2025) built in two phases on the same Kaggle hotel-booking dataset (`international-hotel-booking-analytics`: hotels, users, reviews):

| Phase | Question | What we built | Folder |
|---|---|---|---|
| **1. Machine learning** | Can a hotel's world region be predicted from how guests rated it? | Cleaning + EDA pipeline, five model experiments, a data-leakage investigation, SHAP/LIME explanations | [`phase1_ml/`](phase1_ml) |
| **2. Graph-RAG assistant** | Can travellers ask hotel questions in plain English? | Neo4j knowledge graph, intent + entity extraction, Cypher and FAISS retrieval, three Hugging Face LLMs, Streamlit chat app | [`phase2_graph_rag/`](phase2_graph_rag) |

The two phases share the dataset but not code: the Phase 1 model is not used by the Phase 2 assistant.

---

## Phase 1 – Predicting a hotel's region from review scores

**Data.** Reviews were left-joined to hotels (`hotel_id`) and users (`user_id`), giving **50,000 rows × 29 columns** (25 hotels, one per country; 2,000 users). Checks covered nulls, duplicates, constant columns, category spelling and score ranges. Countries were mapped to **11 regions** as the prediction target. Gender was label-encoded; age group and traveller type were one-hot encoded.

**Data leakage.** Each hotel carries six fixed "baseline" scores (`cleanliness_base`, `staff_base`, …). Because there is only one hotel per country, these values act as a fingerprint: hotel → country → region. Models given these columns (directly, or indirectly through `score − base` deltas) learn to recognise hotels rather than regions, so their accuracy is not a fair measure.

**Results** (stratified 80/20 split, saved notebook outputs):

| # | Model | Features | Test accuracy | Notes |
|---|---|---|---|---|
| 1 | Feed-forward NN (32→16) | scores + demographics + **hotel baselines** | 65.2% | leaks hotel identity (an earlier run was reported at 98%) |
| 2 | Logistic Regression | scores + demographics | 34.5% | weak per-class balance |
| 3 | Feed-forward NN | scores + demographics | 30.2% | collapses towards the majority class |
| 4 | Feed-forward NN | scores + demographics + **deltas** | 61.2% | deltas re-introduce the baselines |
| 5 | **Random Forest** (600 trees, balanced class weights) | scores + demographics | **37.4%** (macro F1 0.31) | chosen model |

For context, always predicting the largest class (Western Europe) scores about 24% on the test set.

**Explainability.** SHAP (TreeExplainer, per-class importance) and LIME (single-prediction explanations) were applied to the Random Forest. Facilities and location scores ranked near the top in both the forest's built-in feature importance (led by value-for-money) and SHAP (led by the overall score); demographics mattered much less. An `infer_country_group()` helper preprocesses raw inputs and returns the predicted region with its class probability.

**Files:** `team03-ms1-acl.ipynb` (full pipeline), `team03-ms1-acl_report.pdf` (write-up), `merged_hotel_reviews.csv` (merged dataset). The notebook reads the raw CSVs from `/kaggle/input/international-hotel-booking-analytics/`; change those paths to run it locally.

---

## Phase 2 – Graph-RAG travel assistant

```
question ─► intent classifier (rules) + entity extractor (spaCy + regex)
   ├─► graph retrieval: parameterised Cypher templates ─► Neo4j
   └─► semantic retrieval: Sentence-Transformers ─► FAISS
   ─► merge + de-duplicate context ─► structured prompt ─► LLM (Hugging Face) ─► Streamlit UI
```

- **Knowledge graph** (`Create_kg.py`): Traveller, Hotel, City, Country, Continent and Review nodes; `WROTE`, `STAYED_AT`, `REVIEWED`, `FROM_COUNTRY`, `LOCATED_IN`, `PART_OF` and `NEEDS_VISA` (between countries) relationships. Loaded in batches with `UNWIND` + `MERGE`, with uniqueness constraints.
- **Query understanding** (`intent_classifier.py`, `entity_extractor.py`): keyword rules choose one of four intents (hotel search, review lookup, visa info, traveller lookup); spaCy NER and regex extract cities, continents, hotels, years, demographics, amenities and score metrics.
- **Graph retrieval** (`baseline_retrieval.py`): the intent and entities select one of a set of pre-written, parameterised Cypher queries.
- **Semantic retrieval** (`embedding_retrieval.py`): review texts embedded with `all-MiniLM-L6-v2` or `all-mpnet-base-v2` and searched with a FAISS `IndexFlatL2` index (top 5). The original Kaggle review texts are synthetic filler, so the team added 505 hand-written reviews (IDs 50001–50505) for semantic search; the team also added an `amenity` column to `hotels.csv`.
- **Generation** (`llm_layer.py`): graph and semantic results are merged and de-duplicated into one context block, wrapped in a persona / context / task prompt with grounding rules, and sent to `google/gemma-2-2b-it`, `mistralai/Mistral-7B-Instruct-v0.2` or `meta-llama/Llama-3.2-3B-Instruct` via the Hugging Face Inference API.
- **Evaluation** (`quantitative_comparison.py`, `queries.txt`): the three LLMs compared on a set of test questions (response time, token usage, success and answer quality).
- **App** (`app.py`): Streamlit chat with a choice of retrieval mode (hybrid / graph only / semantic only), LLM and embedding model; shows the retrieved context, executed Cypher and an interactive graph view (NetworkX + Plotly).

Course specifications are in [`phase2_graph_rag/docs/`](phase2_graph_rag/docs).

### Running Phase 2

```bash
cd phase2_graph_rag
python -m venv .venv && .venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

cp config.example.txt config.txt     # fill in your Neo4j URI/credentials and Hugging Face token
python Create_kg.py                  # build the knowledge graph (once)
streamlit run app.py
```

The Hugging Face token is read from the `HF_TOKEN` environment variable, falling back to `config.txt`. `config.txt` is git-ignored; never commit real credentials.

---

## Repository layout

```
.
├── phase1_ml/
│   ├── team03-ms1-acl.ipynb
│   ├── team03-ms1-acl_report.pdf
│   ├── merged_hotel_reviews.csv
│   └── requirements.txt
├── phase2_graph_rag/
│   ├── app.py                    # Streamlit UI
│   ├── Create_kg.py              # Neo4j ETL
│   ├── intent_classifier.py / entity_extractor.py
│   ├── baseline_retrieval.py     # Cypher templates
│   ├── embedding_retrieval.py    # Sentence-Transformers + FAISS
│   ├── llm_layer.py              # context merge, prompt, LLM calls
│   ├── step3.py / quantitative_comparison.py / test_llm_integration.py
│   ├── hotels.csv / users.csv / reviews.csv / visa.csv
│   ├── queries.txt / rule.txt
│   ├── config.example.txt
│   ├── requirements.txt
│   └── docs/                     # course milestone specifications
└── README.md
```

## Possible next steps

- Group-aware evaluation (e.g. hold out whole hotels) to measure generalisation to unseen hotels.
- Rank-based fusion of graph and semantic results instead of simple concatenation.
- A labelled question set with automated answer scoring for the LLM comparison.

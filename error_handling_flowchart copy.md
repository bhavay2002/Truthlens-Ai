# TruthLens AI — Error Handling Flowchart (Full System)

---

## Table of Contents

1. [Failure Point Inventory](#1-failure-point-inventory)
2. [Format A — ASCII Flowchart](#3-format-b--ascii-flowchart)
3. [Format B — Step-by-Step Textual Explanation](#4-format-c--step-by-step-textual-explanation)

---

## 1. Failure Point Inventory

The following table enumerates every identified failure point across the full TruthLens pipeline, its failure class, the handling strategy applied, and the logging/alert trigger.

| #    | Stage                        | Failure Point                               | Failure Class       | Handling Strategy                            | Log Level | Alert |
| ---- | ---------------------------- | ------------------------------------------- | ------------------- | -------------------------------------------- | --------- | ----- |
| F-01 | Input Validation             | Empty or whitespace-only `text`           | Input Error         | Reject immediately → HTTP 400               | WARNING   | No    |
| F-02 | Input Validation             | `text` field is `null` / wrong type     | Schema Error        | Pydantic raises 422                          | WARNING   | No    |
| F-03 | Input Validation             | `texts` list is empty (batch)             | Input Error         | Reject → HTTP 400                           | WARNING   | No    |
| F-04 | Input Validation             | Batch size > 50                             | Input Error         | Reject → HTTP 400                           | WARNING   | No    |
| F-05 | Input Validation             | Malformed JSON body                         | Parse Error         | FastAPI raises 422                           | WARNING   | No    |
| F-06 | Preprocessing / Tokenization | Over-length input (> 512 tokens)            | Truncation          | Silent truncation to 512 chars pre-API       | INFO      | No    |
| F-07 | Preprocessing / Tokenization | Unicode / encoding error                    | Encoding Error      | Python handles natively; no crash            | WARNING   | No    |
| F-08 | Preprocessing / spaCy Load   | spaCy model missing                         | Dependency Error    | Falls back to blank `en` pipeline          | WARNING   | No    |
| F-09 | Preprocessing / spaCy Load   | spaCy load raises RuntimeError              | Runtime Error       | Exception propagated, analyzer init fails    | ERROR     | YES   |
| F-10 | Analyzer Registry Init       | Import failure of any analyzer module       | Import Error        | Full analyzer init fails → degraded mode    | ERROR     | YES   |
| F-11 | Analyzer Registry Init       | Individual analyzer `__init__` crash      | Runtime Error       | Logged; analyzer excluded from registry      | ERROR     | YES   |
| F-12 | Per-Analyzer Execution       | Analyzer `.analyze()` raises exception    | Runtime Error       | `_safe_run()` catches → returns `{}`    | WARNING   | No    |
| F-13 | Graph Pipeline               | Entity resolution failure                   | Data Error          | Returns empty graph; pipeline continues      | WARNING   | No    |
| F-14 | Graph Pipeline               | NetworkX graph build crash                  | Runtime Error       | `_safe_run()` equivalent catches → `{}` | WARNING   | No    |
| F-15 | HuggingFace API              | HTTP 503 (model loading)                    | External Dep.       | Retry once after 10s delay                   | INFO      | No    |
| F-16 | HuggingFace API              | HTTP 503 after retry                        | External Dep.       | Heuristic fallback engine activated          | WARNING   | YES   |
| F-17 | HuggingFace API              | HTTP 500 (server error)                     | External Dep.       | Immediate heuristic fallback                 | WARNING   | YES   |
| F-18 | HuggingFace API              | HTTP 429 (rate limit)                       | Throttle Error      | Returns `[]` → heuristic fallback         | WARNING   | YES   |
| F-19 | HuggingFace API              | HTTP 401 (bad/missing token)                | Auth Error          | Returns `[]` → heuristic fallback         | WARNING   | No    |
| F-20 | HuggingFace API              | Network timeout (30s)                       | Timeout             | URLError caught → heuristic fallback        | WARNING   | YES   |
| F-21 | HuggingFace API              | Malformed JSON in response                  | Parse Error         | Returns `[]` → heuristic fallback         | WARNING   | No    |
| F-22 | HuggingFace API              | Connection refused                          | Network Error       | Exception caught → heuristic fallback       | WARNING   | YES   |
| F-23 | Heuristic Fallback           | Fallback itself raises exception            | Fallback Error      | Re-raise → 500 Internal Server Error        | ERROR     | YES   |
| F-24 | Explainability Pipeline      | `/explain` requested (local model absent) | Feature Gap         | Returns HTTP 503, explains limitation        | INFO      | No    |
| F-25 | Explainability Pipeline      | SHAP computation crash                      | Runtime Error       | Logged; explanation omitted from response    | WARNING   | No    |
| F-26 | Explainability Pipeline      | LIME non-determinism / seed missing         | Config Error        | Returns explanation; consistency warning     | WARNING   | No    |
| F-27 | Aggregation Engine           | Aggregation input missing keys              | Data Error          | Uses defaults; logs missing keys             | WARNING   | No    |
| F-28 | Aggregation Engine           | Score out of [0, 1] range                   | Validation Error    | Clamped to bounds; logged                    | WARNING   | No    |
| F-29 | System Level                 | Worker OOM (local model load)               | Resource Error      | Worker crash → Gunicorn restarts worker     | ERROR     | YES   |
| F-30 | System Level                 | Gunicorn worker timeout (120s)              | Timeout             | Worker killed → restarted; 504 to client    | ERROR     | YES   |
| F-31 | System Level                 | High concurrency HF throttle                | Throttle Error      | Heuristic fallback for throttled requests    | WARNING   | YES   |
| F-32 | Response Serialization       | Non-serializable object in response         | Serialization Error | FastAPI raises 500                           | ERROR     | YES   |

---


## 2. Format A — ASCII Flowchart

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TRUTHLENS AI — FULL SYSTEM ERROR HANDLING                ║
╚══════════════════════════════════════════════════════════════════════════════╝

                          ┌─────────────────────┐
                          │    USER REQUEST      │
                          │  POST /predict       │
                          │  POST /analyze       │
                          │  POST /batch-predict │
                          └──────────┬──────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │        INPUT VALIDATION          │
                    │  • text empty / null?            │
                    │  • wrong type?                   │
                    │  • batch empty / > 50 items?     │
                    │  • malformed JSON?               │
                    └───────┬─────────────┬───────────┘
                            │             │
                         PASS           FAIL
                            │             │
                            │     ┌───────▼────────────────┐
                            │     │ LOG: WARNING            │
                            │     │ HTTP 400 (empty/batch)  │
                            │     │ HTTP 422 (schema error) │
                            │     └───────────┬────────────┘
                            │                 │
                            │          [END: Error Response]
                            │
            ┌───────────────▼───────────────────┐
            │     PREPROCESSING & TOKENIZATION   │
            │  • input length check              │
            │  • unicode normalization           │
            └───────────────┬───────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │  Length > 512 tokens?      │
              └───┬──────────────────┬────┘
                YES                  NO
                  │                  │
   ┌──────────────▼──┐               │
   │ Truncate to 512  │               │
   │ LOG: INFO        │               │
   └──────────┬───────┘               │
              └────────────┬──────────┘
                           │
            ┌──────────────▼──────────────────────┐
            │       spaCy NLP PROCESSING           │
            └──────────────┬──────────────────────┘
                           │
              ┌────────────▼────────────┐
              │ spaCy Model Available?  │
              └──┬──────────────────┬──┘
              YES                  NO
                 │                  │
                 │      ┌───────────▼──────────────────┐
                 │      │ Model missing?                │
                 │      │  → Blank 'en' fallback        │
                 │      │  → LOG: WARNING               │
                 │      │                               │
                 │      │ Load exception?               │
                 │      │  → LOG: ERROR + ALERT         │
                 │      │  → Skip analyzer init         │
                 │      │  → Jump to GRAPH PIPELINE     │
                 │      └───────────────────────────────┘
                 │
   ┌─────────────▼──────────────────────────────────┐
   │         ANALYZER REGISTRY — INITIALIZATION     │
   └─────────────┬──────────────────────────────────┘
                 │
    ┌────────────▼────────────┐
    │ Import Succeeded?       │
    └──┬───────────────────┬──┘
    YES                   NO
       │                   │
       │        ┌──────────▼────────────────┐
       │        │ LOG: ERROR + ALERT         │
       │        │ DEGRADED MODE activated    │
       │        │ → Skip to GRAPH PIPELINE   │
       │        └────────────────────────────┘
       │
   ┌───▼───────────────────────────────────────┐
   │ Initialize Each Analyzer Instance          │
   │  ArgumentMining  · NarrativeRole           │
   │  PropagandaDetector · FramingAnalyzer      │
   │  DiscourseCoherence · EmotionLexicon       │
   │  IdeologicalLang · SourceAttribution       │
   │  BiasProfile · ContextOmission             │
   └───┬───────────────────────────────────────┘
       │
    ┌──▼──────────────────────┐
    │ Individual Init Failed? │
    └──┬───────────────────┬──┘
    SOME                 NONE
       │                   │
  ┌────▼───────────────┐   │
  │ Exclude failed ones│   │
  │ LOG: ERROR per item│   │
  └────────────────────┘   │
             └─────────────┘
                    │
    ┌───────────────▼───────────────────────────────┐
    │        RUN ANALYZER PIPELINE                   │
    │  Each analyzer called via _safe_run()          │
    └───────────────┬───────────────────────────────┘
                    │
        ┌───────────▼───────────┐
        │ Per-Analyzer Exception │
        └──┬───────────────────┘
           │
     ┌─────┴───────────────────────────────┐
  EXCEPTION                          NO EXCEPTION
     │                                     │
  ┌──▼─────────────────────┐         ┌─────▼────────────┐
  │ _safe_run catches error │         │ Return feature   │
  │ Return {} for that key  │         │ dict as normal   │
  │ LOG: WARNING            │         └─────┬────────────┘
  └──────────────┬──────────┘               │
                 └─────────────┬────────────┘
                               │
                ┌──────────────▼──────────────────┐
                │  Assembled Feature Dict          │
                │  (some keys may be {} on error)  │
                └──────────────┬──────────────────┘
                               │
                ┌──────────────▼──────────────────┐
                │       GRAPH PIPELINE             │
                │  Entity graph build (NetworkX)   │
                └──────────────┬──────────────────┘
                               │
                  ┌────────────▼────────────┐
                  │ Graph Build Succeeded?  │
                  └──┬──────────────────┬──┘
                  YES                  NO
                     │                  │
                     │       ┌──────────▼───────────┐
                     │       │ Return empty graph {} │
                     │       │ LOG: WARNING          │
                     │       └──────────┬────────────┘
                     └─────────────┬────┘
                                   │
╔══════════════════════════════════▼══════════════════════════════════════════╗
║              HUGGINGFACE INFERENCE API CALL — ATTEMPT 1                     ║
╚══════════════════════════════════╤═════════════════════════════════════════╝
                                   │
                    ┌──────────────▼──────────────┐
                    │   HTTP Response Code?        │
                    └─┬────┬──────────┬──────────┬┘
                      │    │          │          │
                    200   503        4xx/5xx    TIMEOUT
                      │    │          │       /NETWORK
                      │    │          │          │
                      │  ┌─▼──────────┐          │
                      │  │Wait 10s    │          │
                      │  │LOG: INFO   │          │
                      │  └─────┬──────┘          │
                      │        │                 │
                      │  ╔═════▼═══════════════╗ │
                      │  ║  HF API RETRY (x1)  ║ │
                      │  ╚═════╤═══════════════╝ │
                      │        │                 │
                      │  ┌─────▼────────────┐   │
                      │  │ HTTP Response?   │   │
                      │  └──┬──────────┬───┘   │
                      │   200         FAIL      │
                      │    │           │        │
                      │    │   ┌───────▼────────▼───────────────────────┐
                      │    │   │     HEURISTIC FALLBACK ENGINE           │
                      │    │   │  Regex + Bias Lexicon + Caps ratio +    │
                      │    │   │  Exclamation scoring                    │
                      │    │   │  LOG: WARNING + ALERT (on throttle/     │
                      │    │   │  timeout/repeated 503/500)              │
                      │    │   └─────────────────┬───────────────────────┘
                      │    │                     │
                      │    │            ┌────────▼────────┐
                      │    │            │ Fallback OK?    │
                      │    │            └──┬───────────┬──┘
                      │    │            YES            NO
                      │    │              │             │
                      │    │              │     ┌───────▼────────────────┐
                      │    │              │     │ LOG: ERROR + ALERT      │
                      │    │              │     │ HTTP 500                │
                      │    │              │     └────────────────────────┘
                      └────┘              │
                                          │
                    ┌─────────────────────▼──────────────────────────┐
                    │  INFERENCE RESULT                               │
                    │   prediction:       FAKE | REAL                 │
                    │   fake_probability: float [0,1]                 │
                    │   real_probability: float [0,1]                 │
                    │   confidence:       max(fake_p, real_p)         │
                    │   source:           hf_api | heuristic_fallback │
                    └─────────────────────┬──────────────────────────┘
                                          │
                         ┌────────────────▼───────────────────┐
                         │   Explainability Requested?         │
                         │   (/explain endpoint)               │
                         └──┬─────────────────────────────────┘
                            │                     │
                           YES                   NO
                            │                     │
               ┌────────────▼─────────┐           │
               │ Local Model Loaded?  │           │
               └──┬───────────────┬──┘           │
               YES               NO              │
                  │               │              │
                  │    ┌──────────▼───────┐      │
                  │    │ HTTP 503         │      │
                  │    │ LOG: INFO        │      │
                  │    └──────────────────┘      │
                  │                              │
    ┌─────────────▼──────────────────────────┐  │
    │  SHAP + LIME + Attention Rollout       │  │
    └─────────────┬──────────────────────────┘  │
                  │                              │
       ┌──────────▼──────────┐                  │
       │ Computation OK?     │                  │
       └──┬──────────────┬───┘                  │
         YES            NO                      │
          │              │                      │
          │     ┌────────▼────────────────┐     │
          │     │ LOG: WARNING             │     │
          │     │ Omit explanation field   │     │
          │     └──────────────────────────┘     │
          └────────────────┬──────────────────────┘
                           │
           ┌───────────────▼───────────────────────────────────┐
           │              AGGREGATION ENGINE                    │
           │  Merge: bias + emotion + narrative + propaganda    │
           │          + discourse + ideology + source + pred    │
           └───────────────┬───────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │ Input Valid?            │
              └──┬──────────────────┬──┘
              YES                  NO
                 │                  │
                 │      ┌───────────▼────────────────┐
                 │      │ Missing keys: use defaults  │
                 │      │ Out-of-range: clamp [0,1]   │
                 │      │ LOG: WARNING per issue      │
                 │      └───────────┬────────────────┘
                 └──────────────────┘
                           │
           ┌───────────────▼─────────────────────────────────────┐
           │  credibility_profile assembled                       │
           └───────────────┬─────────────────────────────────────┘
                           │
           ┌───────────────▼─────────────────────────────────────┐
           │  RESPONSE SERIALIZATION — FastAPI JSON Encoder       │
           └───────────────┬─────────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │ Serialization OK?       │
              └──┬──────────────────┬──┘
              YES                  NO
                 │                  │
                 │       ┌──────────▼────────────────┐
                 │       │ LOG: ERROR + ALERT         │
                 │       │ HTTP 500                   │
                 │       └────────────────────────────┘
                 │
    ┌────────────▼──────────────────────────────────────────────┐
    │                 HTTP 200 — FULL RESPONSE                   │
    │   prediction, fake_probability, confidence, source         │
    │   bias, emotion, narrative, framing, rhetorical_devices    │
    │   argument_mining, information, discourse_coherence        │
    │   context_omission, ideological_language, emotion_targets  │
    │   source_attribution, propaganda_patterns, credibility     │
    └────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│               ALWAYS-ACTIVE: LOGGING & MONITORING LAYER                      │
├──────────────────────────┬──────────────────────┬───────────────────────────┤
│ Structured Log Stream    │ Metrics Collector     │ Alert Manager             │
│ asctime|level|name|msg   │ • request latency     │ • HF fallback > 20%       │
│ All WARNING+ persisted   │ • fallback_rate       │ • HF errors > 10 / 60s   │
│ ERROR → log + alert      │ • error_rate          │ • Analyzer init fail      │
│ Accessible server-side   │ • throughput (req/s)  │ • p99 > 10s               │
│ NOT exposed in 500 resp  │ • memory (RSS)        │ • Worker OOM / timeout    │
└──────────────────────────┴──────────────────────┴───────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│              SYSTEM-LEVEL FAILURE RECOVERY (Gunicorn + OS)                   │
├──────────────────────────────────┬───────────────────────────────────────────┤
│ Worker OOM on model load         │ Gunicorn detects crash → restart worker   │
│ Worker timeout (120s)            │ Gunicorn SIGKILL → restart worker         │
│ High concurrency / HF throttle   │ Per-request heuristic fallback activated  │
│ All events                       │ LOG: ERROR + ALERT → ops team notified    │
└──────────────────────────────────┴───────────────────────────────────────────┘
```

---

## 3. Format B — Step-by-Step Textual Explanation

This section traces a request through the full pipeline, explaining every decision point, failure branch, and recovery mechanism in narrative form.

---

### Step 1 — User Request Entry

A client sends a `POST` request to `/predict`, `/analyze`, or `/batch-predict` with a JSON body. The request is received by the FastAPI application, which begins parsing the payload. Every request is stamped in the uvicorn access log with method, path, and origin.

---

### Step 2 — Input Validation Gate

FastAPI delegates to Pydantic for schema enforcement. The following failure conditions are checked in order:

- **Empty or whitespace text (`F-01`):** `text.strip()` is called immediately. If the result is empty, the handler raises `HTTPException(400)` with the message `"text must not be empty"`. A `WARNING` log entry is emitted. The request terminates here.
- **Null or wrong-type field (`F-02`):** Pydantic raises a `422 Unprocessable Entity` with a detailed validation error body. The request terminates here.
- **Empty batch or batch exceeding 50 items (`F-03, F-04`):** `HTTPException(400)` is raised. Request terminates here.
- **Malformed JSON body (`F-05`):** FastAPI's JSON parser raises a `422`. Request terminates here.

**If validation passes:** the raw `text` string advances to preprocessing.

---

### Step 3 — Preprocessing and Tokenization

The text is prepared for the downstream model:

- **Over-length input (`F-06`):** The HuggingFace API call in `_hf_classify` enforces `text[:512]` before sending. For the local model path, the tokenizer truncates to 512 tokens. This is a silent, recoverable truncation logged at `INFO` level. No error is surfaced to the caller.
- **Unicode / encoding issues (`F-07`):** Python's native string handling in 3.12 and Pydantic's UTF-8 enforcement means most encoding problems are caught at the validation layer. Any remaining issues are handled natively without crashing.

---

### Step 4 — spaCy NLP Initialization

The first call to any analyzer that requires a spaCy `Doc` triggers lazy model loading:

- **Model not installed (`F-08`):** `_resolve_model()` detects via `is_package()` that the requested model (e.g., `en_core_web_sm`) is missing. It falls back to `spacy.blank("en")` and adds a sentencizer. A `WARNING` is logged. All subsequent analyzers receive a blank pipeline — entity extraction and NER-dependent features degrade gracefully.
- **Load raises a RuntimeError (`F-09`):** The exception propagates out of `get_nlp()`. The analyzer registry catches it during initialization. An `ERROR` is logged and an alert is triggered. Analyzer initialization is skipped entirely; the system enters **Degraded Mode** (prediction only, no analysis).

---

### Step 5 — Analyzer Registry Initialization

The `_get_analyzers()` function attempts to import and instantiate 17 analyzers on the first `/analyze` call (protected by a threading lock):

- **Import failure (`F-10`):** If any `from src.analysis.X import Y` statement fails (module missing, dependency not installed), the entire `try` block fails. `_analyzer_error` is set to the full traceback. An `ERROR` log entry and alert are fired. The request continues in **Degraded Mode** — only the HuggingFace prediction result is returned.
- **Individual instance initialization failure (`F-11`):** If one analyzer's `__init__` raises an exception, it is caught and logged as `ERROR`. The failed analyzer is excluded from the registry. All other analyzers continue normally.

---

### Step 6 — Per-Analyzer Execution

Each loaded analyzer is called through `_safe_run(fn, *args, **kwargs)`:

- **Exception during `.analyze()` (`F-12`):** `_safe_run` catches `Exception` universally. It logs a `WARNING` with the analyzer name and exception. It returns `{}` for that analyzer's output key. The pipeline continues with the remaining analyzers. No single analyzer crash can bring down the request.
- **Successful execution:** The feature dict is merged into the cumulative analysis result.

---

### Step 7 — Graph Pipeline Execution

Entity and narrative graphs are built from the assembled feature dict using NetworkX:

- **Entity resolution failure (`F-13`):** If entity extraction produced empty lists, the graph is built with no nodes. The pipeline continues with an empty graph.
- **Graph build exception (`F-14`):** Equivalent to a safe-run pattern — the exception is caught, an empty dict is returned for graph features, a `WARNING` is logged, and the pipeline continues.

---

### Step 8 — HuggingFace Inference API (Attempt 1)

The assembled text is sent to the HF Inference API endpoint:

- **HTTP 200:** The response JSON is parsed. Label scores are extracted. If the response structure is `[[{label, score}]]` or `[{label, score}]`, both formats are handled.
- **HTTP 503 — Model Loading (`F-15`):** The HF API returns 503 when the model is cold-starting on Hugging Face's infrastructure. The system logs `"HF model loading, retrying in 10s…"` at `INFO` level and waits exactly 10 seconds before the single retry attempt.
- **Any other error (500, 429, 401, timeout, network error) on Attempt 1 (`F-17 – F-22`):** Proceeds directly to the **Heuristic Fallback Engine** without retry.

---

### Step 9 — HuggingFace Inference API (Retry, Attempt 2)

Executed only after a 503 response on Attempt 1:

- **HTTP 200:** Parsing proceeds normally.
- **503 again (`F-16`):** The retry is exhausted. A `WARNING` is logged and an **alert** is fired (heuristic fallback rate monitor). The **Heuristic Fallback Engine** is activated.
- **500 / 429 / 401 / Timeout / Network (`F-17 – F-22`):** Same as above — alert triggered, heuristic fallback activated.

---

### Step 10 — Heuristic Fallback Engine

The regex and lexicon engine in `_heuristic_predict()` computes a score using:

1. Count of bias/disinformation keywords (e.g., "hoax", "scandal", "cover-up")
2. Count of exclamation marks
3. Ratio of uppercase characters to total length

Score formula: `min(hits × 0.08 + exclamations × 0.04 + caps_ratio × 0.3, 1.0)`

Prediction: `"FAKE"` if score > 0.45, otherwise `"REAL"`. Source field is set to `"heuristic_fallback"`.

- **Fallback engine itself raises an exception (`F-23`):** This is a critical failure. It is logged as `ERROR`, an alert is fired, and a `500 Internal Server Error` is returned to the client. This case is considered extremely unlikely as the heuristic uses only built-in Python.

---

### Step 11 — Inference Result Assembly

At this point, one of two paths produced the result:

| Source             | `source` field value          | Accuracy            |
| ------------------ | ------------------------------- | ------------------- |
| HuggingFace API    | `"huggingface_inference_api"` | ~0.891 (truthlens2) |
| Heuristic fallback | `"heuristic_fallback"`        | ~0.613              |

The final inference result dict contains: `prediction`, `fake_probability`, `real_probability`, `confidence`, `source`.

---

### Step 12 — Explainability Pipeline

Explainability is only triggered by the `/explain` endpoint:

- **Local model not available (`F-24`):** HTTP 503 is returned immediately with a message explaining that explainability requires the local model. This is the expected behavior in lightweight deployment mode.
- **SHAP / LIME crash (`F-25`):** Caught, `WARNING` logged, explanation omitted from the response. The prediction result is still returned.
- **LIME non-determinism (`F-26`):** A `WARNING` recommends enforcing `random_state=42`. If no seed is set, results may vary across identical requests.

---

### Step 13 — Aggregation Engine

All feature dicts (bias, emotion, narrative, propaganda, discourse, ideology, source attribution) plus the inference result are merged into a unified credibility profile:

- **Missing keys in feature dict (`F-27`):** Defaults (typically `0.0` or `{}`) are substituted for any missing keys. A `WARNING` is logged identifying which keys were missing.
- **Score out of [0, 1] range (`F-28`):** Any score that exceeds bounds is clamped. A `WARNING` is logged. This prevents downstream consumers from receiving invalid probability values.

---

### Step 14 — Response Serialization

FastAPI's JSON encoder serializes the final response dict:

- **Non-serializable object (`F-32`):** If any component returned a non-JSON-serializable object (e.g., a NumPy array, a Python `set`, a custom class without `__dict__`), FastAPI raises a `500 Internal Server Error`. This is logged as `ERROR` and an alert is fired. Server-side only — the internal path is never exposed to the client.

---

### Step 15 — HTTP 200 Response Delivery

A complete, structured JSON response is returned to the client containing all analysis dimensions. Latency is recorded by the metrics collector.

---

### Step 16 — System-Level Recovery (Always Active)

Independent of any request, Gunicorn monitors worker processes:

- **Worker OOM (`F-29`):** Gunicorn detects the worker crash and automatically spawns a replacement. The affected request receives a 502. An `ERROR` alert is fired.
- **Worker timeout (`F-30`):** After 120 seconds without a response, Gunicorn sends `SIGKILL` to the worker and restarts it. The client receives a 504. An alert fires.
- **High concurrency + HF throttling (`F-31`):** Each throttled request independently activates the heuristic fallback. If the fallback rate exceeds 20% of requests in a window, a `CRITICAL` alert fires.

*TruthLens AI * *Conforms to: IEEE 829, NIST AI RMF 1.0, Google SRE Error Budget Principles*

<style>#mermaid-1777995582360{font-family:sans-serif;font-size:16px;fill:#333;}#mermaid-1777995582360 .error-icon{fill:#552222;}#mermaid-1777995582360 .error-text{fill:#552222;stroke:#552222;}#mermaid-1777995582360 .edge-thickness-normal{stroke-width:2px;}#mermaid-1777995582360 .edge-thickness-thick{stroke-width:3.5px;}#mermaid-1777995582360 .edge-pattern-solid{stroke-dasharray:0;}#mermaid-1777995582360 .edge-pattern-dashed{stroke-dasharray:3;}#mermaid-1777995582360 .edge-pattern-dotted{stroke-dasharray:2;}#mermaid-1777995582360 .marker{fill:#333333;}#mermaid-1777995582360 .marker.cross{stroke:#333333;}#mermaid-1777995582360 svg{font-family:sans-serif;font-size:16px;}#mermaid-1777995582360 .label{font-family:sans-serif;color:#333;}#mermaid-1777995582360 .label text{fill:#333;}#mermaid-1777995582360 .node rect,#mermaid-1777995582360 .node circle,#mermaid-1777995582360 .node ellipse,#mermaid-1777995582360 .node polygon,#mermaid-1777995582360 .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#mermaid-1777995582360 .node .label{text-align:center;}#mermaid-1777995582360 .node.clickable{cursor:pointer;}#mermaid-1777995582360 .arrowheadPath{fill:#333333;}#mermaid-1777995582360 .edgePath .path{stroke:#333333;stroke-width:1.5px;}#mermaid-1777995582360 .flowchart-link{stroke:#333333;fill:none;}#mermaid-1777995582360 .edgeLabel{background-color:#e8e8e8;text-align:center;}#mermaid-1777995582360 .edgeLabel rect{opacity:0.5;background-color:#e8e8e8;fill:#e8e8e8;}#mermaid-1777995582360 .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#mermaid-1777995582360 .cluster text{fill:#333;}#mermaid-1777995582360 div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:sans-serif;font-size:12px;background:hsl(80,100%,96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#mermaid-1777995582360:root{--mermaid-font-family:sans-serif;}#mermaid-1777995582360:root{--mermaid-alt-font-family:sans-serif;}#mermaid-1777995582360 flowchart-v2{fill:apa;}</style>

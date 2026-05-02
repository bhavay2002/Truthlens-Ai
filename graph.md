# `graph.md` — TruthLens AI: System Graphs & Flow Diagrams

Visual reference for the complete TruthLens ML pipeline. Each section is a self-contained diagram with a short explanation. Read sequentially for a full system walkthrough, or jump to any section independently.

---

## Table of Contents

1. [High-Level System Architecture](#1-high-level-system-architecture)
2. [Detailed Data Flow Diagram](#2-detailed-data-flow-diagram)
3. [Feature Engineering Pipeline](#3-feature-engineering-pipeline)
4. [Model Architecture](#4-model-architecture)
5. [Training Workflow](#5-training-workflow)
6. [Inference Pipeline](#6-inference-pipeline)
7. [Explainability Flow](#7-explainability-flow)
8. [Evaluation Flow](#8-evaluation-flow)
9. [Component Interaction Graph](#9-component-interaction-graph)
10. [Deployment Architecture](#10-deployment-architecture)
11. [Simple Visual Explanation (Non-Technical)](#11-simple-visual-explanation)

---

## 1. High-Level System Architecture

The TruthLens system is a nine-stage pipeline. Raw news text enters on the left; calibrated, explainable predictions exit on the right.

```mermaid
flowchart LR
    A([Raw Article Text]) --> B[src/data\nLoading & Preprocessing]
    B --> C[src/analysis\nLinguistic Analysis]
    C --> D[src/features\nFeature Engineering]
    D --> E[src/models\nMultiTask Model]
    E --> F[src/training\nTraining Loop]
    F --> G[src/evaluation\nCalibration & Metrics]
    G --> H[src/inference\nProduction Serving]
    H --> I[src/explainability\nExplainability]
    I --> J([Structured Report])

    style A fill:#e8f5e9,stroke:#388e3c
    style J fill:#e3f2fd,stroke:#1565c0
    style E fill:#fce4ec,stroke:#c62828
    style H fill:#fff8e1,stroke:#f57f17
```

**Flow summary:**
- `src/data` cleans and tokenises raw text.
- `src/analysis` runs 15+ linguistic detectors (bias, emotion, propaganda, framing, etc.).
- `src/features` fuses hand-crafted features into model-ready tensors.
- `src/models` hosts the `MultiTaskTruthLensModel` (RoBERTa encoder + six task heads).
- `src/training` drives the supervised multi-task learning loop.
- `src/evaluation` measures calibration, uncertainty, and cross-task correlation.
- `src/inference` serves predictions with caching, monitoring, and drift detection.
- `src/explainability` generates SHAP, LIME, and attention-based explanations.

---

## 2. Detailed Data Flow Diagram

Shows every transformation from raw input to final structured report, including the graph subpipeline and aggregation layer.

```mermaid
flowchart TD
    IN([Article Text]) --> PRE[Preprocessing\nspaCy tokenisation\nsentence splitting\ncleaning]

    PRE --> FA[Feature Extraction\nFeaturePipeline.extract]
    PRE --> GA[Graph Construction\nEntityGraphBuilder\nNarrativeGraphBuilder]
    PRE --> AA[Analysis Modules\nAnalysisIntegrationRunner\n15+ detectors]

    FA --> |bias_* emotion_*\nnarrative_* discourse_*| FM[Feature Merger\nFeatureMerger.merge]
    GA --> |entity nodes\nedge weights| GM[Graph Metrics\nGraphAnalyzer.analyze]
    AA --> |module outputs| PR[Bias Profile\nBiasProfileBuilder.build_profile]

    FM --> FP[FeaturePreparer\nscale → tensor]
    GM --> PR
    PR --> AGG[AggregationPipeline\nweighted score fusion]

    FP --> |aux feature tensor| ENG[InferenceEngine\nRoBERTa + task heads]
    AGG --> |credibility score| ENG

    ENG --> |per-task logits| CAL[Calibration\ntemperature scaling\nPlatt scaling]
    CAL --> |calibrated probs| POST[PostProcessor\nsoftmax → argmax → labels]

    POST --> |predictions| REP[ReportGenerator\ngenerate_report]
    AGG --> |aggregation block| REP
    AA --> |analysis_modules| REP

    REP --> |full report dict| FMT[ResultFormatter]
    FMT --> API[API Response]
    FMT --> DASH[Dashboard Report]
    FMT --> RES[Research Export]

    REP --> EXP[ExplainabilityPipeline\nSHAP · LIME · Attention]
    EXP --> |explanations| REP

    style IN fill:#e8f5e9,stroke:#388e3c
    style ENG fill:#fce4ec,stroke:#c62828
    style REP fill:#e3f2fd,stroke:#1565c0
    style API fill:#fff8e1,stroke:#f57f17
    style DASH fill:#fff8e1,stroke:#f57f17
    style RES fill:#fff8e1,stroke:#f57f17
```

---

## 3. Feature Engineering Pipeline

Three parallel extractor branches feed into a shared fusion layer that produces a single float tensor.

```mermaid
flowchart TD
    IN([Feature Dict\nbias_* emotion_*\nnarrative_* discourse_*]) --> BE[BiasExtractor\npartisan lean\nsource credibility\nword-choice bias]
    IN --> FE[FramingExtractor\nepisodic vs thematic\nurgency score\ndiscourse markers]
    IN --> IE[IdeologicalExtractor\nleft-right score\npopulism index\nrhetoric intensity]

    BE --> FL[Flatten & Concatenate\n1-D float list]
    FE --> FL
    IE --> FL

    FL --> SS[StandardScaler\nzero-mean unit-variance]
    SS --> TEN[torch.tensor\nfloat32]
    TEN --> DIM{expected_dim\ncheck}
    DIM -->|match| OUT([Auxiliary Feature Tensor\nshape: feature_dim])
    DIM -->|mismatch| ERR([RuntimeError])

    OUT --> HEAD[Model Auxiliary Head\nlinear projection → task logit offset]

    style IN fill:#e8f5e9,stroke:#388e3c
    style OUT fill:#e3f2fd,stroke:#1565c0
    style ERR fill:#ffebee,stroke:#c62828
    style HEAD fill:#fce4ec,stroke:#c62828
```

**Feature config flags** (`FeaturePreparationConfig`):

| Flag | Effect |
|---|---|
| `use_bias_features` | Include / exclude `BiasExtractor` branch |
| `use_framing_features` | Include / exclude `FramingExtractor` branch |
| `use_ideological_features` | Include / exclude `IdeologicalExtractor` branch |
| `scale_features` | Apply `StandardScaler` before tensorisation |

---

## 4. Model Architecture

`MultiTaskTruthLensModel` shares a single RoBERTa encoder across all six classification heads.

```mermaid
flowchart TD
    TXT([Tokenised Text\ninput_ids · attention_mask]) --> ENC[RoBERTa Encoder\nroberta-base\n12 layers · 768-dim hidden\n125M parameters]

    ENC --> |CLS embedding\n768-dim| POOL[Pooling Layer\nCLS token extraction]
    POOL --> DROP[Dropout\np=0.1]

    DROP --> H1[Emotion Head\nLinear 768→6\nsoftmax]
    DROP --> H2[Narrative Head\nLinear 768→4\nsoftmax]
    DROP --> H3[Propaganda Head\nLinear 768→2\nsoftmax]
    DROP --> H4[Bias Head\nLinear 768→3\nsoftmax]
    DROP --> H5[Ideology Head\nLinear 768→5\nsoftmax]
    DROP --> H6[Narrative Frame Head\nLinear 768→4\nsoftmax]

    H1 --> |logits| OUT1([emotion logits])
    H2 --> |logits| OUT2([narrative logits])
    H3 --> |logits| OUT3([propaganda logits])
    H4 --> |logits| OUT4([bias logits])
    H5 --> |logits| OUT5([ideology logits])
    H6 --> |logits| OUT6([narrative_frame logits])

    AUX([Auxiliary Feature Tensor\nFeaturePreparer output]) --> AUXH[Auxiliary Projection\nLinear → task offsets]
    AUXH --> |logit offsets| H1
    AUXH --> |logit offsets| H3
    AUXH --> |logit offsets| H4

    style ENC fill:#fce4ec,stroke:#c62828
    style POOL fill:#fce4ec,stroke:#c62828
    style DROP fill:#fce4ec,stroke:#c62828
    style AUX fill:#e8f5e9,stroke:#388e3c
```

**Checkpoint:** `bhavaygupta2002/truthlens_v1/checkpoint.pt`  
**Encoder:** `roberta-base` (frozen or fine-tuned per training config)  
**Tasks:** emotion · narrative · propaganda · bias · ideology · narrative_frame

---

## 5. Training Workflow

```mermaid
flowchart TD
    DS([Dataset\nCSV / JSONL]) --> DL[DataLoader\nbatch_size · shuffle\ncollate_fn]
    DL --> |batch| TOK[Tokenizer\nAutoTokenizer.from_pretrained\nmax_length=512 · truncation]

    TOK --> |input_ids\nattention_mask| FWD[Forward Pass\nMultiTaskTruthLensModel]
    FWD --> |per-task logits| LOSS[Loss Engine\nper-task CrossEntropy\nweighted sum]

    LOSS --> |scalar loss| BP[Backpropagation\nloss.backward]
    BP --> |gradients| OPT[Optimizer\nAdamW · lr scheduling\ngradient clipping]
    OPT --> |updated weights| FWD

    LOSS --> LB[LossBalancer\ndynamic task weight\nadjustment]
    LB --> |adjusted weights| LOSS

    OPT --> CHK{Checkpoint\nstep?}
    CHK -->|yes| SAVE[Checkpointing\nsave checkpoint.pt\nmodel metadata]
    CHK -->|no| CONT[Continue]

    FWD --> |validation split| EVAL[EvaluationEngine\nmetrics per task]
    EVAL --> |metrics dict| TRACK[ExperimentTracker\nMLflow logging]
    TRACK --> |early stopping?| SCHED[TaskScheduler\nadjust task sampling]
    SCHED --> DL

    SAVE --> FINAL([Trained Checkpoint\nbhavaygupta2002/truthlens_v1])

    style DS fill:#e8f5e9,stroke:#388e3c
    style FWD fill:#fce4ec,stroke:#c62828
    style FINAL fill:#e3f2fd,stroke:#1565c0
    style TRACK fill:#fff8e1,stroke:#f57f17
```

Key training components:

| Component | File | Role |
|---|---|---|
| `Trainer` | `trainer.py` | Outer training loop |
| `LossEngine` | `loss_engine.py` | Weighted multi-task loss computation |
| `LossBalancer` | `loss_balancer.py` | Dynamic per-task weight adjustment |
| `TaskScheduler` | `task_scheduler.py` | Curriculum and task sampling |
| `ExperimentTracker` | `experiment_tracker.py` | MLflow metric/artefact logging |
| `DistributedEngine` | `distributed_engine.py` | Multi-GPU DDP wrapper |

---

## 6. Inference Pipeline

Full request lifecycle from HTTP POST to cached, monitored, logged response.

```mermaid
flowchart TD
    REQ([HTTP POST /predict\nPredictRequest]) --> SING[predict_api._get_service\ndouble-checked lock singleton]
    SING --> SVC[PredictionService.predict]

    SVC --> KEY[Cache Key\nSHA-256 version:text]
    KEY --> MCACHE{Memory LRU\nhit?}

    MCACHE -->|hit| CACHED([Return Cached Result\ncached=True])
    MCACHE -->|miss| DLOCK[Single-Flight Lock\nper key — LAT-5]
    DLOCK --> DCACHE{Disk Cache\ngzip JSON hit?}

    DCACHE -->|hit| DCACHED[Decompress & Deserialise\nstore in LRU]
    DCACHED --> CACHED
    DCACHE -->|miss| TOK2[AutoTokenizer\nencode · truncate · pad]

    TOK2 --> |input tensors| AMP[AMP Context\ntorch.cuda.amp.autocast\non CUDA only]
    AMP --> |fp16/bf16| MOD[MultiTaskTruthLensModel\nforward pass]
    MOD --> |raw logits dict| CALIB[Calibration\ntemperature ÷ Platt scale]
    CALIB --> |scaled logits| PP[PostProcessor.process\nsoftmax → argmax → labels\nPP-3: iterate logits.keys]

    PP --> |predictions + probs| RESULT[Result Dict\nlabel · confidence\nfake_probability · task_outputs]
    RESULT --> WRCACHE[Write Cache\nLRU + gzip disk]
    RESULT --> MON[InferenceMonitor.update\nlatency · confidence · entropy\nrolling 500-window alerts]
    RESULT --> LOG[InferenceLogger.log_prediction\nJSON audit line\ntrace_id · article_id]

    WRCACHE --> RESP([PredictResponse\nprocessing_time_ms · cached=False])
    MON --> RESP
    LOG --> RESP

    style REQ fill:#e8f5e9,stroke:#388e3c
    style MOD fill:#fce4ec,stroke:#c62828
    style RESP fill:#e3f2fd,stroke:#1565c0
    style CACHED fill:#fff8e1,stroke:#f57f17
```

**Batch path** (high-throughput):

```mermaid
flowchart LR
    TEXTS([List of Texts]) --> CHUNK[Chunk into\nbatch_size windows]
    CHUNK --> ENG[InferenceEngine.predict_batch\nper chunk]
    ENG --> STACK[Stack logits & probs\nper task → np.ndarray]
    STACK --> DRIFT{run_drift_detection?}
    DRIFT -->|yes| DD[DriftDetector.detect\nKL · JS · PSI · Wasserstein\nvs stored baseline]
    DRIFT -->|no| OUT2
    DD --> OUT2([Batch Results Dict\n+ drift report])

    style TEXTS fill:#e8f5e9,stroke:#388e3c
    style OUT2 fill:#e3f2fd,stroke:#1565c0
```

---

## 7. Explainability Flow

Multiple explanation methods run in parallel; results are aggregated, cached, and embedded in the report.

```mermaid
flowchart TD
    PRED([Model Prediction\nlogits · probs · CLS embedding]) --> ORCH[ExplainabilityPipeline\norchestrator.py]

    ORCH --> SHAP[SHAPExplainer\nshapley values\nper token · per task]
    ORCH --> LIME[LIMEExplainer\nperturbation-based\nlocal surrogates]
    ORCH --> ATT[AttentionRollout\nmulti-head attention\nrollout across layers]
    ORCH --> BIAS_EXP[BiasExplainer\nword-level bias\nattribution]
    ORCH --> PROP_EXP[PropagandaExplainer\ntechnique attribution\nhighlighted spans]
    ORCH --> EMO_EXP[EmotionExplainer\nsentence-level\nemotion attribution]

    SHAP --> AGG2[ExplanationAggregator\nensemble importance scores\nconsistency checks]
    LIME --> AGG2
    ATT --> AGG2
    BIAS_EXP --> AGG2
    PROP_EXP --> AGG2
    EMO_EXP --> AGG2

    AGG2 --> CAL2[ExplanationCalibrator\nalign explanation\nwith prediction confidence]
    CAL2 --> CACHE2[ExplanationCache\nstore artefacts]

    CACHE2 --> VIZ[ExplanationVisualizer\nHTML highlight view\nattention heatmap]
    CACHE2 --> JSON2[ExplanationReportGenerator\njson · html artefacts]

    VIZ --> OUT3([Visual Explanation\nHTML / PNG])
    JSON2 --> OUT4([Explanation Report\narticle_id scoped])

    style PRED fill:#fce4ec,stroke:#c62828
    style AGG2 fill:#e8f5e9,stroke:#388e3c
    style OUT3 fill:#e3f2fd,stroke:#1565c0
    style OUT4 fill:#e3f2fd,stroke:#1565c0
```

---

## 8. Evaluation Flow

```mermaid
flowchart TD
    MODEL([Trained Checkpoint]) --> EP[EvaluationPipeline\nevaluate_model.py]
    TESTD([Test Dataset\nwith ground-truth labels]) --> EP

    EP --> INF2[InferenceEngine.predict_for_evaluation\nstacked logits + probs\nshape N × n_classes per task]

    INF2 --> CAL3[compute_calibration\nECE · MCE · reliability diagram\nPlatt scaling coefficients]
    INF2 --> UNC[uncertainty_statistics\nentropy mean · p95\nconfidence histogram]
    INF2 --> CORR[compute_task_correlation\nSpearman ρ between tasks\ncorrelation matrix]
    INF2 --> METR[MetricsEngine\naccuracy · F1 · AUC-ROC\nper task × per class]
    INF2 --> FAIR[FairnessEvaluator\nequal opportunity\ncalibration by group]
    INF2 --> THRESH[ThresholdOptimizer\nPR curve analysis\noptimal decision threshold]

    CAL3 --> DASH2[EvaluationDashboard\ninteractive HTML]
    UNC --> DASH2
    CORR --> DASH2
    METR --> DASH2
    FAIR --> DASH2
    THRESH --> DASH2

    METR --> MLFLOW[MLflowTracker\nlog metrics · tags\nartefact store]
    CAL3 --> REPORT2[ReportWriter\nsave_report → report.json\nPDF report]
    METR --> REPORT2

    DASH2 --> OUT5([evaluation_dashboard.html])
    REPORT2 --> OUT6([report.json · report.pdf])

    style MODEL fill:#fce4ec,stroke:#c62828
    style TESTD fill:#e8f5e9,stroke:#388e3c
    style DASH2 fill:#fff8e1,stroke:#f57f17
    style OUT5 fill:#e3f2fd,stroke:#1565c0
    style OUT6 fill:#e3f2fd,stroke:#1565c0
```

Evaluation outputs:

| Output | Description |
|---|---|
| ECE / MCE | Expected and Maximum Calibration Error |
| Reliability diagram | Confidence-vs-accuracy bucket plot |
| Task correlation matrix | Spearman ρ between all six task probability vectors |
| Uncertainty histogram | Entropy distribution across test set |
| Fairness report | Calibration parity across demographic groups |
| PDF report | Full printable evaluation summary |

---

## 9. Component Interaction Graph

Shows which modules depend on which, across all nine packages.

```mermaid
flowchart TD
    subgraph DATA ["src/data"]
        DL2[DataLoader]
        PRE2[Preprocessor]
    end

    subgraph ANALYSIS ["src/analysis"]
        IR[AnalysisIntegrationRunner]
        BPB[BiasProfileBuilder]
        PPD[PropagandaPatternDetector]
        FA2[FramingAnalysis]
        EA[EmotionTargetAnalysis]
        ARG[ArgumentMining]
        COH[DiscourseCoherenceAnalyzer]
    end

    subgraph GRAPH ["src/graph"]
        EGB[EntityGraphBuilder]
        NGB[NarrativeGraphBuilder]
        GAN[GraphAnalyzer]
        GP[GraphPipeline]
    end

    subgraph FEATURES ["src/features"]
        FPL[FeaturePipeline]
        FMR[FeatureMerger]
        FPR[FeaturePreparer]
    end

    subgraph AGGREGATION ["src/aggregation"]
        AGGP[AggregationPipeline]
        TSC[TruthLensScoreCalculator]
        RM[RiskAssessment]
        WM[WeightManager]
    end

    subgraph MODELS ["src/models"]
        ENC2[RoBERTa Encoder]
        HEADS[Task Heads ×6]
        CAL4[ModelCalibration]
    end

    subgraph TRAINING ["src/training"]
        TR[Trainer]
        LE[LossEngine]
        LB2[LossBalancer]
        ET[ExperimentTracker]
    end

    subgraph EVALUATION ["src/evaluation"]
        EVPL[EvaluationPipeline]
        METR2[MetricsEngine]
        CAL5[compute_calibration]
        CORR2[task_correlation]
    end

    subgraph INFERENCE ["src/inference"]
        IE2[InferenceEngine]
        PS[PredictionService]
        IC[InferenceCache]
        IM[InferenceMonitor]
        DD2[DriftDetector]
        IL[InferenceLogger]
        BIE[BatchInferenceEngine]
        AA[ArticleAnalyzer]
        RG[ReportGenerator]
        RF[ResultFormatter]
    end

    subgraph EXPLAINABILITY ["src/explainability"]
        SE[SHAPExplainer]
        LE2[LIMEExplainer]
        AR[AttentionRollout]
        ERG[ExplanationReportGenerator]
    end

    subgraph API ["api/"]
        APP[FastAPI app.py]
        ROUTER[predict_api router]
    end

    DL2 --> PRE2
    PRE2 --> FPL
    PRE2 --> IR
    PRE2 --> EGB

    FPL --> FMR
    FMR --> FPR
    FPR --> IE2

    IR --> BPB
    BPB --> AGGP
    AGGP --> TSC
    AGGP --> RM
    WM --> AGGP

    EGB --> GAN
    NGB --> GP
    GP --> GAN
    GAN --> BPB

    ENC2 --> HEADS
    HEADS --> CAL4

    IE2 --> ENC2
    IE2 --> CAL4
    IE2 --> PS
    PS --> IC
    PS --> IM
    PS --> IL
    BIE --> IE2
    BIE --> DD2
    AA --> PS
    AA --> AGGP
    AA --> RG
    RG --> RF

    TR --> ENC2
    TR --> LE
    LE --> LB2
    TR --> ET

    EVPL --> IE2
    EVPL --> METR2
    EVPL --> CAL5
    EVPL --> CORR2

    SE --> ERG
    LE2 --> ERG
    AR --> ERG
    ERG --> RG

    APP --> ROUTER
    ROUTER --> PS

    style MODELS fill:#fce4ec,stroke:#c62828
    style INFERENCE fill:#fff8e1,stroke:#f57f17
    style API fill:#e8f5e9,stroke:#388e3c
    style EXPLAINABILITY fill:#e3f2fd,stroke:#1565c0
```

---

## 10. Deployment Architecture

```mermaid
flowchart TD
    subgraph CLIENT ["Client Layer"]
        BR[Browser / API Client]
        CLI2[run_inference.py CLI]
    end

    subgraph REPLIT ["Replit Cloud\n.replit deployment"]
        subgraph BUILD ["Build Phase"]
            B1[pip install torch CPU-only\nindex-url pytorch.org/whl/cpu]
            B2[pip install -r requirements.txt]
            B1 --> B2
        end

        subgraph SERVE ["Runtime — Gunicorn + Uvicorn\n--workers 1 --timeout 120"]
            APP2[FastAPI app.py\nport 5000]
            APP2 --> IR2[/predict\nPredictRequest → PredictResponse]
            APP2 --> IH[/predict/health\nMonitoring snapshot]
            APP2 --> IB[/predict/batch\nBatchPredictRequest]
        end
    end

    subgraph MODEL_STORE ["Model Storage"]
        HF[HuggingFace Hub\nbhavaygupta2002/truthlens_v1\ncheckpoint.pt]
    end

    subgraph CACHE_STORE ["Cache Layer"]
        MLRU[In-process LRU\n512 slots]
        DISK[gzip JSON disk\n.cache/inference/]
    end

    subgraph LOGS ["Observability"]
        JLOG[JSON audit logs\nInferenceLogger]
        MON2[Rolling metrics\nInferenceMonitor]
        DRIFT2[Drift alerts\nDriftDetector]
    end

    BR --> |HTTPS| APP2
    CLI2 --> |direct Python| APP2
    APP2 --> SINGLETON[PredictionService singleton\ndouble-checked lock]
    SINGLETON --> MLRU
    MLRU --> |miss| DISK
    DISK --> |miss| ENG2[InferenceEngine\nloads from MODEL_STORE]
    HF --> ENG2
    ENG2 --> SINGLETON
    SINGLETON --> JLOG
    SINGLETON --> MON2
    SINGLETON --> DRIFT2

    style REPLIT fill:#fffde7,stroke:#f9a825
    style MODEL_STORE fill:#fce4ec,stroke:#c62828
    style CACHE_STORE fill:#e8f5e9,stroke:#388e3c
    style LOGS fill:#e3f2fd,stroke:#1565c0
```

**Deployment constraints:**

| Constraint | Reason |
|---|---|
| CPU-only torch wheel | CUDA torch exceeds the 8 GiB image limit |
| `--workers 1` | Prevents duplicate model loads (one 500 MB model per worker) |
| `--timeout 120` | Accommodates first-request CUDA warm-up / cold start |
| Disk cache sharding | `key[:2]` subdirectory prefix prevents flat-directory FS slowdown |

---

## 11. Simple Visual Explanation

For non-technical reviewers: what TruthLens actually does, in plain language.

```mermaid
flowchart LR
    A([You paste a\nnews article]) --> B{TruthLens AI}
    B --> C([Is it real or fake?\nconfidence score])
    B --> D([What emotions\ndoes it use?])
    B --> E([Is the language\nbiased?])
    B --> F([Does it spread\npropaganda?])
    B --> G([What political\nideology?])
    B --> H([How is the\nstory framed?])

    style A fill:#e8f5e9,stroke:#388e3c,color:#1b5e20
    style B fill:#fce4ec,stroke:#c62828,color:#b71c1c
    style C fill:#e3f2fd,stroke:#1565c0
    style D fill:#e3f2fd,stroke:#1565c0
    style E fill:#e3f2fd,stroke:#1565c0
    style F fill:#e3f2fd,stroke:#1565c0
    style G fill:#e3f2fd,stroke:#1565c0
    style H fill:#e3f2fd,stroke:#1565c0
```

**Plain-language walkthrough:**

```mermaid
flowchart TD
    S1["① READ\nThe AI reads your article\nword by word"] --> S2
    S2["② ANALYSE\nIt looks for emotional language,\npropaganda tricks, biased words,\nand political framing"] --> S3
    S3["③ THINK\nA deep learning model\n(trained on thousands of articles)\nscores each dimension"] --> S4
    S4["④ EXPLAIN\nIt highlights exactly which words\nand sentences triggered each score"] --> S5
    S5["⑤ REPORT\nYou get a credibility score,\nconfidence percentage,\nand a full breakdown"]

    style S1 fill:#e8f5e9,stroke:#388e3c
    style S2 fill:#fff8e1,stroke:#f57f17
    style S3 fill:#fce4ec,stroke:#c62828
    style S4 fill:#e3f2fd,stroke:#1565c0
    style S5 fill:#f3e5f5,stroke:#6a1b9a
```

**What each score means:**

| Output | Plain meaning |
|---|---|
| `fake_probability` | How likely the article contains misinformation (0 = trustworthy, 1 = likely fake) |
| `emotion` | Which emotion the article tries to provoke (fear, anger, joy, etc.) |
| `bias` | Whether the language leans politically left, centre, or right |
| `propaganda` | Whether the article uses known influence techniques |
| `ideology` | The political worldview underlying the article |
| `narrative_frame` | How the story is packaged (conflict, human interest, economic, etc.) |
| `credibility_score` | Single 0–100 aggregate score from all dimensions combined |

---

## Legend

```mermaid
flowchart LR
    A([Start / End\ngreen border]) --- B[Processing Step\nno fill]
    B --- C{Decision\nor Branch}
    C --- D([Output / Result\nblue border])

    style A fill:#e8f5e9,stroke:#388e3c
    style D fill:#e3f2fd,stroke:#1565c0

    E[Model Component\nred border] --- F[API / Serving\nyellow border]
    F --- G[Data / Input\ngreen border]

    style E fill:#fce4ec,stroke:#c62828
    style F fill:#fff8e1,stroke:#f57f17
    style G fill:#e8f5e9,stroke:#388e3c
```

| Colour | Meaning |
|---|---|
| Green border | Raw data input or final output |
| Red border | Neural network / model components |
| Blue border | Structured results and reports |
| Yellow border | API, serving, and cache layer |
| No fill | Internal processing steps |

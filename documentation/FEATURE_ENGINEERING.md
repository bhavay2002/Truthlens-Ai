
# FEATURE_ENGINEERING.md

# TruthLens AI Feature Engineering System

This document describes the  **feature engineering system used in TruthLens AI** .

Feature engineering converts **raw article text into structured signals** that can be used by machine learning models and analysis modules.

TruthLens uses a **multi-layer feature system** combining:

* lexical features
* semantic features
* syntactic features
* ideological signals
* narrative features
* propaganda signals
* emotional signals
* graph-based features

These features provide  **structured information that helps the model detect misinformation patterns** .

---

# Feature Engineering Overview

Feature extraction pipeline:

```text
Article Text
      ↓
Text Preprocessing
      ↓
Tokenization
      ↓
Feature Extractors
      ↓
Feature Fusion
      ↓
Unified Feature Representation
```

Features are generated using modules located in:

```text
src/features/
```

The system is designed to support  **modular feature extraction and scalable experimentation** .

---

# Feature Categories

TruthLens extracts features from multiple dimensions of article content.

| Feature Category    | Description                      |
| ------------------- | -------------------------------- |
| Text Features       | Basic linguistic properties      |
| Bias Features       | Indicators of ideological bias   |
| Emotion Features    | Emotional tone and intensity     |
| Narrative Features  | Narrative framing signals        |
| Propaganda Features | Manipulative rhetoric patterns   |
| Discourse Features  | Argument structure and coherence |
| Graph Features      | Entity interaction signals       |

---

# Text Features

Text features capture **basic linguistic properties** of articles.

Location:

```text
src/features/text/
```

Examples include:

* token counts
* sentence length statistics
* word frequency distributions
* lexical diversity
* syntactic complexity

Example modules:

```text
lexical_features.py
semantic_features.py
syntactic_features.py
token_features.py
```

These features provide  **baseline linguistic signals** .

---

# Bias Features

Bias features detect  **ideological framing and partisan language** .

Location:

```text
src/features/bias/
```

Examples include:

* ideological lexicons
* framing language
* partisan indicators

Modules:

```text
bias_features.py
bias_lexicon_features.py
framing_features.py
ideological_features.py
```

These features help identify  **politically biased narratives** .

---

# Emotion Features

Emotion features analyze the  **emotional tone of articles** .

Location:

```text
src/features/emotion/
```

Examples:

* emotion intensity
* emotional trajectory across the article
* emotional target detection

Modules:

```text
emotion_features.py
emotion_intensity_features.py
emotion_lexicon_features.py
emotion_target_features.py
emotion_trajectory_features.py
```

Emotion features help detect  **emotionally manipulative content** .

---

# Narrative Features

Narrative features capture  **story structure and framing techniques** .

Location:

```text
src/features/narrative/
```

Examples include:

* narrative frames
* conflict detection
* narrative roles

Modules:

```text
narrative_features.py
narrative_frame_features.py
narrative_role_features.py
conflict_features.py
```

These features identify  **strategic storytelling used in misinformation** .

---

# Propaganda Features

Propaganda features detect  **persuasive and manipulative rhetoric** .

Location:

```text
src/features/propaganda/
```

Examples include:

* loaded language
* fear appeals
* exaggeration patterns

Modules:

```text
propaganda_features.py
propaganda_lexicon_features.py
manipulation_patterns.py
```

These features detect  **propaganda techniques used in misinformation campaigns** .

---

# Discourse Features

Discourse features analyze the  **argument structure of articles** .

Location:

```text
src/features/discourse/
```

Examples:

* argument structure
* claim–evidence relationships
* discourse coherence

Modules:

```text
argument_structure_features.py
discourse_features.py
```

These features evaluate  **logical consistency and information structure** .

---

# Graph-Based Features

Graph features capture  **relationships between entities and interactions** .

Location:

```text
src/features/graph/
```

Examples:

* entity interaction patterns
* narrative propagation signals

Modules:

```text
entity_graph_features.py
interaction_graph_features.py
```

Graph features help detect  **coordinated narratives or information spread** .

---

# Feature Fusion

After individual features are extracted, they are  **combined into a unified representation** .

Location:

```text
src/features/fusion/
```

Fusion pipeline:

```text
Multiple Feature Sets
        ↓
Feature Scaling
        ↓
Feature Selection
        ↓
Feature Fusion
        ↓
Unified Feature Vector
```

Modules:

```text
feature_fusion.py
feature_scaling.py
feature_selection.py
```

---

# Feature Pipelines

Feature pipelines coordinate feature extraction across modules.

Location:

```text
src/features/pipelines/
```

Key pipelines:

| Pipeline                  | Purpose                             |
| ------------------------- | ----------------------------------- |
| feature_pipeline.py       | standard feature extraction         |
| batch_feature_pipeline.py | batch processing for large datasets |

Pipeline flow:

```text
Input Dataset
      ↓
Feature Extractors
      ↓
Feature Fusion
      ↓
Output Feature Matrix
```

---

# Feature Caching

Feature extraction can be computationally expensive.

TruthLens includes a  **feature caching system** .

Location:

```text
src/features/cache/
```

Cache components:

```text
cache_manager.py
feature_cache.py
```

Benefits:

* faster training iterations
* reduced computation cost
* reusable feature outputs

---

# Feature Importance

TruthLens includes tools to analyze  **feature importance** .

Location:

```text
src/features/importance/
```

Methods include:

* permutation importance
* SHAP importance
* feature ablation

Modules:

```text
feature_ablation.py
permutation_importance.py
shap_importance.py
```

These tools help determine  **which signals most influence predictions** .

---

# Feature Validation

Feature validation ensures correctness and consistency.

Location:

```text
src/features/
```

Modules:

```text
feature_schema_validator.py
feature_statistics.py
dataset_feature_generator.py
```

Validation checks include:

* feature schema integrity
* missing values
* feature distribution checks

---

# Feature Engineering Principles

TruthLens feature system follows several design principles.

### Modularity

Each feature module is independent.

### Interpretability

Features are designed to be understandable and explainable.

### Extensibility

New feature modules can be added easily.

### Efficiency

Caching and batch pipelines improve performance.

---

# Feature Engineering Workflow

Complete workflow:

```text
Article Text
      ↓
Preprocessing
      ↓
Tokenization
      ↓
Feature Extraction
      ↓
Feature Fusion
      ↓
Feature Validation
      ↓
Model Input
```

---

# Future Improvements

Planned feature engineering improvements include:

* multilingual feature extraction
* advanced discourse features
* knowledge graph features
* misinformation propagation signals
* temporal narrative features

---

If you want, I can also generate  **one extremely useful document for ML engineers** :

**`PIPELINES.md`**

It would explain  **every pipeline in your system** :

* data pipeline
* feature pipeline
* training pipeline
* inference pipeline
* TruthLens analysis pipeline

That will make the architecture **much easier for contributors to understand.**

"""
File: src/eda/eda.py

Purpose
-------
Advanced Exploratory Data Analysis (EDA) module for fake news datasets.

Provides:
- Dataset statistics
- Label distribution analysis
- Text feature analysis
- Vocabulary analysis
- N-gram analysis
- TF-IDF keyword extraction
- Word clouds
- Automatic report generation

Outputs
-------
reports/
 ├── figures/
 │     ├── *.png
 │
 └── eda_summary.json

Inputs
------
df : pandas.DataFrame
csv_path : str

Outputs
-------
EDA figures + JSON summary report

Dependencies
------------
pandas
numpy
matplotlib
seaborn
sklearn
nltk
wordcloud
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk.corpus import stopwords
from wordcloud import WordCloud

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")

try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    try:
        import nltk

        nltk.download("stopwords", quiet=True)
        STOPWORDS = set(stopwords.words("english"))
    except Exception:
        logger.warning("NLTK stopwords unavailable; continuing with empty stopword list")
        STOPWORDS = set()


class FakeNewsEDA:
    """
    Modular EDA pipeline for fake news datasets.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing 'text' and 'label'.

    output_dir : Path
        Directory for saving figures.
    """

    def __init__(self, df: pd.DataFrame, output_dir: str | Path = Path("reports/figures")):

        self.df = df.copy()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.report_dir = Path("reports")
        self.report_dir.mkdir(exist_ok=True)

        self.summary: Dict[str, Any] = {}

    # --------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------

    def dataset_summary(self) -> None:

        logger.info("Running dataset summary")

        self.summary["shape"] = tuple(self.df.shape)
        self.summary["columns"] = list(self.df.columns)

    # --------------------------------------------------
    # DATA TYPE CHECK
    # --------------------------------------------------

    def check_data_types(self) -> None:

        if "label" in self.df.columns and self.df["label"].dtype == "object":

            self.df["label"] = self.df["label"].map(
                {"real": 0, "fake": 1}
            )

    # --------------------------------------------------
    # MISSING VALUE HANDLING
    # --------------------------------------------------

    def handle_missing(self) -> None:

        missing = self.df.isnull().sum().to_dict()

        self.summary["missing_values"] = missing

        if "text" in self.df.columns:
            self.df = self.df.dropna(subset=["text"])
        self.df = self.df.fillna("")

    # --------------------------------------------------
    # DUPLICATE REMOVAL
    # --------------------------------------------------

    def remove_duplicates(self) -> None:

        before = len(self.df)

        self.df = self.df.drop_duplicates()

        removed = before - len(self.df)

        self.summary["duplicates_removed"] = removed

    # --------------------------------------------------
    # LABEL DISTRIBUTION
    # --------------------------------------------------

    def label_distribution(self) -> None:
        if "label" not in self.df.columns:
            logger.warning("Skipping label distribution: label column missing")
            return

        counts = self.df["label"].value_counts().to_dict()

        self.summary["label_distribution"] = counts

        sns.countplot(x="label", data=self.df)

        plt.title("Label Distribution")

        plt.savefig(self.output_dir / "label_distribution.png")
        plt.close()

    # --------------------------------------------------
    # TEXT STATISTICS
    # --------------------------------------------------

    def text_statistics(self) -> None:
        if "text" not in self.df.columns:
            logger.warning("Skipping text statistics: text column missing")
            return

        self.df["text_length"] = self.df["text"].str.len()

        self.df["word_count"] = self.df["text"].apply(
            lambda x: len(str(x).split())
        )

        self.df["sentence_count"] = self.df["text"].apply(
            lambda x: len(re.findall(r"[.!?]", str(x)))
        )

        stats = self.df[
            ["text_length", "word_count", "sentence_count"]
        ].describe().to_dict()

        self.summary["text_statistics"] = stats

    # --------------------------------------------------
    # DOCUMENT LENGTH ANALYSIS
    # --------------------------------------------------

    def document_length_by_label(self) -> None:
        if "word_count" not in self.df.columns or "label" not in self.df.columns:
            return

        sns.histplot(
            data=self.df,
            x="word_count",
            hue="label",
            bins=40
        )

        plt.title("Document Length by Label")

        plt.savefig(self.output_dir / "doc_length_by_label.png")
        plt.close()

    # --------------------------------------------------
    # SKEWNESS ANALYSIS
    # --------------------------------------------------

    def skewness(self) -> None:
        if "text_length" not in self.df.columns or len(self.df) == 0:
            return

        skew = self.df["text_length"].skew()

        self.summary["text_length_skewness"] = float(skew)

        sns.histplot(self.df["text_length"], bins=50)

        plt.title("Text Length Distribution")

        plt.savefig(self.output_dir / "text_length_distribution.png")
        plt.close()

    # --------------------------------------------------
    # OUTLIER DETECTION
    # --------------------------------------------------

    def detect_outliers(self) -> None:
        if "text_length" not in self.df.columns or len(self.df) == 0:
            return

        Q1 = self.df["text_length"].quantile(0.25)
        Q3 = self.df["text_length"].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = self.df[
            (self.df["text_length"] < lower)
            | (self.df["text_length"] > upper)
        ]

        self.summary["outlier_count"] = len(outliers)

        sns.boxplot(x=self.df["text_length"])

        plt.title("Text Length Outliers")

        plt.savefig(self.output_dir / "text_length_outliers.png")
        plt.close()

    # --------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------

    def feature_engineering(self) -> None:
        if "text" not in self.df.columns:
            return

        def avg_word_len(text):

            words = str(text).split()

            if not words:
                return 0

            return np.mean([len(w) for w in words])

        self.df["avg_word_length"] = self.df["text"].apply(avg_word_len)

        self.df["uppercase_ratio"] = self.df["text"].apply(
            lambda x:
            sum(1 for c in str(x) if c.isupper()) /
            max(len(str(x)), 1)
        )

    # --------------------------------------------------
    # VOCABULARY ANALYSIS
    # --------------------------------------------------

    def vocabulary_analysis(self) -> None:
        if "text" not in self.df.columns:
            return

        words = []

        for text in self.df["text"]:

            words.extend(
                re.findall(r"\b\w+\b", str(text).lower())
            )

        vocab = set(words)

        diversity = len(vocab) / max(len(words), 1)

        self.summary["vocab_size"] = len(vocab)
        self.summary["lexical_diversity"] = float(diversity)

    # --------------------------------------------------
    # WORD FREQUENCY
    # --------------------------------------------------

    def word_frequency(self, top_n: int = 20) -> None:
        if "text" not in self.df.columns:
            return

        words = []

        for text in self.df["text"]:

            tokens = re.findall(r"\b[a-z]{3,}\b", str(text).lower())

            tokens = [w for w in tokens if w not in STOPWORDS]

            words.extend(tokens)

        freq = Counter(words)

        common = freq.most_common(top_n)

        words_plot = [w for w, _ in common]
        counts = [c for _, c in common]

        sns.barplot(x=counts, y=words_plot)

        plt.title("Top Words")

        plt.savefig(self.output_dir / "word_frequency.png")
        plt.close()

    # --------------------------------------------------
    # NGRAM ANALYSIS
    # --------------------------------------------------

    def ngram_analysis(self, n: int = 2) -> None:
        if "text" not in self.df.columns or len(self.df) == 0:
            return

        vectorizer = CountVectorizer(
            stop_words="english",
            ngram_range=(n, n),
            max_features=15
        )

        try:
            X = vectorizer.fit_transform(self.df["text"])
        except ValueError:
            logger.warning("Skipping %s-gram analysis: empty vocabulary", n)
            return

        counts = np.asarray(X.sum(axis=0)).ravel()

        vocab = vectorizer.get_feature_names_out()

        pairs = list(zip(vocab, counts))
        pairs.sort(key=lambda x: x[1], reverse=True)

        words = [p[0] for p in pairs]
        values = [p[1] for p in pairs]

        sns.barplot(x=values, y=words)

        plt.title(f"Top {n}-grams")

        plt.savefig(self.output_dir / f"{n}gram_frequency.png")
        plt.close()

    # --------------------------------------------------
    # TF-IDF KEYWORDS
    # --------------------------------------------------

    def tfidf_keywords(self) -> None:
        if "text" not in self.df.columns or len(self.df) == 0:
            self.summary["top_tfidf_words"] = []
            return

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000
        )

        try:
            X = vectorizer.fit_transform(self.df["text"])
        except ValueError:
            logger.warning("Skipping TF-IDF keyword extraction: empty vocabulary")
            self.summary["top_tfidf_words"] = []
            return

        sums = np.asarray(X.sum(axis=0)).ravel()

        words = vectorizer.get_feature_names_out()

        pairs = list(zip(words, sums))
        pairs.sort(key=lambda x: x[1], reverse=True)

        self.summary["top_tfidf_words"] = pairs[:20]

    # --------------------------------------------------
    # WORD CLOUDS
    # --------------------------------------------------

    def generate_wordclouds(self) -> None:
        if "text" not in self.df.columns or "label" not in self.df.columns:
            return

        fake_text = " ".join(
            self.df[self.df["label"] == 1]["text"]
        )

        real_text = " ".join(
            self.df[self.df["label"] == 0]["text"]
        )

        if not fake_text.strip() or not real_text.strip():
            logger.warning("Skipping wordcloud generation: insufficient class text")
            return

        fake_wc = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate(fake_text)

        real_wc = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate(real_text)

        plt.imshow(fake_wc)
        plt.axis("off")
        plt.savefig(self.output_dir / "fake_wordcloud.png")
        plt.close()

        plt.imshow(real_wc)
        plt.axis("off")
        plt.savefig(self.output_dir / "real_wordcloud.png")
        plt.close()

    # --------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------

    def save_report(self) -> None:

        report_path = self.report_dir / "eda_summary.json"

        with report_path.open("w", encoding="utf-8") as f:

            json.dump(self.summary, f, indent=2)

        logger.info(f"EDA summary saved: {report_path}")

    # --------------------------------------------------
    # RUN FULL PIPELINE
    # --------------------------------------------------

    def run(self) -> None:

        logger.info("Starting EDA pipeline")

        self.dataset_summary()
        self.check_data_types()
        self.handle_missing()
        self.remove_duplicates()

        self.label_distribution()
        self.text_statistics()
        self.document_length_by_label()
        self.skewness()
        self.detect_outliers()

        self.feature_engineering()

        self.vocabulary_analysis()
        self.word_frequency()

        self.ngram_analysis(2)
        self.ngram_analysis(3)

        self.tfidf_keywords()
        self.generate_wordclouds()

        self.save_report()

        logger.info("EDA pipeline completed")


def run_eda(csv_path: str) -> None:

    df = pd.read_csv(csv_path)

    eda = FakeNewsEDA(df)

    eda.run()

"""
Professional EDA Module for Fake News Dataset
Includes Level-1 NLP EDA features

Features
✔ Data types validation
✔ Missing value handling
✔ Duplicate removal
✔ Distribution analysis
✔ Skewness check
✔ Outlier inspection
✔ Target relationship
✔ Multicollinearity check
✔ Feature engineering
✔ Text statistics
✔ Vocabulary analysis
✔ Word frequency
✔ N-gram analysis
✔ Wordcloud generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


class FakeNewsEDA:

    def __init__(self, df, output_dir="reports/figures"):

        self.df = df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # DATA TYPE CHECK
    # --------------------------------------------------
    def check_data_types(self):

        logger.info("\n=== DATA TYPES ===")
        logger.info(self.df.dtypes)

        if self.df["label"].dtype == "object":
            self.df["label"] = self.df["label"].map({"real":0,"fake":1})

    # --------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------
    def handle_missing(self):

        logger.info("\n=== MISSING VALUES ===")
        logger.info(self.df.isnull().sum())

        self.df = self.df.dropna(subset=["text"])
        self.df = self.df.fillna("")

    # --------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------
    def remove_duplicates(self):

        logger.info("\n=== DUPLICATES ===")

        before = len(self.df)

        self.df = self.df.drop_duplicates()

        logger.info(f"Removed {before-len(self.df)} duplicates")

    # --------------------------------------------------
    # LABEL DISTRIBUTION
    # --------------------------------------------------
    def label_distribution(self):

        logger.info("\n=== LABEL DISTRIBUTION ===")

        counts = self.df["label"].value_counts()
        logger.info(counts)

        sns.countplot(x="label", data=self.df)

        plt.title("Label Distribution")

        plt.savefig(self.output_dir / "label_distribution.png")
        plt.close()

    # --------------------------------------------------
    # TEXT STATISTICS
    # --------------------------------------------------
    def text_statistics(self):

        logger.info("\n=== TEXT STATISTICS ===")

        self.df["text_length"] = self.df["text"].str.len()

        self.df["word_count"] = self.df["text"].apply(
            lambda x: len(str(x).split())
        )

        self.df["sentence_count"] = self.df["text"].apply(
            lambda x: len(re.findall(r"[.!?]", str(x)))
        )

        logger.info(self.df[[
            "text_length",
            "word_count",
            "sentence_count"
        ]].describe())

    # --------------------------------------------------
    # SKEWNESS
    # --------------------------------------------------
    def skewness(self):

        logger.info("\n=== SKEWNESS ===")

        skew = self.df["text_length"].skew()

        logger.info(f"text_length skewness: {skew:.2f}")

        sns.histplot(self.df["text_length"], bins=50)

        plt.title("Text Length Distribution")

        plt.savefig(self.output_dir / "text_length_distribution.png")
        plt.close()

    # --------------------------------------------------
    # OUTLIERS
    # --------------------------------------------------
    def detect_outliers(self):

        logger.info("\n=== OUTLIERS ===")

        Q1 = self.df["text_length"].quantile(0.25)
        Q3 = self.df["text_length"].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR

        outliers = self.df[
            (self.df["text_length"] < lower) |
            (self.df["text_length"] > upper)
        ]

        logger.info(f"Outliers: {len(outliers)}")

        sns.boxplot(x=self.df["text_length"])

        plt.title("Text Length Outliers")

        plt.savefig(self.output_dir / "text_length_outliers.png")
        plt.close()

    # --------------------------------------------------
    # TARGET RELATIONSHIP
    # --------------------------------------------------
    def target_relationship(self):

        sns.boxplot(x="label", y="text_length", data=self.df)

        plt.title("Text Length vs Label")

        plt.savefig(self.output_dir / "text_length_vs_label.png")
        plt.close()

    # --------------------------------------------------
    # MULTICOLLINEARITY
    # --------------------------------------------------
    def correlation_matrix(self):

        self.df["word_count"] = self.df["text"].apply(
            lambda x: len(str(x).split())
        )

        numeric = self.df[["text_length","word_count"]]

        corr = numeric.corr()

        sns.heatmap(corr, annot=True)

        plt.title("Feature Correlation")

        plt.savefig(self.output_dir / "correlation_matrix.png")
        plt.close()

    # --------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------
    def feature_engineering(self):

        logger.info("\n=== FEATURE ENGINEERING ===")

        def _avg_word_len(text):
            words = str(text).split()
            if not words:
                return 0.0
            return float(np.mean([len(w) for w in words]))

        self.df["avg_word_length"] = self.df["text"].apply(_avg_word_len)

        self.df["uppercase_ratio"] = self.df["text"].apply(
            lambda x: (
                sum(1 for c in str(x) if c.isupper()) /
                max(len(str(x)), 1)
            )
        )

        logger.info("New features created")

    # --------------------------------------------------
    # VOCABULARY ANALYSIS
    # --------------------------------------------------
    def vocabulary_analysis(self):

        logger.info("\n=== VOCABULARY ===")

        words = []

        for text in self.df["text"]:
            words.extend(re.findall(r"\b\w+\b", str(text).lower()))

        vocab = set(words)

        logger.info(f"Total tokens: {len(words)}")
        logger.info(f"Vocabulary size: {len(vocab)}")

    # --------------------------------------------------
    # WORD FREQUENCY
    # --------------------------------------------------
    def word_frequency(self, top_n=20):

        logger.info("\n=== WORD FREQUENCY ===")

        words = []

        for text in self.df["text"]:
            words.extend(re.findall(r"\b[a-z]{3,}\b", str(text).lower()))

        freq = Counter(words)

        common = freq.most_common(top_n)

        logger.info(common)

        words_plot = [w for w,_ in common]
        counts = [c for _,c in common]

        sns.barplot(x=counts,y=words_plot)

        plt.title("Top Words")

        plt.savefig(self.output_dir / "word_frequency.png")
        plt.close()

    # --------------------------------------------------
    # NGRAM ANALYSIS
    # --------------------------------------------------
    def ngram_analysis(self,n=2):

        logger.info(f"\n=== {n}-GRAM ANALYSIS ===")

        vectorizer = CountVectorizer(
            stop_words="english",
            ngram_range=(n,n),
            max_features=20
        )

        X = vectorizer.fit_transform(self.df["text"])

        counts = np.asarray(X.sum(axis=0)).ravel()

        vocab = vectorizer.get_feature_names_out()

        pairs = list(zip(vocab,counts))

        pairs.sort(key=lambda x:x[1], reverse=True)

        logger.info(pairs[:10])

    # --------------------------------------------------
    # WORDCLOUD
    # --------------------------------------------------
    def generate_wordclouds(self):

        logger.info("\n=== WORDCLOUDS ===")

        text = " ".join(self.df["text"].astype(str))

        wc = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate(text)

        plt.imshow(wc)
        plt.axis("off")

        plt.savefig(self.output_dir / "wordcloud.png")
        plt.close()

    # --------------------------------------------------
    # RUN FULL EDA
    # --------------------------------------------------
    def run(self):

        logger.info("\nRUNNING EDA")

        self.check_data_types()
        self.handle_missing()
        self.remove_duplicates()
        self.label_distribution()
        self.text_statistics()
        self.skewness()
        self.detect_outliers()
        self.target_relationship()
        self.correlation_matrix()
        self.feature_engineering()

        # LEVEL 1 ADDITIONS
        self.vocabulary_analysis()
        self.word_frequency()
        self.ngram_analysis(2)
        self.ngram_analysis(3)
        self.generate_wordclouds()

        logger.info("\nEDA COMPLETE")


def run_eda(csv_path):

    df = pd.read_csv(csv_path)

    eda = FakeNewsEDA(df)

    eda.run()


if __name__ == "__main__":

    import sys

    if len(sys.argv) > 1:
        run_eda(sys.argv[1])

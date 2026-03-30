from __future__ import annotations

from pathlib import Path
from typing import Dict
import pandas as pd


# =======================================================
# PATH CONFIGURATION
# =======================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SEMEVAL_DIR = Path(r"C:\Users\bhava\Downloads\SemEval")

TRAIN_FILE = SEMEVAL_DIR / "SemEval-Train.csv"
DEV_FILE = SEMEVAL_DIR / "SemEval-Validation.csv"
TEST_FILE = SEMEVAL_DIR / "SemEval-Test.csv"

OUTPUT_FILE = PROJECT_ROOT / "data" / "interim" / "semeval_emotions.csv"


# =======================================================
# SEMEVAL LABEL MAPPING
# =======================================================

SEMEVAL_LABELS: Dict[int, str] = {
    0: "anger",
    1: "anticipation",
    2: "disgust",
    3: "fear",
    4: "joy",
    5: "love",
    6: "optimism",
    7: "pessimism",
    8: "sadness",
    9: "surprise",
    10: "trust"
}

LABEL_COLUMNS = list(SEMEVAL_LABELS.values())


# =======================================================
# LOAD SEMEVAL EXCEL
# =======================================================

def load_semeval(file_path):

    df = pd.read_csv(file_path)

    text_candidates = ["text", "Tweet", "tweet", "Sentence"]

    for col in text_candidates:
        if col in df.columns:
            df = df.rename(columns={col: "text"})
            break

    if "text" not in df.columns:
        raise ValueError(
            f"No text column found. Available columns: {list(df.columns)}"
        )

    # convert numeric label columns to int
    for col in df.columns:
        if str(col).isdigit():
            df.rename(columns={col: int(col)}, inplace=True)

    return df

# =======================================================
# CONVERT MULTI-COLUMN LABELS → ID STRING
# =======================================================

def convert_labels(df: pd.DataFrame):

    def build_label_string(row):

        labels = []
    
        for idx in SEMEVAL_LABELS.keys():
    
            if idx in row.index and row[idx] == 1:
                labels.append(str(idx))
    
        return ",".join(labels)

    df["labels"] = df.apply(build_label_string, axis=1)

    return df


# =======================================================
# MERGE DATASETS
# =======================================================

def merge_semeval():

    print("\nLoading SemEval datasets...")

    train = load_semeval(TRAIN_FILE)
    dev = load_semeval(DEV_FILE)
    test = load_semeval(TEST_FILE)

    print("Train size:", len(train))
    print("Validation size:", len(dev))
    print("Test size:", len(test))

    merged = pd.concat([train, dev, test], ignore_index=True)

    merged = merged.drop_duplicates(subset=["text"])

    merged = convert_labels(merged)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    merged[["text", "labels"]].to_csv(OUTPUT_FILE, index=False)

    print("Merged dataset size:", len(merged))
    print("Saved merged dataset:", OUTPUT_FILE)

    return merged


# =======================================================
# ANALYZE EMOTION DISTRIBUTION
# =======================================================

def analyze_emotions(df: pd.DataFrame):

    counts = {v: 0 for v in SEMEVAL_LABELS.values()}

    for labels in df["labels"]:

        if pd.isna(labels) or labels == "":
            continue

        label_ids = labels.split(",")

        for lid in label_ids:

            idx = int(lid)
            emotion = SEMEVAL_LABELS[idx]

            counts[emotion] += 1

    result = (
        pd.DataFrame(list(counts.items()), columns=["emotion", "count"])
        .sort_values("count", ascending=False)
    )

    print("\nSemEval Emotion Distribution:\n")
    print(result)

    return result


# =======================================================
# MAIN PIPELINE
# =======================================================

def main():

    merged_df = merge_semeval()

    analyze_emotions(merged_df)


if __name__ == "__main__":
    main()
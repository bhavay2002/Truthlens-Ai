import pandas as pd
import logging
from pathlib import Path
import json
from src.utils.settings import load_settings

logger = logging.getLogger(__name__)
SETTINGS = load_settings()
RAW_PATH = SETTINGS.data.raw_dir
INTERIM_PATH = SETTINGS.data.interim_dir


def load_isot():
    """Load ISOT dataset"""

    fake = pd.read_csv(RAW_PATH / "isot" / "Fake.csv")
    true = pd.read_csv(RAW_PATH / "isot" / "True.csv")

    fake["label"] = 1
    true["label"] = 0

    df = pd.concat([fake, true], ignore_index=True)

    df = df.rename(columns={
        "title": "title",
        "text": "text"
    })

    return df[["title", "text", "label"]]


def load_liar():
    """Load LIAR dataset"""

    liar_path = RAW_PATH / "liar_dataset" / "train.tsv"
    liar_columns = [
        "id",
        "label",
        "statement",
        "subject",
        "speaker",
        "speaker_job_title",
        "state_info",
        "party_affiliation",
        "barely_true_counts",
        "false_counts",
        "half_true_counts",
        "mostly_true_counts",
        "pants_on_fire_counts",
        "context",
    ]

    # LIAR TSV is typically headerless; force names to avoid KeyError on "label".
    df = pd.read_csv(liar_path, sep="\t", header=None, names=liar_columns)

    if "label" not in df.columns or "statement" not in df.columns:
        raise ValueError("LIAR dataset missing required columns: label/statement")

    fake_labels = ["false", "pants-fire", "barely-true"]

    df["label"] = df["label"].astype(str).str.strip().str.lower().apply(
        lambda x: 1 if x in fake_labels else 0
    )

    df = df.rename(columns={"statement": "text"})

    df["title"] = ""
    df["text"] = df["text"].astype(str).fillna("").str.strip()

    return df[["title", "text", "label"]]


def load_fakenewsnet():
    """Load FakeNewsNet dataset"""

    rows = []
    source_roots = [RAW_PATH / "FakeNewsNet", RAW_PATH / "fakenewsnet"]
    source_root = next((path for path in source_roots if path.exists()), source_roots[0])

    for file in source_root.rglob("news content.json"):

        try:
            with file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            rows.append({
                "title": data.get("title"),
                "text": data.get("text"),
                "label": 1 if "fake" in str(file) else 0
            })

        except Exception:
            continue

    return pd.DataFrame(rows)


def merge_datasets():
    """Merge all datasets"""

    logger.info("Loading ISOT dataset...")
    isot = load_isot()

    logger.info("Loading LIAR dataset...")
    liar = load_liar()

    logger.info("Loading FakeNewsNet dataset...")
    fakenewsnet = load_fakenewsnet()

    df = pd.concat([isot, liar, fakenewsnet], ignore_index=True)

    logger.info("Total samples: %s", len(df))

    return df


def save_dataset(df):
    """Save merged dataset"""

    INTERIM_PATH.mkdir(parents=True, exist_ok=True)

    output_path = INTERIM_PATH / "merged_dataset.csv"

    df.to_csv(output_path, index=False)

    logger.info("Merged dataset saved to: %s", output_path)


if __name__ == "__main__":

    dataset = merge_datasets()

    save_dataset(dataset)

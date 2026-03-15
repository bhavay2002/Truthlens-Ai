import pandas as pd
import logging
from pathlib import Path
import json
from src.utils.settings import load_settings

logger = logging.getLogger(__name__)
SETTINGS = load_settings()
RAW_PATH = SETTINGS.data.raw_dir
INTERIM_PATH = SETTINGS.data.interim_dir
REQUIRED_COLUMNS = ["title", "text", "label"]


def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()

    if "title" not in prepared.columns:
        prepared["title"] = ""
    if "text" not in prepared.columns:
        raise ValueError("Dataset is missing required 'text' column")
    if "label" not in prepared.columns:
        raise ValueError("Dataset is missing required 'label' column")

    prepared["title"] = prepared["title"].astype(str).fillna("").str.strip()
    prepared["text"] = prepared["text"].astype(str).fillna("").str.strip()
    prepared["label"] = prepared["label"].astype(int)

    prepared = prepared[prepared["text"].str.len() > 0].reset_index(drop=True)
    return prepared[REQUIRED_COLUMNS]


def load_isot():
    """Load ISOT dataset"""

    fake_path = RAW_PATH / "isot" / "Fake.csv"
    true_path = RAW_PATH / "isot" / "True.csv"

    if not fake_path.exists():
        raise FileNotFoundError(f"Missing ISOT fake file: {fake_path}")
    if not true_path.exists():
        raise FileNotFoundError(f"Missing ISOT true file: {true_path}")

    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    fake["label"] = 1
    true["label"] = 0

    df = pd.concat([fake, true], ignore_index=True)
    return _ensure_schema(df)


def load_liar():
    """Load LIAR dataset"""

    liar_path = RAW_PATH / "liar_dataset" / "train.tsv"
    if not liar_path.exists():
        raise FileNotFoundError(f"Missing LIAR file: {liar_path}")

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

    fake_labels = {"false", "pants-fire", "pants on fire", "barely-true"}

    df["label"] = df["label"].astype(str).str.strip().str.lower().apply(
        lambda x: 1 if x in fake_labels else 0
    )

    df = df.rename(columns={"statement": "text"})

    df["title"] = ""
    df["text"] = df["text"].astype(str).fillna("").str.strip()

    return _ensure_schema(df)


def load_fakenewsnet():
    """Load FakeNewsNet dataset"""

    rows = []
    source_roots = [RAW_PATH / "FakeNewsNet", RAW_PATH / "fakenewsnet"]
    source_root = next((path for path in source_roots if path.exists()), source_roots[0])
    if not source_root.exists():
        logger.warning("FakeNewsNet source directory not found: %s", source_root)
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    for file in source_root.rglob("news content.json"):

        try:
            with file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            rows.append({
                "title": data.get("title"),
                "text": data.get("text"),
                "label": 1 if "fake" in str(file).lower() else 0
            })

        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    return _ensure_schema(pd.DataFrame(rows))


def merge_datasets():
    """Merge all datasets"""

    datasets = []

    for name, loader in [
        ("ISOT", load_isot),
        ("LIAR", load_liar),
        ("FakeNewsNet", load_fakenewsnet),
    ]:
        try:
            logger.info("Loading %s dataset...", name)
            loaded = loader()
            if loaded.empty:
                logger.warning("%s dataset is empty and will be skipped", name)
                continue
            datasets.append(loaded)
        except FileNotFoundError as error:
            logger.warning("Skipping %s dataset: %s", name, error)
        except Exception as error:
            logger.warning("Skipping %s dataset due to load error: %s", name, error)

    if not datasets:
        raise FileNotFoundError("No datasets could be loaded from configured raw paths")

    df = pd.concat(datasets, ignore_index=True)
    df = _ensure_schema(df)

    logger.info("Total samples after merge: %s", len(df))

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

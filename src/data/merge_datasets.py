import pandas as pd
from pathlib import Path
import json

RAW_PATH = Path("data/raw")
INTERIM_PATH = Path("data/interim")


def load_isot():
    """Load ISOT dataset"""

    fake = pd.read_csv(RAW_PATH / "isot/Fake.csv")
    true = pd.read_csv(RAW_PATH / "isot/True.csv")

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

    df = pd.read_csv(RAW_PATH / "liar_dataset/train.tsv", sep="\t")

    fake_labels = ["false", "pants-fire", "barely-true"]

    df["label"] = df["label"].apply(
        lambda x: 1 if x in fake_labels else 0
    )

    df = df.rename(columns={"statement": "text"})

    df["title"] = ""

    return df[["title", "text", "label"]]


def load_fakenewsnet():
    """Load FakeNewsNet dataset"""

    rows = []

    for file in (RAW_PATH / "fakenewsnet").rglob("news content.json"):

        try:
            with open(file, "r", encoding="utf-8") as f:
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

    print("Loading ISOT dataset...")
    isot = load_isot()

    print("Loading LIAR dataset...")
    liar = load_liar()

    print("Loading FakeNewsNet dataset...")
    fakenewsnet = load_fakenewsnet()

    df = pd.concat([isot, liar, fakenewsnet], ignore_index=True)

    print("Total samples:", len(df))

    return df


def save_dataset(df):
    """Save merged dataset"""

    INTERIM_PATH.mkdir(parents=True, exist_ok=True)

    output_path = INTERIM_PATH / "merged_dataset.csv"

    df.to_csv(output_path, index=False)

    print(f"Merged dataset saved to: {output_path}")


if __name__ == "__main__":

    dataset = merge_datasets()

    save_dataset(dataset)
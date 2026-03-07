import pandas as pd

def extract_metadata_features(df):

    df["title_length"] = df["title"].apply(lambda x: len(str(x)))
    df["text_length"] = df["text"].apply(lambda x: len(str(x)))

    if "author" in df.columns:
        df["has_author"] = df["author"].notnull().astype(int)

    return df

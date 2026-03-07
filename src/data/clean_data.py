import pandas as pd
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text


def clean_dataframe(df):
    df = df.drop_duplicates()
    df = df.dropna(subset=["text"])

    df["text"] = df["text"].apply(clean_text)

    return df

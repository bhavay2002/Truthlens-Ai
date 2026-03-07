import pandas as pd

def load_csv(path: str):
    df = pd.read_csv(path)
    return df


def merge_datasets(fake_path, real_path):
    fake = pd.read_csv(fake_path)
    real = pd.read_csv(real_path)

    fake["label"] = 1
    real["label"] = 0

    df = pd.concat([fake, real], ignore_index=True)
    return df

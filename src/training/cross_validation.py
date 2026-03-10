from sklearn.model_selection import StratifiedKFold
import numpy as np


def cross_validate_model(df, train_function, n_splits=5):

    X = df["text"]
    y = df["label"]

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        trainer, val_dataset = train_function(train_df)

        metrics = trainer.evaluate(val_dataset)

        scores.append(metrics["eval_loss"])

    return np.mean(scores)
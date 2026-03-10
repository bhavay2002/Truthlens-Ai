import optuna
from transformers import Trainer, TrainingArguments
from transformers import RobertaForSequenceClassification
from src.utils.config_loader import get_config_value, get_path, load_config

CONFIG = load_config()
MODEL_NAME = get_config_value(CONFIG, "model", "name", default="roberta-base")
MODELS_DIR = get_path(CONFIG, "paths", "models_dir", default="models")


def model_init():
    return RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )


def objective(trial, train_dataset, val_dataset):

    learning_rate = trial.suggest_float(
        "learning_rate", 1e-6, 5e-5, log=True
    )

    batch_size = trial.suggest_categorical(
        "batch_size", [8, 16]
    )

    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=2,
        evaluation_strategy="epoch",
        save_strategy="no"
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()

    metrics = trainer.evaluate()

    return metrics["eval_loss"]


def run_optuna(train_dataset, val_dataset):

    study = optuna.create_study(direction="minimize")

    study.optimize(
        lambda trial: objective(
            trial,
            train_dataset,
            val_dataset
        ),
        n_trials=10
    )

    return study.best_params

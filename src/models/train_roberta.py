import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(df):
    """
    Train RoBERTa model with proper train/validation/test splits
    """
    try:
        logger.info("Starting model training...")
        
        # Split data: 70% train, 15% validation, 15% test
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
        
        logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}")

        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        # Create datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)

        def tokenize(example):
            return tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=256
            )

        train_dataset = train_dataset.map(tokenize, batched=True)
        val_dataset = val_dataset.map(tokenize, batched=True)
        test_dataset = test_dataset.map(tokenize, batched=True)

        # Set format for PyTorch
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=2
        )

        training_args = TrainingArguments(
            output_dir="./models",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=2,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs",
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            save_total_limit=2
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        logger.info("Training model...")
        trainer.train()

        # Save model and tokenizer
        logger.info("Saving model...")
        model.save_pretrained("./models/roberta_model")
        tokenizer.save_pretrained("./models/roberta_model")
        
        # Save test dataset for later evaluation
        test_df.to_csv("./data/processed/test_set.csv", index=False)
        logger.info("Training complete!")
        
        return trainer, test_dataset
    
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

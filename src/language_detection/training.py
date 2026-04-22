from pathlib import Path
from typing import Dict

import evaluate
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def train_model(
    csv_path: str,
    output_dir: str = "models/xlm_r_lang_model",
    model_name: str = "xlm-roberta-base",
    text_column: str = "text",
    label_column: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
):
    df = pd.read_csv(csv_path)
    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(f"CSV must include '{text_column}' and '{label_column}' columns.")

    labels = sorted(df[label_column].unique().tolist())
    label2id: Dict[str, int] = {label: i for i, label in enumerate(labels)}
    id2label: Dict[int, str] = {i: label for label, i in label2id.items()}

    train_df, val_df = train_test_split(
        df[[text_column, label_column]],
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_column],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df[label_column] = train_df[label_column].map(label2id)
    val_df[label_column] = val_df[label_column].map(label2id)

    train_dataset = Dataset.from_pandas(train_df.rename(columns={text_column: "text", label_column: "label"}))
    val_dataset = Dataset.from_pandas(val_df.rename(columns={text_column: "text", label_column: "label"}))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True)

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels_np = eval_pred
        preds = predictions.argmax(axis=-1)
        return accuracy.compute(predictions=preds, references=labels_np)

    training_args = TrainingArguments(
        output_dir=str(Path(output_dir).parent / "checkpoints"),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    return output_path

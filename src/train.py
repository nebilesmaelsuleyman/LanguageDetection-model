import evaluate
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from src.config import MODEL_CHECKPOINT, SAVE_PATH, DEVICE, set_seed
from src.data import load_and_validate, prepare_dataset
import os

def tokenize_function(examples, tokenizer):
    result = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    result["labels"] = examples["label_idx"]
    return result

def train_model(data_path):
    set_seed()
    
    # Load and prepare data
    raw_df = load_and_validate(data_path)
    if raw_df.empty:
        raise ValueError("No data found to train.")
        
    ds, id2label, label2id, label_encoder = prepare_dataset(raw_df)
    
    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    tokenized_ds = ds.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=ds["train"].column_names)
    
    # Model Setup
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    ).to(DEVICE)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    
    # Save Model
    os.makedirs(SAVE_PATH, exist_ok=True)
    trainer.save_model(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    
    return trainer, tokenized_ds, label_encoder

if __name__ == "__main__":
    from src.config import DATA_PATH
    train_model(DATA_PATH)

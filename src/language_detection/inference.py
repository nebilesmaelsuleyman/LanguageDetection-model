from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class LanguageDetector:
    def __init__(self, model_dir: str = "models/xlm_r_lang_model"):
        model_path = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def predict(self, text: str) -> Tuple[str, float]:
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self.model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs).item())
        label = self.model.config.id2label[str(pred_id)] if isinstance(self.model.config.id2label, dict) and str(pred_id) in self.model.config.id2label else self.model.config.id2label[pred_id]
        confidence = float(probs[pred_id].item())
        return label, confidence

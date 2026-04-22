from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import DEFAULT_MODEL_DIR


class LanguageDetector:
    """Load a fine-tuned language detection model and run predictions.

    Args:
        model_dir: Directory containing a saved Hugging Face model/tokenizer.
    """

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        model_path = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def predict(self, text: str) -> Tuple[str, float]:
        """Predict the language label for text and return (label, confidence)."""
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self.model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs).item())
        label = self._label_for_id(pred_id)
        confidence = float(probs[pred_id].item())
        return label, confidence

    def supported_labels(self) -> List[str]:
        """Return model labels ordered by their class id."""
        id2label = self.model.config.id2label
        if not isinstance(id2label, dict):
            raise TypeError("Expected id2label to be a dict mapping class ids to labels.")
        return [label for _, label in sorted(id2label.items(), key=lambda item: int(item[0]))]

    def _label_for_id(self, pred_id: int) -> str:
        id2label = self.model.config.id2label
        if isinstance(id2label, dict):
            if pred_id in id2label:
                return id2label[pred_id]
            if str(pred_id) in id2label:
                return id2label[str(pred_id)]
            raise KeyError(f"Missing label mapping for class id {pred_id}.")
        return id2label[pred_id]

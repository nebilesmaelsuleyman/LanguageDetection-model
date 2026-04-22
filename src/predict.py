import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import SAVE_PATH, DEVICE

class LanguageDetector:
    def __init__(self, model_path=SAVE_PATH):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True).to(DEVICE)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.nn.functional.softmax(logits, dim=-1)
        conf, idx = torch.max(probs, dim=-1)

        return self.id2label[idx.item()], conf.item()

if __name__ == "__main__":
    detector = LanguageDetector()
    lang, score = detector.predict("akkam bultan")
    print(f"Predicted: {lang} (Confidence: {score:.4f})")

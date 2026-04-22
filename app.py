import os

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = os.getenv("MODEL_DIR", "xlm_r_lang_model")


def _load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


try:
    TOKENIZER, MODEL = _load_model()
    LOAD_ERROR = None
except Exception as exc:  # pragma: no cover - depends on local model artifacts
    TOKENIZER, MODEL = None, None
    LOAD_ERROR = str(exc)


EXPECTED_MODEL_LABELS = {0: "Amharic", 1: "Afan Oromo", 2: "English"}


def predict_language(text: str) -> str:
    if not text or not text.strip():
        return "Please enter text."

    if LOAD_ERROR is not None or TOKENIZER is None or MODEL is None:
        return f"Model could not be loaded from '{MODEL_DIR}'. Error: {LOAD_ERROR}"

    encoded = TOKENIZER(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = MODEL(**encoded).logits
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = torch.argmax(probabilities).item()
    confidence = float(probabilities[predicted_index].item())

    config_labels = getattr(MODEL.config, "id2label", None)
    id2label = config_labels if config_labels is not None else EXPECTED_MODEL_LABELS
    language = id2label.get(str(predicted_index), id2label.get(predicted_index, f"Unknown ({predicted_index})"))

    return f"Predicted language: {language} (confidence: {confidence:.2%})"


def build_app() -> gr.Interface:
    return gr.Interface(
        fn=predict_language,
        inputs=gr.Textbox(lines=4, placeholder="Enter text..."),
        outputs=gr.Textbox(label="Prediction"),
        title="Multilingual Language Detector",
        description="Detect whether text is Amharic, Afan Oromo, or English.",
    )


if __name__ == "__main__":
    build_app().launch()

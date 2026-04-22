import os

import gradio as gr

from src.language_detection.inference import LanguageDetector


MODEL_DIR = os.getenv("MODEL_DIR", "models/xlm_r_lang_model")
detector = LanguageDetector(model_dir=MODEL_DIR)
supported_languages = ", ".join(
    str(label) for _, label in sorted(detector.model.config.id2label.items(), key=lambda item: int(item[0]))
)


def predict_language(text: str):
    if not text or not text.strip():
        return "Please enter text.", 0.0
    label, confidence = detector.predict(text)
    return label, confidence


app = gr.Interface(
    fn=predict_language,
    inputs=gr.Textbox(label="Input Text"),
    outputs=[
        gr.Textbox(label="Predicted Language"),
        gr.Number(label="Confidence"),
    ],
    title="Multilingual Language Detector",
    description=f"Detects: {supported_languages}",
)


if __name__ == "__main__":
    app.launch()

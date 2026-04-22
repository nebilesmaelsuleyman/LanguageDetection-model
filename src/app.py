import gradio as gr
from src.predict import LanguageDetector

# Initialize the detector
detector = LanguageDetector()

def gradio_predict(text):
    lang, score = detector.predict(text)
    # Return probabilities for the label component
    return {lang: float(score)}

def main():
    demo = gr.Interface(
        fn=gradio_predict,
        inputs=gr.Textbox(lines=2, placeholder="Enter text here...", label="Input Text"),
        outputs=gr.Label(num_top_classes=3, label="Prediction"),
        title="Multilingual Language Detector (XLM-RoBERTa)",
        description="Enter text in Amharic, Afan Oromo, or English to identify the language.",
        examples=["akkam bultan", "እንዴት ነህ?", "How are you doing today?"]
    )

    demo.launch(share=True, debug=False)

if __name__ == "__main__":
    main()

# Multilingual Language Detector

This repository contains a multilingual language detection app based on XLM-RoBERTa.

The model is trained to classify text into Amharic, Afan Oromo, or English.

## Project structure

- `app.py` - Gradio inference app
- `requirements.txt` - Python dependencies
- `xlm_r_lang_model/` - local Hugging Face model artifacts (`config.json`, tokenizer files, model weights)

## Usage (Hugging Face Spaces / local)

1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place model files in `xlm_r_lang_model/` (or set `MODEL_DIR` to another path).
4. Run the app:
   ```bash
   python app.py
   ```

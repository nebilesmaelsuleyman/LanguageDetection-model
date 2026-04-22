# Multilingual Language Detector

This repository contains a deployed version of a multilingual language detection model based on XLM-RoBERTa.

The model is trained to classify text into Amharic, Afan Oromo, or English.

## Usage (Hugging Face Spaces)

This application can be easily deployed to Hugging Face Spaces. The `app.py` script serves a Gradio interface, and the `xlm_r_lang_model` directory contains all necessary model artifacts.

To run locally:

1.  Clone this repository.
2.  Navigate to the `github_model_deployment` directory.
3.  Install dependencies: `pip install -r requirements.txt`
4.  Run the Gradio app: `python app.py`

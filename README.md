# Multilingual Language Detector

This repository contains a multilingual language detection pipeline based on XLM-RoBERTa.

The model is trained to classify text into Amharic, Afan Oromo, or English.

## Project Structure

```text
.
├── app.py
├── scripts/
│   └── train.py
├── src/
│   └── language_detection/
│       ├── __init__.py
│       ├── inference.py
│       └── training.py
└── models/
    └── xlm_r_lang_model/   # created after training
```

## Usage

To run locally:

1.  Clone this repository.
2.  Navigate to the project directory.
3.  Install dependencies: `pip install -r requirements.txt`
4.  Train model:
    - `python scripts/train.py --csv /absolute/path/to/data.csv --text-column text --label-column label`
5.  Run the Gradio app:
    - `python app.py`

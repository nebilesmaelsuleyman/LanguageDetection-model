# Multilingual Language Detector

This project detects between Amharic, Afan Oromo, and English texts using a fine-tuned `xlm-roberta-base` model.

## Project Structure

```
├── data/               # Place your training JSON files here
├── models/             # The trained model will be saved here
├── src/
│   ├── config.py       # Configuration and hyper-parameters
│   ├── data.py         # Data loading and preprocessing pipeline
│   ├── train.py        # Training script
│   ├── evaluate.py     # Evaluation functions (confusion matrix)
│   ├── predict.py      # Inference class for predictions
│   └── app.py          # Gradio Web UI
├── requirements.txt
└── README.md
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. To train the model:
First, add your training JSON files to the `./data` folder (or adjust the `DATA_PATH` in `src/config.py`).
```bash
python -m src.train
```

3. To predict via script:
```bash
python -m src.predict
```

4. To run the web interface:
```bash
python -m src.app
```

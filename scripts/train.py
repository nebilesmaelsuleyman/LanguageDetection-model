import argparse

from src.language_detection.config import DEFAULT_MODEL_DIR
from src.language_detection.training import train_model


def main():
    parser = argparse.ArgumentParser(description="Train language detection model.")
    parser.add_argument("--csv", required=True, help="Path to CSV data file.")
    parser.add_argument("--output-dir", default=DEFAULT_MODEL_DIR, help="Directory to save model.")
    parser.add_argument("--text-column", default="text", help="CSV text column name.")
    parser.add_argument("--label-column", default="label", help="CSV label column name.")
    args = parser.parse_args()

    train_model(
        csv_path=args.csv,
        output_dir=args.output_dir,
        text_column=args.text_column,
        label_column=args.label_column,
    )


if __name__ == "__main__":
    main()

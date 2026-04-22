import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, DatasetDict
from src.config import FULL_LABEL_MAP

def load_and_validate(path):
    all_dfs = []
    if not os.path.exists(path):
        print(f"Path {path} not found.")
        return pd.DataFrame({'text': [], 'label': []})

    if os.path.isdir(path):
        json_files = glob.glob(os.path.join(path, "*.json"))
        for file_path in json_files:
            temp_df = pd.read_json(file_path)
            if 'data' in temp_df.columns and (isinstance(temp_df['data'].iloc[0], list) or isinstance(temp_df['data'].iloc[0], dict)):
                 temp_df = pd.json_normalize(temp_df['data'])
            all_dfs.append(temp_df)
    else:
        df = pd.read_json(path)
        if 'data' in df.columns:
            df = pd.json_normalize(df['data'])
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    if 'text' in combined_df.columns and 'label' in combined_df.columns:
        combined_df = combined_df.dropna(subset=['text', 'label'])
        combined_df = combined_df.drop_duplicates()
        
    return combined_df

def prepare_dataset(raw_df):
    raw_df['text'] = raw_df['text'].astype(str).str.strip().str.lower()
    raw_df['label'] = raw_df['label'].map(FULL_LABEL_MAP)

    label_encoder = LabelEncoder()
    raw_df['label_idx'] = label_encoder.fit_transform(raw_df['label'])
    
    id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
    label2id = {label: i for i, label in enumerate(label_encoder.classes_)}

    train_df, val_df = train_test_split(
        raw_df, test_size=0.2, stratify=raw_df['label_idx'], random_state=42
    )

    ds = DatasetDict({
        'train': Dataset.from_pandas(train_df.reset_index(drop=True)),
        'validation': Dataset.from_pandas(val_df.reset_index(drop=True))
    })
    
    return ds, id2label, label2id, label_encoder

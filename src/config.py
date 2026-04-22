import torch
import os
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42
DATA_PATH = os.environ.get('DATA_PATH', './data/clean_traingReady_datasets')
MODEL_CHECKPOINT = "xlm-roberta-base"
SAVE_PATH = os.environ.get('SAVE_PATH', './models/xlm_r_lang_model')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FULL_LABEL_MAP = {'am': 'Amharic', 'om': 'Afan Oromo', 'en': 'English'}

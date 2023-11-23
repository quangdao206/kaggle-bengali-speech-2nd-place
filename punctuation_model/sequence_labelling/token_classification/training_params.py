import torch
from transformers import AutoTokenizer

MAX_LEN = 256
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH="ai4bharat/IndicBERTv2-MLM-Sam-TLM"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
FULL_FINETUNING = True
EPOCHS = 6
MAX_GRAD_NORM = 1.0
CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'runs'
LOAD_CHECKPOINT = True
LEARNING_RATE = 3e-5

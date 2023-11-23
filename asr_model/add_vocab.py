import re
import os 
from utils import *
import pandas as pd
import json
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2ForCTC

if __name__ == "__main__":

    data_path = "/path_to_training_data"

    # download and save pretrained
    processor = Wav2Vec2Processor.from_pretrained(
        "ai4bharat/indicwav2vec_v1_bengali"
    )
    model = Wav2Vec2ForCTC.from_pretrained("ai4bharat/indicwav2vec_v1_bengali")

    processor.save_pretrained("pretrained/ai4bharat/indicwav2vec_v1_bengali/")
    model.save_pretrained("pretrained/ai4bharat/indicwav2vec_v1_bengali/")

    train_df = pd.read_csv(os.path.join(data_path, "train_comp_processed.csv"))
    train_shru = pd.read_csv(os.path.join(data_path, "train_shru_processed.csv"))
    train_respin = pd.read_csv(os.path.join(data_path, "train_respin_processed.csv"))
    train_ulca = pd.read_csv(os.path.join(data_path, "train_ulca_processed.csv"))

    all_data = pd.concat([train_df, train_shru, train_respin, train_ulca])
    all_data = all_data.dropna(subset=['sentence'])

    texts = all_data["sentence"].tolist()
    vocab_list = []
    for text in texts:
        vocab_list.extend(list(text))
    vocab_list = list(set(vocab_list))

    old_vocab = json.load(
        open("pretrained/ai4bharat/indicwav2vec_v1_bengali/vocab.json", "rb")
    )
    new_vocab = list(set(vocab_list) - set(old_vocab.keys()) - set([" "]))

    len_old_vocab = len(old_vocab)
    for k in range(0, len(new_vocab)):
        old_vocab[new_vocab[k]] = k + len_old_vocab

    print(len_old_vocab, len(old_vocab))
    print(new_vocab)

    vocab_dict = json.dumps(old_vocab, ensure_ascii=False)
    with open(
        "pretrained/ai4bharat/indicwav2vec_v1_bengali/vocab.json", "w"
    ) as fp:
        fp.write(vocab_dict)
import typing as tp
from pathlib import Path
from functools import partial
from dataclasses import dataclass, field

import pandas as pd
import pyctcdecode
import numpy as np
from tqdm.auto import tqdm
import argparse

import librosa

import jiwer
import pyctcdecode
import kenlm
import os
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC
from bnunicodenormalizer import Normalizer

SAMPLING_RATE = 16_000

class BengaliSRTestDataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        audio_paths: list[str],
        sampling_rate: int
    ):
        self.audio_paths = audio_paths
        self.sampling_rate = sampling_rate
        
    def __len__(self,):
        return len(self.audio_paths)
    
    def __getitem__(self, index: int):
        audio_path = self.audio_paths[index]
        sr = self.sampling_rate
        w = librosa.load(audio_path, sr=sr, mono=False)[0]
        
        return w

def main():
    parser = argparse.ArgumentParser(
        description="ASR model validation."
    )
    parser.add_argument(
        "--data_path",
        help="Path to folder with csv transcription file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_path",
        help="Path to inference model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--filter_csv_path",
        help="Path to csv file for data filtering",
        default="",
        type=str,
    )

    model = Wav2Vec2ForCTC.from_pretrained(args.model_path)
    processor = Wav2Vec2Processor.from_pretrained(args.model_path)

    selected_cols = ["path","sentence","length"]

    train_comp = pd.read_csv(os.path.join(args.data_path, f"train_comp_processed.csv"))
    train_comp = train_comp[selected_cols]
    train_comp["source"] = "read"

    ext_file = f"train_shru_processed.csv"
    print(f"Loading {ext_file}...")
    train_shru = pd.read_csv(os.path.join(args.data_path, ext_file))
    train_shru = train_shru[selected_cols]
    train_shru["source"] = "spontaneous"

    ext_file = f"train_respin_processed.csv"
    print(f"Loading {ext_file}...")
    train_respin = pd.read_csv(os.path.join(args.data_path, ext_file))
    train_respin = train_respin[selected_cols]
    train_respin["source"] = "read"

    ext_file = f"train_ulca_processed.csv"
    print(f"Loading {ext_file}...")
    train_ulca = pd.read_csv(os.path.join(args.data_path, ext_file))
    train_ulca = train_ulca[selected_cols]
    train_ulca["source"] = "spontaneous"

    train_df = pd.concat([train_comp, train_respin, train_ulca, train_shru])

    train_df = train_df[train_df.length> (CFG.min_input_length_in_sec * 16000)]
    train_df = train_df[train_df.length< (CFG.max_input_length_in_sec * 16000)]

    train_df = train_df.dropna(subset=['sentence'])
    train_df = train_df.query('sentence.str.len() > 5')
    train_df = train_df.reset_index()

    valid = train_df[["path","sentence", "length", "source"]]

    valid_audio_paths = valid["path"].tolist()

    valid_dataset = BengaliSRTestDataset(
        valid_audio_paths, SAMPLING_RATE
    )

    collate_func = partial(
        processor.feature_extractor,
        return_tensors="pt", sampling_rate=SAMPLING_RATE,
        padding=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=32, shuffle=False,
        num_workers=16, collate_fn=collate_func, drop_last=False,
        pin_memory=True,
    )

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    model = model.to(device)
    model = model.eval()
    model = model.half()

    pred_sentence_list = []

    with torch.no_grad():
        for batch in tqdm(valid_loader):
            x = batch["input_values"]
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(True):
                y = model(x).logits
            y = torch.argmax(y, dim=-1)
            y = y.detach().cpu().numpy()

            for l in y:  
                sentence = processor.decode(l)
                pred_sentence_list.append(sentence)

    valid["pred_sentence"] = pred_sentence_list
    valid["wer"] = [
        jiwer.wer(s, p_s)
        for s, p_s in tqdm(valid[["sentence", "pred_sentence"]].values)
    ]

    valid.to_csv(args.filter_csv_path, index=False)
    print(valid["wer"].mean())

if __name__ == "__main__":
    main()
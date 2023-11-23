import pandas as pd
import numpy as np
import json
import os
import ast
import librosa
import glob
import csv
import pickle
from datasets import Audio
from datasets import Dataset
from bnunicodenormalizer import Normalizer 
bnorm=Normalizer()
from datasets import concatenate_datasets
from tqdm.auto import tqdm

chars_to_ignore_regex = '[\।\,\?\!\;\:\"\—\‘\'\‚\“\”\…]'
data_path = "/path_to_training_data"
audio_dir = data_path + "/train_mp3s/"
num_workers = 16

def get_audio_length(audio_path):
    w = librosa.load(audio_path, sr=16_000, mono=False)[0]
    if len(w.shape)==2:
        print(w.shape)
    return w.shape[0]

def remove_special_characters(text):
    text = re.sub(chars_to_ignore_regex, ' ', text)
    text = ' '.join(text.split())
    return text

def normalize(text):
    _words = [bnorm(word)['normalized']  for word in text.split()]
    text =  " ".join([word for word in _words if word is not None])
    return text

# Prefilter data according to hosts' recommendation
df_qa = pd.read_csv(os.path.join(data_path, "NISQA_wavfiles.csv"),sep=",")
df_qa.rename(columns={'deg':'id'},inplace=True) ## rename to match other dfs
df_qa['id'] = df_qa['id'].apply(lambda x:x.split('.')[0])  ## remove .wav
df_qa = df_qa[df_qa.mos_pred>1.5]
df = pd.read_csv(os.path.join(data_path, "train_metadata_corrected.csv"),sep=",")
df["path"] = df['id'].apply(lambda x:audio_dir+x+".mp3")
df.set_index('id',inplace=True)
df = df.join(df_qa.set_index('id'), how='inner')
df = df.dropna(subset='yellowking_preds')[df.ykg_wer < 3]

df["sentence"] = [ normalize(x) for x in tqdm(df["sentence"]) ]
df["sentence"] = [ remove_special_characters(x) for x in tqdm(df["sentence_p"]) ]
df["length"] = [ get_audio_length(x) for x in tqdm(df["path"]) ]

df = df[["path", "sentence", "length"]]
df_meta.to_csv(os.path.join(data_path, "train_comp_processed.csv"), index=False)

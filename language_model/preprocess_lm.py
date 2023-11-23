import re
import random
import os
import pandas as pd
from utils import *
from tqdm.auto import tqdm
from joblib import delayed, Parallel
from bnunicodenormalizer import Normalizer 

chars_to_ignore_regex_v2 = '[\।\,\?\!\-\;\:\"\—\‘\'\‚\“\”\…]'
chars_to_ignore_regex_v2_nosep = '[\.]'
chars_to_ignore_regex_v4 = '[\(\)\{\}\[\]\>\<\=]'

def remove_special_characters_v4(text):
    text = re.sub(chars_to_ignore_regex_v2_nosep, '', text)
    text = re.sub(chars_to_ignore_regex_v2, ' ', text)
    text = re.sub(chars_to_ignore_regex_v4, ' ', text)
    text = ' '.join(text.split())
    return text

def normalize(text):
    _words = [bnorm(word)['normalized']  for word in text.split()]
    text =  " ".join([word for word in _words if word is not None])
    return text

with open("/path_to_unprocessed_corpus.txt", "r") as read_file, open("/path_to_processed_corpus.txt", "w") as write_file:
    lines = Parallel(n_jobs=16)(delayed(normalize)(x) for x in read_file)
    lines = Parallel(n_jobs=16)(delayed(remove_special_characters_v4)(x) for x in read_file)
    for line in tqdm(lines):
        write_file.write(line + os.linesep)


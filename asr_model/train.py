import torch
import os
import numpy as np
import pandas as pd
import random
import wandb
from tqdm.auto import tqdm
import librosa
import warnings
import argparse
warnings.filterwarnings("ignore")

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import DataLoader, IterableDataset
from functools import partial

from torch.utils.data import Dataset

import transformers
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, AutoProcessor, AutoConfig, AutoFeatureExtractor, AutoModelForSpeechSeq2Seq, AutoTokenizer
from transformers import TrainingArguments, Trainer, Seq2SeqTrainer, Seq2SeqTrainingArguments, HfArgumentParser, set_seed
from datasets import load_dataset, load_metric, Audio
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Union, Optional
from models import Wav2Vec2ForCTCV2
from audiomentations import *

os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_PROJECT"] = "b_speech"

# Training config class.
class CFG:
    dns_noise_path = "/path_to_DNS_challenge_noise"
    musan_path = "/path_to_MUSAN"

    pretrained_path = "pretrained/ai4bharat/indicwav2vec_v1_bengali"
    save_dir_stage_1 = "ckpt_stage1"
    save_dir_stage_2 = "ckpt_stage2"
    save_dir_stage_3 = "ckpt_stage3"

    sample_rate = 16000
    epochs = 5
    lr = 4e-5

    # Dropout configs for pretrained wav2vec2 model.
    attention_dropout = 0.1
    hidden_dropout = 0.1
    feat_proj_dropout = 0.1
    mask_time_prob = 0.1
    layerdrop = 0.1
    mask_feature_prob = 0.05	

    max_input_length_in_sec = 13.0
    min_input_length_in_sec = 2.0

    # Trainer arugments.
    trainer = TrainingArguments(
      output_dir="weights",
      group_by_length=False,
      length_column_name="input_length",
      per_device_train_batch_size=4,
      per_device_eval_batch_size=4,
      gradient_accumulation_steps=1,
      num_train_epochs=epochs,
      gradient_checkpointing=False,
      fp16=True,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_steps=100, # number of bathes after which to log metrics from the model
    report_to="wandb",
      learning_rate=lr,
    weight_decay=0.0025,
      dataloader_num_workers=16,
    warmup_ratio=0.1,
      save_total_limit=5,
      push_to_hub=False,
      load_best_model_at_end=True,
    greater_is_better=False,
    metric_for_best_model='eval_wer',
      lr_scheduler_type="cosine",
      remove_unused_columns=False,
    )

augment_read_speech = Compose([
    TimeStretch(min_rate=0.8, max_rate=2.0, p=0.5, leave_length_unchanged=False),
    RoomSimulator(
        p=0.3),
    OneOf([
        AddBackgroundNoise(
            sounds_path=[
                CFG.dns_noise_path,
            ],
            min_snr_in_db=5.0,
            max_snr_in_db=30.0,
            noise_transform=PolarityInversion(),
            p=1.0
        ),
        AddBackgroundNoise(
            sounds_path=[
                CFG.musan_path,
            ],
            min_snr_in_db=5.0,
            max_snr_in_db=30.0,
            noise_transform=PolarityInversion(),
            p=1.0
        ),
        AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=1.0),
    ], p=0.7),
    Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.2),
    ])

augment_spontaneous_speech = Compose([
    TimeStretch(min_rate=0.8, max_rate=1.1, p=0.3, leave_length_unchanged=False),
    RoomSimulator(
        p=0.3),
    OneOf([
        AddBackgroundNoise(
            sounds_path=[
                CFG.dns_noise_path,
            ],
            min_snr_in_db=5.0,
            max_snr_in_db=30.0,
            noise_transform=PolarityInversion(),
            p=1.0
        ),
        AddBackgroundNoise(
            sounds_path=[
                CFG.musan_path,
            ],
            min_snr_in_db=5.0,
            max_snr_in_db=30.0,
            noise_transform=PolarityInversion(),
            p=1.0
        ),
        AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=1.0),
    ], p=0.3),
    Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.2),
    ])

class BengaliDataset(Dataset):
        
    def __init__(self, config, df, processor, split):
        self.df = df
        self.cfg = config
        self.arch = config.arch
        self.paths = df['path']
        self.sentences = df['sentence']
        self.sources = df['source']
        self.lengths = df['length'].to_numpy()
        self.len = len(self.df) 
        self.sr = 16_000

        self.processor = processor
        self.split = split

    def __len__(self):
        return self.len

    def load_audio(self, idx):
        idx %= len(self.df)
        audio_path = self.paths[idx]
        sentence = self.sentences[idx]
        source = self.sources[idx]
        num_frames = self.lengths[idx]
        concat_augment=False

        wav = librosa.load(audio_path, sr=self.sr, mono=False)[0]
        wav = np.trim_zeros(wav, 'fb')

        if self.split=="train":
            if (num_frames<(self.cfg.max_input_length_in_sec*8000)) and (np.random.uniform() < 0.5):
                num_frames_concat = self.cfg.max_input_length_in_sec*self.sr-num_frames
                possible_indexes = np.where(self.lengths < num_frames_concat)[0]
                if len(possible_indexes)>0:
                    concat_augment=True
                    selected_index = np.random.choice(possible_indexes)
                    audio_path_concat = self.paths[selected_index]
                    sentence_concat = self.sentences[selected_index]
                    wav_concat = librosa.load(audio_path_concat, sr=self.sr, mono=False)[0]
                    wav_concat = np.trim_zeros(wav_concat, 'fb')
                    wav = np.concatenate((wav, wav_concat))
                    sentence = sentence + " " + sentence_concat

            try:
                if source=="spontaneous":
                    wav = augment_spontaneous_speech(samples=wav, sample_rate=self.sr)
                else:
                    wav = augment_read_speech(samples=wav, sample_rate=self.sr)
            except:
                print(audio_path)

        wav = np.expand_dims(wav, axis=0)

        input_values = self.processor(wav, sampling_rate=self.sr).input_values[0]

        input_length = len(input_values)
        with self.processor.as_target_processor():
            labels = self.processor(sentence).input_ids

        return {
            'input_values':input_values,
            'input_length':input_length,
            'labels':labels
        }

    def __getitem__(self, idx): 
        if idx >= self.len:
            raise IndexError(f'index {idx} out of range {self.len}')
        return self.load_audio(idx)

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def main():
    # seed
    seed = 20
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser(
        description="ASR model training."
    )
    parser.add_argument(
        "--data_path",
        help="Path to folder with csv transcription file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--stage",
        help="Training stage",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--filter_csv_path",
        help="Path to csv file for data filtering",
        default="",
        type=str,
    )

    args = parser.parse_args()

    if args.stage==2:
        CFG.epochs = 3
        CFG.lr = 3e-5
        CFG.pretrained_path = CFG.save_dir_stage_1
    elif args.stage==3:
        CFG.epochs = 3
        CFG.lr = 2e-5
        CFG.pretrained_path = CFG.save_dir_stage_2

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

    ext_file = f"valid_kathbath_processed.csv"
    print(f"Loading {ext_file}...")
    valid_kathbath = pd.read_csv(os.path.join(args.data_path, ext_file))
    valid_kathbath = valid_kathbath[selected_cols]
    valid_kathbath["source"] = "read"

    train_df = pd.concat([train_comp, train_respin, train_ulca, train_shru])
    valid_df = valid_kathbath

    train_df = train_df[train_df.length> (CFG.min_input_length_in_sec * 16000)]
    train_df = train_df[train_df.length< (CFG.max_input_length_in_sec * 16000)]

    train_df = train_df.dropna(subset=['sentence'])
    train_df = train_df.query('sentence.str.len() > 5')
    valid_df = valid_df.dropna(subset=['sentence'])
    valid_df = valid_df.query('sentence.str.len() > 5')

    if args.filter_csv_path != "":
        noise_df = pd.read_csv(args.filter_csv_path)
        # Filter 10% noisiest data
        noise_df = noise_df.sort_values('wer', ascending=False)[:int(noise_df.shape[0]*0.1)]
        noise_df = noise_df["path"].tolist()
        train_df = train_df[~train_df.path.isin(noise_df)]

    train_df = train_df.reset_index()
    valid_df = valid_df.reset_index()
    print("Split length: ", len(train_df), len(valid_df))

    train_df = train_df[["path","sentence", "length", "source"]]
    valid_df = valid_df[["path","sentence", "length", "source"]]

    processor = Wav2Vec2Processor.from_pretrained(CFG.pretrained_path)

    train_dataset = BengaliDataset(CFG, train_df, processor, "train")
    valid_dataset = BengaliDataset(CFG, valid_df, processor, "valid")

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    wer_metric = load_metric("wer")

    # Loading model.
    print("Loading model...")
    model = Wav2Vec2ForCTCV2.from_pretrained(
        CFG.pretrained_path, 
        ignore_mismatched_sizes=True,
        attention_dropout=CFG.attention_dropout,
        hidden_dropout=CFG.hidden_dropout,
        feat_proj_dropout=CFG.feat_proj_dropout,
        mask_time_prob=CFG.mask_time_prob,
        mask_feature_prob=CFG.mask_feature_prob,
        layerdrop=CFG.layerdrop,
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    model.config.ctc_zero_infinity = True 

    new_vocab_size = len(processor.tokenizer.get_vocab())
    print("New vocab size: ", new_vocab_size)
    model.resize_lm_head(new_num_tokens=len(processor.tokenizer.get_vocab()))
    model.config.vocab_size = new_vocab_size

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=CFG.trainer,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,   
        tokenizer=processor.feature_extractor,
    )

    print("Start training...")
    trainer.train(
        # "weights/checkpoint-27036"
    )

    if args.stage==1:
        final_dir = CFG.save_dir_stage_1
    elif args.stage==2:
        final_dir = CFG.save_dir_stage_2
    else:
        final_dir = CFG.save_dir_stage_3

    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)

if __name__ == "__main__":
    main()


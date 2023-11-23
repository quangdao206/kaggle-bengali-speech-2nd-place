import os
import pandas as pd
import warnings
import json
import yaml
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, AutoModelForTokenClassification, AutoConfig, AutoModel, BertPreTrainedModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import TokenClassifierOutput
from typing import List, Optional, Tuple, Union

import training_params
from dataset_loader import PunctuationDataset, PunctuationDatasetForTrainer
from utils import process_data, folder_with_time_stamps
import wandb
import evaluate
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer, Seq2SeqTrainer, Seq2SeqTrainingArguments, HfArgumentParser, set_seed
warnings.filterwarnings('ignore')

os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_PROJECT"] = "pct_2"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

# seed
seed = 400
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

log_folder, checkpoint_folder, train_encoder_file_path, _ = folder_with_time_stamps(training_params.LOG_DIR,
                                                                                 training_params.CHECKPOINT_DIR)

print(f"train encoder path -> {train_encoder_file_path}")

os.makedirs(log_folder, exist_ok=True)
os.makedirs(checkpoint_folder, exist_ok=True)

writer = SummaryWriter(log_folder)

config = {
  "learning_rate": training_params.LEARNING_RATE,
  "batch_size": training_params.BATCH_SIZE,
  "num_epochs" : training_params.EPOCHS,
  "max_len" : training_params.MAX_LEN,
  "load_checkpoint" : training_params.LOAD_CHECKPOINT,
  "max_grad_norm" : training_params.MAX_GRAD_NORM
}

names = yaml.safe_load(open('../config.yaml'))

PROJECT_NAME = names['PROJECT_NAME']
exp = f"new_v3plus_s{seed}_ner"
print(f"Training experiment {exp}...")

train_dir = '/path_to_training_data'

train_sentences, train_labels, train_encoder, tag_values = process_data(os.path.join(train_dir, f'train_punctuation_s{seed}.csv'))
valid_sentences, valid_labels, _, _ = process_data(os.path.join(train_dir, f'valid_punctuation_s{seed}.csv'))

print("--------------------------------Tag Values----------------------------------")
print(tag_values)

id2label = {i: label for i, label in enumerate(tag_values)}
label2id = {v: k for k, v in id2label.items()}


class ResidualLSTM(nn.Module):

    def __init__(self, d_model,rnn):
        super(ResidualLSTM, self).__init__()
        self.downsample=nn.Linear(d_model,d_model//2)
        if rnn=='GRU':
            self.LSTM=nn.GRU(d_model//2, d_model//2, num_layers=2, bidirectional=False, dropout=0.2)
        else:
            self.LSTM=nn.LSTM(d_model//2, d_model//2, num_layers=2, bidirectional=False, dropout=0.2)
        self.dropout1=nn.Dropout(0.2)
        self.norm1= nn.LayerNorm(d_model//2)
        self.linear1=nn.Linear(d_model//2, d_model*4)
        self.linear2=nn.Linear(d_model*4, d_model)
        self.dropout2=nn.Dropout(0.2)
        self.norm2= nn.LayerNorm(d_model)

    def forward(self, x):
        res=x
        x=self.downsample(x)
        x, _ = self.LSTM(x)
        x=self.dropout1(x)
        x=self.norm1(x)
        x=F.relu(self.linear1(x))
        x=self.linear2(x)
        x=self.dropout2(x)
        x=res+x
        return self.norm2(x)

class CustomModel(nn.Module):
    def __init__(self, config_path, num_labels):
        super(CustomModel, self).__init__()
        self.config = AutoConfig.from_pretrained(config_path)
        self.num_labels = num_labels

        self.model = AutoModel.from_pretrained(
            config_path, add_pooling_layer=False
        )

        self.lstm = ResidualLSTM(self.config.hidden_size,'LSTM')
        self.classification_head = nn.Linear(self.config.hidden_size,self.num_labels)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        x = self.lstm(sequence_output.permute(1,0,2)).permute(1,0,2)
        logits = self.classification_head(x)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


model = CustomModel(training_params.MODEL_PATH, len(tag_values))

final_dir = f"train_demo_{exp}"

train_dataset = PunctuationDatasetForTrainer(texts=train_sentences, labels=train_labels,
                                   tag2idx=label2id, split="train")
valid_dataset = PunctuationDatasetForTrainer(texts=valid_sentences, labels=valid_labels,
                                   tag2idx=label2id, split="valid")

data_collator = DataCollatorForTokenClassification(tokenizer=training_params.TOKENIZER)
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [tag_values[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [tag_values[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    print(results)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

trainer_args = TrainingArguments(
     output_dir=f"weights_{exp}",
     length_column_name="input_length",
     per_device_train_batch_size=128,
     per_device_eval_batch_size=128,
     gradient_accumulation_steps=1,
   evaluation_strategy="epoch",
     num_train_epochs=6,
     gradient_checkpointing=False,
     fp16=True,
   save_strategy="epoch",
   logging_steps=100, # number of bathes after which to log metrics from the model
   report_to="wandb",
     learning_rate=3e-5,
   weight_decay=0.0025,
     dataloader_num_workers=16,
   warmup_ratio=0.1,
     save_total_limit=3,
     push_to_hub=False,
     load_best_model_at_end=True,
   greater_is_better=True,
   metric_for_best_model='eval_f1',
     lr_scheduler_type="cosine",
     remove_unused_columns=False,
   )

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        labels = torch.nn.functional.one_hot(labels, num_classes=5)
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = SmoothFocalLoss(reduction='mean')
        loss = loss_fct(logits.view(-1, 5), labels.view(-1, 5))
        return (loss, outputs) if return_outputs else loss

trainer = Trainer(
       model=model,
       data_collator=data_collator,
       args=trainer_args,
       compute_metrics=compute_metrics,
       train_dataset=train_dataset,
       eval_dataset=valid_dataset,
       tokenizer=training_params.TOKENIZER,
   )

print("Start training...")

trainer.train(
#    "weights/checkpoint-6429"
)

trainer.save_model(final_dir)
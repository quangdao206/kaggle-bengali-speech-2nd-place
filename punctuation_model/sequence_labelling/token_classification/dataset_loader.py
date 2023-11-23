import pandas as pd
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from training_params import TOKENIZER
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

class PunctuationDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tag2idx):
        self.texts = texts
        self.labels = labels
        self.tag2idx = tag2idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        sentence = self.texts[item].split()
        text_label = self.labels[item].split()

        tokenized_sentence = []
        labels = []

        for word, label in zip(sentence, text_label):
            # Tokenize the word and count number of subwords
            tokenized_word = TOKENIZER.tokenize(word)
            n_subwords = len(tokenized_word)

            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)

            # Add the same label to the new list of labels `n_subwords` times
            labels.extend([label] * n_subwords)

        input_ids = pad_sequences([TOKENIZER.convert_tokens_to_ids(tokenized_sentence)],
                                  maxlen=MAX_LEN, dtype="long", value=0.0,
                                  truncating="post", padding="post")

        tags = pad_sequences([[self.tag2idx.get(l) for l in labels]],
                             maxlen=MAX_LEN, value=self.tag2idx["PAD"], padding="post",
                             dtype="long", truncating="post")

        attention_masks = [float(i != 0.0) for i in input_ids[0]]

        return {
            "ids": torch.tensor(input_ids[0], dtype=torch.long),
            "mask": torch.tensor(attention_masks, dtype=torch.long),
            "target_tag": torch.tensor(tags[0], dtype=torch.long),
        }

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def ner_format_label(label):
    return label.replace('blank', 'O').replace('end', 'B-END').replace('comma', 'B-COMMA').replace('qm', 'B-QM').replace('exclm', 'B-EXCLM')

class PunctuationDatasetForTrainer(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tag2idx, split="train"):
        self.texts = texts
        self.labels = labels
        self.tag2idx = tag2idx
        self.mask_token = TOKENIZER.convert_tokens_to_ids(TOKENIZER.mask_token)
        self.mask_aug_prob = 0.15
        self.split = split

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        sentence = self.texts[item].split()
        text_label = ner_format_label(self.labels[item]).split()
        all_labels = [[self.tag2idx.get(l) for l in text_label]]

        tokenized_inputs = TOKENIZER(
            sentence, truncation=True, max_length=512, is_split_into_words=True
        )
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels[0]

        tokenized_inputs = {key: torch.as_tensor(val) for key, val in tokenized_inputs.items()}
        if self.split=="train":
            ix = torch.rand(size=(len(tokenized_inputs['input_ids']),)) < self.mask_aug_prob
            tokenized_inputs['input_ids'][ix] = self.mask_token
        return tokenized_inputs

if __name__=="__main__":
    df = pd.read_csv('data/train_sample.csv')
    print("File Read")
    tag_values = ['blank', 'end', 'comma', 'qm']
    tag_values.append("PAD")
    encoder = {t: i for i, t in enumerate(tag_values)}
    print(encoder)
    '''
    def split_string(line):
        return str(line).split()
    sentences = Parallel(n_jobs=-1)(delayed(split_string)(s) for s in tqdm(df['sentence']))
    sentences = np.asarray(sentences)
    punctuations = Parallel(n_jobs=-1)(delayed(split_string)(s) for s in tqdm(df['label']))
    punctuations = np.asarray(punctuations)
    '''
    sentences = df['sentence'].values
    punctuations = df['label'].values
    d = PunctuationDataset(sentences, punctuations, encoder).__getitem__(0)
    print(type(d))
    print(d['ids'])
    print(d['mask'])
    print(d['target_tag'])
# Bengali.AI Speech Recognition 2nd Place Solution

[Competition](https://www.kaggle.com/competitions/bengaliai-speech/overview)

[Solution Write-up](https://www.kaggle.com/competitions/bengaliai-speech/discussion/447976#2486531)

This documentation outlines how to reproduce the 2nd place solution for Kaggle Bengali.AI Speech Recognition

## ASR Model 

### Data Preparation

Speech Datasets:

- Competition data
- [Shrutilipi](https://ai4bharat.iitm.ac.in/shrutilipi/)
- [MADASR](https://sites.google.com/view/respinasrchallenge2023/dataset?authuser=0)
- [ULCA](https://github.com/Open-Speech-EkStep/ULCA-asr-dataset-corpus)
- [Kathbath Hard](https://github.com/AI4Bharat/vistaar) (for validation)

Noise Datasets:
- [DNS Challenge 2020](https://github.com/microsoft/DNS-Challenge/tree/interspeech2020/master)
- [MUSAN](https://www.openslr.org/17/)

Speech transcriptions should be preprocessed similar to example in `asr_model/preprocess.py`. Each dataset's labels should be stored in a `csv` file in the format `"path", "sentence", "length"`.

### Model Training

First run the following script to generate custom vocab for pretrained model:

```
    cd asr_model/
    python add_vocab.py

```

Then run first training pass on all dataset:
```
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --data_path "/path_to_dataset" --stage 1

```

Perform inference on train set to generate WER score:
```
    python valid.py  --data_path "/path_to_dataset" --filter_csv_path "/path/to/infer_score.csv"  --model_path "ckpt_stage1"

```

Second pass of the training will be run in 3 stages on filtered dataset:
```
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --data_path "/path_to_dataset" --filter_csv_path "/path/to/infer_score.csv" --stage 1
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --data_path "/path_to_dataset" --filter_csv_path "/path/to/infer_score.csv" --stage 2
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --data_path "/path_to_dataset" --filter_csv_path "/path/to/infer_score.csv" --stage 3
```

## Language Model 

Code for language model training is modified from [here](https://github.com/Open-Speech-EkStep/vakyansh-wav2vec2-experimentation/blob/main/scripts/lm/run_lm_pipeline.sh).

### Data Preparation

Create corpus from these datasets:
- [IndicCorp V1+V2](https://github.com/AI4Bharat/IndicBERT/tree/main#indiccorp-v2).
- [Bharat Parallel Corpus Collection](https://ai4bharat.iitm.ac.in/bpcc/).
- [Samanantar](https://ai4bharat.iitm.ac.in/samanantar/).
- [Bengali poetry dataset](https://www.kaggle.com/datasets/truthr/free-bengali-poetry).
- [WMT News Crawl](https://data.statmt.org/news-crawl/).
- Hate speech corpus from https://github.com/rezacsedu/Classification_Benchmarks_Benglai_NLP.

Concatenate the texts into a singular file, deduplicate lines and preprocess using `language_model/preprocess_lm.py`.

### Model Training

Modify the path of the processed corpus then run the following script:
```
    cd language_model/
    sh run_lm_pipeline.sh

```

## Punctuation Model 
Code for punctuation model training is modified from [here](https://github.com/Open-Speech-EkStep/punctuation-ITN/tree/wandb-v1/sequence_labelling).

### Data Preparation

Create corpus from these datasets:
- [IndicCorp V1+V2](https://github.com/AI4Bharat/IndicBERT/tree/main#indiccorp-v2).
- Competition data.

Concatenate the texts into a singular file, deduplicate lines and preprocess using `punctuation_model/sequence_labelling/make_data.py`.

### Model Training

Modify the path of the processed corpus then run the following script:
```
    cd punctuation_model/sequence_labelling/token_classification/
    python train.py

```
# Training punctuation model


In this we finetune a [IndicBERT](https://indicnlp.ai4bharat.org/indic-bert/) model (pretrained on 12 indic languages corpora) 

Code is linked with Wandb to monitor our training in real-time. And all input data, intermediate reults and resulting checkpoint are picked and stored in Google Cloud Platform (GCP) bucket

## 1. Prepare data for training

First upload raw data text file and dictionary file in gcp bucket, make changes in [config.yaml](https://github.com/Open-Speech-EkStep/punctuation-ITN/blob/wandb-v1/sequence_labelling/config.yaml) based on your data. and run [make_data.py](https://github.com/Open-Speech-EkStep/punctuation-ITN/blob/wandb-v1/sequence_labelling/make_data.py) to generate train, test and valid csvs.

In [config.yaml](https://github.com/Open-Speech-EkStep/punctuation-ITN/blob/wandb-v1/sequence_labelling/config.yaml) edit PROJECT_NAME for name of wandb project, LANG language code, RAW_FILE_NAME and DICT_NAME for raw text file and dict file name uploaded in bucket, after [process_raw_text.py](https://github.com/Open-Speech-EkStep/punctuation-ITN/blob/wandb-v1/sequence_labelling/prep_scripts/process_raw_text.py) length of corpus will be reduced SAMPLE_LEN will filter out these many sentences for cleaned text and TEST_AND_VALID_LEN will separated out these many sentences for valid and test set

Cleaning steps
1. normalize text corpus
2. tokenize text corpus
3. replace foreign characters not punctuation and numerals with space



For punctuation symbol we have taken only [".", ",", "?"] these 3 symbols, which can be changed in [process_raw_text.py](https://github.com/Open-Speech-EkStep/punctuation-ITN/blob/wandb-v1/sequence_labelling/prep_scripts/process_raw_text.py) and [prepare_csv.py](https://github.com/Open-Speech-EkStep/punctuation-ITN/blob/wandb-v1/sequence_labelling/prep_scripts/prepare_csv.py) 

## 2. Start Training 

format of input csvs file for training

> sentence_index,sentence,label

where label maps what is the next punctuation symbol for the corresponding word in sentence.



To start training change training parameters from [training_params.py](https://github.com/Open-Speech-EkStep/punctuation-ITN/blob/wandb-v1/sequence_labelling/token_classification/training_params.py) and run  [train.py](https://github.com/Open-Speech-EkStep/punctuation-ITN/blob/wandb-v1/sequence_labelling/token_classification/train.py). In train.py edit variable 'checkpoint_bucket_path' write bucket path where to store checkpoints.



## 3. Inference

To infer sentences check this file [inference.py](https://github.com/Open-Speech-EkStep/punctuation-ITN/blob/wandb-v1/sequence_labelling/token_classification/inference.py)

Also there is a spearate repository to try already built punctation model in indic langauges [indic-punct](https://github.com/Open-Speech-EkStep/indic-punct#punctuation)

## Citation 
```
@misc{https://doi.org/10.48550/arxiv.2203.16825,
  doi = {10.48550/ARXIV.2203.16825},
  
  url = {https://arxiv.org/abs/2203.16825},
  
  author = {Gupta, Anirudh and Chhimwal, Neeraj and Dhuriya, Ankur and Gaur, Rishabh and Shah, Priyanshi and Chadha, Harveen Singh and Raghavan, Vivek},
  
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {indic-punct: An automatic punctuation restoration and inverse text normalization framework for Indic languages},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
```

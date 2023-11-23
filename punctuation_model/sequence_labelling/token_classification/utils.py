import pandas as pd
import datetime
import training_params
from joblib import Parallel, delayed
from tqdm import tqdm

def process_data(data_csv):
    df = pd.read_csv(data_csv)
    df.dropna(inplace = True)
    tag_values = ['O', 
            'B-END', 
            'I-END', 
            'B-COMMA', 
            'I-COMMA', 
            'B-QM', 
            'I-QM', 
            'B-EXCLM',
            'I-EXCLM',
            ]

    encoder = {t: i for i, t in enumerate(tag_values)}
    print(f"Encoder: {encoder}")
    sentences = df['sentence'].values
    labels = df['label'].values
    return sentences, labels, encoder, tag_values


def folder_with_time_stamps(log_folder, checkpoint_folder):
    folder_hook = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_saving = log_folder + '/' + folder_hook
    checkpoint_saving = checkpoint_folder + '/' + folder_hook
    train_encoder_file_path = 'label_encoder_' + folder_hook + '.json'
    return log_saving, checkpoint_saving, train_encoder_file_path, folder_hook

if __name__=="__main__":
    pass

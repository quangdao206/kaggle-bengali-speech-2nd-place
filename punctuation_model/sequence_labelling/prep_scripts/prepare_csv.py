from joblib.parallel import delayed
import wandb
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import os

class PrepareCsv:
    def __init__(self, config_path):
        self.config_path = config_path
        self.names = yaml.safe_load(open(self.config_path))
    
    def fix_for_first_ch_punc(self, line):
        if line and (line[0] in ['।', ',', '?','!']):
            line = line[1:]
            return ' '.join(line.split())
        return line
    
    def split_sen_with_label(self, line):
        line = self.fix_for_first_ch_punc(line)
        words, labels = [], []
        word_list = line.split()
        for w in word_list:
            if w in [',', '।', '?', '!']:
                if w == ',':
                    lab = 'comma'
                elif w == '।':
                    lab = 'end'
                elif w == '?':
                    lab = 'qm'
                elif w == '!':
                    lab = 'exclm'
                labels.pop()
                labels.append(lab)
            else:
                lab = 'blank'
                words.append(w)
                labels.append(lab)
                
        yield words, labels
    
    def transform_data(self, inppath, outpath):
        df = pd.read_csv(inppath, sep='\t')
        line_list = df['sentence'].to_list()

        outfile = open(outpath, 'w')
        print('sentence_index,sentence,label', file=outfile)

        def process(ix, line):
            g = self.split_sen_with_label(line)
            words, labels = next(g)
            if len(words) == len(labels):
                return [ix+1," ".join(words)," ".join(labels)]

        out = Parallel(n_jobs=-1)(delayed(process)(ix, line) for ix, line in tqdm(enumerate(line_list)))

        for i in tqdm(out):
            if "," not in i[1]:
                print(*i, sep = ',', file=outfile)


        outfile.close()
    
    def upload_file_to_bucket(self, src, dst):
        cmd = 'gsutil -m cp ' + src + ' ' + dst
        print(cmd)
        os.system(cmd)
    
    def get_training_data(self, training_folder):
        
        PROJECT_NAME = self.names['PROJECT_NAME']
        TRAIN_CLEAN_NAME = self.names['TRAIN_CLEAN_NAME']
        VALID_CLEAN_NAME = self.names['VALID_CLEAN_NAME']
        TEST_CLEAN_NAME = self.names['TEST_CLEAN_NAME']

        TRAIN_NAME = self.names['TRAIN_NAME']
        VALID_NAME = self.names['VALID_NAME']
        TEST_NAME = self.names['TEST_NAME']
        VERSION = self.names['VERSION']

        self.transform_data(inppath=os.path.join(training_folder, 'train_processed.tsv'), outpath=os.path.join(training_folder, f'train_{VERSION}.csv'))
        self.transform_data(inppath=os.path.join(training_folder, 'valid_processed.tsv'), outpath=os.path.join(training_folder, f'valid_{VERSION}.csv'))
        self.transform_data(inppath=os.path.join(training_folder, 'test_processed.tsv'), outpath=os.path.join(training_folder, f'test_{VERSION}.csv'))
        

if __name__ == '__main__':
    # PrepareCsv().upload_file_to_bucket()
    pass

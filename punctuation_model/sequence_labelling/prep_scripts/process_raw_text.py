import re
import os
import wandb
import yaml
import argparse
import random
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from bnunicodenormalizer import Normalizer 
bnorm=Normalizer()

from indicnlp.tokenize.indic_tokenize import trivial_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

def normalize(text):
    _words = [bnorm(word)['normalized']  for word in text.split()]
    text =  " ".join([word for word in _words if word is not None])
    return text

class ProcessRawText:
    def __init__(self, config_path):
        self.config_path = config_path
        self.names = yaml.safe_load(open(self.config_path))
    
    def make_pattern_from_dict(self, dictionary):
        '''
        generate pattern for filtering text using dictionary file
        '''
        with open(dictionary, mode = 'r', encoding='UTF-8') as file:
            dictionary = file.readlines()
        char = [i.split(' ')[0] for i in dictionary] 
        pattern = '[^ '+''.join(char)+'।,?!]'
        return pattern

    def find_punctuation_count(self, line):
        '''
        count values of punctuation comma, end and qm in each sentence
        '''
        sent = []
        sent.append(line)
        punc = re.findall('[।,?!]+', line)
        return sent + [punc.count(ch) for ch in [',', '।', '?','!']]

    def process_sent(self, sent):
        '''
        normalize and tokenize sentence
        '''
        normalized = sent
        processed = ' '.join(trivial_tokenize(normalized, lang))
        return processed

    def filter_line(self, line):
        '''
        replace foreign characters not punctuation and numerals with space
        '''
        out = None

        if re.search(punc,line):  
            line = self.process_sent(line.strip())
            clean_line = re.sub(pattern, ' ', line)
            # fix for first character punctation
            if not (clean_line and (clean_line[0] in ['।', ',', '?','!'])):
                # fix for consecutive punctuation
                temp_line = clean_line.replace(" ","")
                if not regex.search(temp_line):
                    out =  ' '.join(clean_line.split())
        return out
    
    def upload_file_to_bucket(self, src, dst):
        cmd = 'gsutil -m cp ' + src + ' ' + dst
        print(cmd)
        os.system(cmd)

    def get_clean_data(self, processed_folder):
        PROJECT_NAME = self.names['PROJECT_NAME']
        RAW_DATA_NAME = self.names['RAW_DATA_NAME']
        DICT_NAME = self.names['DICT_NAME']
        CLEAN_DATA_NAME = self.names['CLEAN_DATA_NAME']
        global lang
        lang = self.names['LANG']
        inpfname, outfname = self.names['RAW_FILE_NAME'], self.names['PROCESSED_TSV_FILENAME']

        text_dir = processed_folder
        
        d_dir = "/workspace/test/pretrained/lm_ai4bharat/"

        # global pattern, punc, regex, normalizer
        global pattern, punc, regex

        punc = '[।,?!]+'
        regex = re.compile(r'[।,?!]{2,}')
        pattern = self.make_pattern_from_dict(os.path.join(d_dir, 'dict.ltr.txt'))

        line_list = []
        with open(os.path.join(text_dir, inpfname)) as inpfile:
            line_list.extend(inpfile.readlines())

        print(len(line_list))
        line_list = [l.strip() for l in line_list]
        line_list = [l for l in line_list if (len(l)>0) and (len(l)<512) and (l[-1] in ['।','?','!'])]
        random.seed(400)
        line_list = random.sample(line_list, 10_000_000)
        print(len(line_list))

        outfile = open(os.path.join(text_dir, outfname), 'w')  
        print("sentence\tcomma_count\tend_count\tqm_count\texclm_count", file=outfile)

        gen = (self.filter_line(line) for line in line_list)

        out = Parallel(n_jobs=64)(delayed(self.find_punctuation_count)(line) for line in tqdm(gen) if line)

        for line in tqdm(out):
            print(*line, sep="\t", file=outfile)

        outfile.close()

        

if __name__ == '__main__':
    # ProcessRawText(config_path='').get_clean_data(
    #     processed_folder='',
    #     lang=''
    # )
    pass

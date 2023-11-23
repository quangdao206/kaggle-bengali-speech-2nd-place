import yaml
import wandb
import pandas as pd
import os
import random

class SplitData:
    def __init__(self, config_path):
        self.config_path = config_path
        self.names = yaml.safe_load(open(self.config_path))
    
    def get_sample_data(self, data, count):
        # sampled_data = data
        random_state = 400
        qm_df = data[data.qm_count>0].sample(n=int(count*0.1), random_state=random_state)
        exclm_df = data[data.exclm_count>0].sample(n=int(count*0.05), random_state=random_state)
        sampled_data = data[(~data.sentence.isin(qm_df.sentence)) & (~data.sentence.isin(exclm_df.sentence))].sample(n=int(count*0.85), random_state=random_state)
        sampled_data = pd.concat([sampled_data, qm_df, exclm_df])

        # random.seed(200)
        # indices_to_sample = random.sample(list(range(data.shape[0])), count)
        # sampled_data = data.iloc[indices_to_sample]

        sampled_data.dropna(inplace=True)
        sampled_data.reset_index(drop=True, inplace= True)
        print(f"sample data shape : {sampled_data.shape}")
        return sampled_data
    
    def select_rows_from_df(sefl, df_old, ix):
        df_new = df_old.loc[df_old.index[ix]]
        df_new.reset_index(drop=True, inplace=True)
        return df_new
    
    def extract_punc_ration(self, data):
        total_end = data['end_count'].sum()
        total_comma = data['comma_count'].sum()
        total_qm = data['qm_count'].sum()
        total_exclm = data['exclm_count'].sum()

        total_punctuations = total_comma+total_qm+total_end+total_exclm
        end_ratio = total_end/total_punctuations
        comma_ratio = total_comma/total_punctuations
        qm_ratio = total_qm/total_punctuations
        exclm_ratio = total_exclm/total_punctuations

        print(f"end : {end_ratio:.3f}, commma : {comma_ratio:.3f}, qm : {qm_ratio:.3f}, exclm : {exclm_ratio:.3f}")
        
    
    def split(self, data, count, data_dir):
        indices_to_sample = random.sample(list(range(data.shape[0])), count)
        num_lines = data.shape[0]
        train_indices = set(range(num_lines)) - set(indices_to_sample)
        train_indices = list(train_indices)
        count_test_valid = count // 2
        valid_indices = indices_to_sample[:count_test_valid]
        test_indices = indices_to_sample[count_test_valid:]

        df_train = self.select_rows_from_df(data, train_indices)
        df_valid = self.select_rows_from_df(data, valid_indices)
        df_test = self.select_rows_from_df(data, test_indices)

        print(f"Length -> train : {df_train.shape[0]}, valid : {df_valid.shape[0]}, test : {df_test.shape[0]}")

        print("punctutaion ratio train :")
        self.extract_punc_ration(df_train)
        print("punctutaion ratio valid :")
        self.extract_punc_ration(df_valid)
        print("punctutaion ratio test :")
        self.extract_punc_ration(df_test)
        
        df_train.to_csv(os.path.join(data_dir, 'train_processed.tsv'), sep='\t',  index=False)
        df_valid.to_csv(os.path.join(data_dir, 'valid_processed.tsv'), sep='\t',  index=False)
        df_test.to_csv(os.path.join(data_dir, 'test_processed.tsv'), sep='\t',  index=False)
    
    def upload_file_to_bucket(self, src, dst):
        cmd = 'gsutil -m cp ' + src + ' ' + dst
        print(cmd)
        os.system(cmd)
    
    def split_data(self, processed_folder):
        data_size = self.names['SAMPLE_LEN']
        test_valid_size = self.names['TEST_AND_VALID_LEN']
        PROJECT_NAME = self.names['PROJECT_NAME']
        CLEAN_DATA_NAME = self.names['CLEAN_DATA_NAME']
        TRAIN_CLEAN_NAME = self.names['TRAIN_CLEAN_NAME']
        VALID_CLEAN_NAME = self.names['VALID_CLEAN_NAME']
        TEST_CLEAN_NAME = self.names['TEST_CLEAN_NAME']

        data_dir = processed_folder

        fname = self.names['PROCESSED_TSV_FILENAME']

        df = pd.read_csv(os.path.join(data_dir, fname), sep='\t')

        df = self.get_sample_data(df, data_size)

        self.split(df, test_valid_size, data_dir)

        pass

if __name__ == '__main__':
    # SplitData().split_data()
    pass

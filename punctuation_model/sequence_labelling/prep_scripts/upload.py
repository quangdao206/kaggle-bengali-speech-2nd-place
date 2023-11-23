import wandb
import yaml
import os

class Upload:
    def __init__(self, config_path):
        self.config_path = config_path
        self.names = yaml.safe_load(open(self.config_path))

    def upload_raw_data(self, raw_folder):
        PROJECT_NAME = self.names['PROJECT_NAME']
        RAW_DATA_NAME = self.names['RAW_DATA_NAME']
        DICT_NAME = self.names['DICT_NAME']
        fname = self.names['RAW_FILE_NAME']

        run = wandb.init(project=PROJECT_NAME, job_type="upload")

        raw_data_at = wandb.Artifact(RAW_DATA_NAME,type="raw_data")
        
        raw_data_loc = os.path.join(raw_folder, fname)
        raw_data_at.add_reference(raw_data_loc)
        run.log_artifact(raw_data_at)

        dict_at = wandb.Artifact(DICT_NAME,type="dictionary")

        dict_loc = os.path.join(raw_folder,'dict.ltr.txt')
        dict_at.add_reference(dict_loc)
        run.log_artifact(dict_at)

        run.finish()

if __name__ == '__main__':
    # Upload(config_path='wandb_artifact_config.yaml').upload_raw_data(
    #     raw_folder='gs://punctuation-itn/punctuation_data/telugu/data/raw',
    #     lang='te'
    # )
    pass
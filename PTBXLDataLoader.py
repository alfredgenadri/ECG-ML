from requests import get
from zipfile import ZipFile
from io import BytesIO
from os import path, getcwd

import pandas as pd
import numpy as np
import wfdb
import ast
from tqdm import tqdm

class PTBXLDataLoader:

    dataset_name = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
    _dataset = 'https://physionet.org/static/published-projects/ptb-xl/{name}.zip'.format(name=dataset_name[:-1])

    def __init__(self, zip_url: str=_dataset, local_dir: str='ecg_dataset/', high_res: bool=False, val_fold: int=10) -> None:
        
        self.url = zip_url
        self._local_dir = local_dir + self.dataset_name
        self._high_res = high_res
        self._val_fold = val_fold

    def _download_data(self) -> None:
        
        response = get(self.url)
        if response.ok:
            zip = ZipFile(BytesIO(response.content))
            zip.extractall(path.join(getcwd(), self._local_dir))
        else:
            print('GET request failed')

    def _load_raw_data(self, df, path):
        
        return [wfdb.rdsamp(path + f) for f in tqdm(df.filename_hr if self._high_res else df.filename_lr)]

    def _generate_datasets(self) -> tuple:

        # read patient database and convert SCP-ECG codes to dict
        df = pd.read_csv(self._local_dir + 'ptbxl_database.csv', index_col='ecg_id')
        df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
        
        # read SCP-ECG code database and keep arrhythmias with diagnostic for aggregation with patient database
        scp_df = pd.read_csv(self._local_dir + 'scp_statements.csv', index_col=0)
        scp_df = scp_df[scp_df.diagnostic == 1]

        # Diagnostic class aggregation between patient and SCP-ECG code dataframes
        df['diagnostic_superclass'] = df.scp_codes.apply(lambda x: list(set([scp_df.loc[key].diagnostic_class for key in x.keys() if key in scp_df.index])))
        df['len_diagnostic_superclass'] = df['diagnostic_superclass'].apply(len)
        df = df[df.len_diagnostic_superclass > 0]

        classes = pd.Series(np.concatenate(df['diagnostic_superclass'].values)).unique()
        labels = pd.DataFrame(0, index=df.index, columns=classes, dtype='int')
        
        for i in labels.index:
            for k in df.loc[i].diagnostic_superclass:
                labels.loc[i, k] = 1

        raw = self._load_raw_data(df, self._local_dir)
        data = np.array([signal for signal, meta in raw])

        return df, data, labels, classes
    
    def _create_train_test_sets(self, df, data, labels, classes) -> tuple:

        train_inputs = data[np.where(df.strat_fold < self._val_fold - 1)]
        train_labels = labels[(df.strat_fold < self._val_fold - 1)]

        val_inputs = data[np.where(df.strat_fold == self._val_fold - 1)]
        val_labels = labels[(df.strat_fold == self._val_fold - 1)]

        test_inputs = data[np.where(df.strat_fold == self._val_fold)]
        test_labels = labels[(df.strat_fold == self._val_fold)]

        return train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels, classes

    def load_data(self) -> tuple:
        
        if not path.exists(path.join(getcwd(), 'ecg_dataset')):
            self._download_data()
        df, data, labels, classes = self._generate_datasets()
        return self._create_train_test_sets(df, data, labels, classes)

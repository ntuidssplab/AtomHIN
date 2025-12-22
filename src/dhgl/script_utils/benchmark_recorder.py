import os

import pandas as pd
from filelock import FileLock


class BenchmarkRecorder:

    def __init__(self, csv_file: str, file_lock_dir: str = '/tmp/benchmark'):
        self.csv_file = csv_file
        os.makedirs(file_lock_dir, exist_ok=True)
        filename = os.path.realpath(csv_file).replace(os.sep, '.')[-50:]
        self.lock = os.path.join(file_lock_dir, f'{filename}.lock')
        self.df = None
        return

    def get_df(self):
        if os.path.isfile(self.csv_file):
            self.df = pd.read_csv(self.csv_file)
            return self.df
        return None

    def __len__(self):
        df = self.get_df()
        if df is None:
            return 0
        return len(df)

    def add_row(self, metrics: dict):
        with FileLock(self.lock):
            if os.path.isfile(self.csv_file):
                df = pd.read_csv(self.csv_file)
                row = pd.DataFrame(metrics, index=[0])
                df = pd.concat([df, row]).reset_index(drop=True)

            else:
                df = pd.DataFrame(metrics, index=[0])

            df.to_csv(self.csv_file, index=False)
        return

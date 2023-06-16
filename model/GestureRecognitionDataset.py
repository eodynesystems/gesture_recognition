import pandas as pd
from glob import glob
from pathlib import PureWindowsPath
from tqdm import tqdm
from Signal import Signal
import os

class GestureRecognitionDataset():
    def __init__(self, path_to_dataset, save_df=True):
        self.path_to_dataset = path_to_dataset
        self.save_df=save_df
    
    def recording_ok(self, fpath):
        try:
            Signal(path_to_file=fpath)
            return True
        except:
            return False
        
    def get_features(self):
        files = glob(f"{self.path_to_dataset}/*/*.npy")
        df = None
        for fpath in tqdm(files):
            if not self.recording_ok(fpath):
                os.remove(fpath)
                continue

            win_fpath = PureWindowsPath(fpath)
            subject = win_fpath.parts[-2][:-1]
            take = win_fpath.parts[-2][-1]
            gesture = win_fpath.stem.split("_")[0]
            iteration = int(win_fpath.stem.split("_")[1])
            
            signal = Signal(path_to_file=fpath)
            features = signal.get_features()
            if df is None:
                columns = []
                for feat in signal.features.keys():
                    columns.extend([f"{feat}_{idx}" for idx in range(8)])
                columns.extend(["gesture", "subject", "take"])
                df=pd.DataFrame(columns=columns)
            
            temp_df = pd.DataFrame(features, columns = columns[:-3])
            temp_df["gesture"] = [gesture]*temp_df.shape[0]
            temp_df["iteration"] = [iteration]*temp_df.shape[0]
            temp_df["subject"] = [subject]*temp_df.shape[0]
            temp_df["take"] = [take]*temp_df.shape[0]

            df = pd.concat([df, temp_df], ignore_index=True)

        if self.save_df:
            df.to_csv("../data/features.csv", index=None)

        return df
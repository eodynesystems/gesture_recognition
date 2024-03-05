import pandas as pd
from glob import glob
from pathlib import PureWindowsPath
from tqdm import tqdm
from model import Signal
import os

class GestureRecognitionDataset():
    def __init__(self, path_to_dataset, version = "v1", save_df=True, remove_transition=False):
        self.path_to_dataset = path_to_dataset
        self.save_df=save_df
        self.version = version
        self.remove_transition = remove_transition
    
    def recording_ok(self, fpath):
        try:
            Signal(path_to_file=fpath)
            return True
        except:
            return False
    
    def timestamp_to_iter(self, df):
        "converts timestamp to iteration in a dataframe"
        result = pd.DataFrame(columns=df.columns)
        for subject in df.subject.unique():
            df_sub = df[df.subject == subject]
            for gesture in df_sub.gesture.unique():
                df_ges = df_sub[df_sub.gesture == gesture]
                for take in df_ges["take"].unique():
                    df_take = df_ges[df_ges["take"] == take]
                    num_dict = {i:idx+1 for idx, i in enumerate(df_take.timestamp.unique())}
                    temp_df = df_take.replace(to_replace=num_dict)
                    result = pd.concat([result, temp_df])
        result.rename(columns = {"timestamp":"iteration"}, inplace=True)
        return result
    

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
            

            if self.version == "v1":
                gesture = win_fpath.stem.split("_")[0]
                iteration = int(win_fpath.stem.split("_")[1])
            
            if self.version == "v2":
                gesture = win_fpath.stem[:-20]
                timestamp = win_fpath.stem[-19:]
                
            signal = Signal(path_to_file=fpath)
            if (gesture == "Neutral" or gesture == "hand_neutral"):
                features = signal.get_features()
            else:
                features = signal.get_features(remove_transition=self.remove_transition)
            if df is None:
                columns = []
                for feat in signal.features.keys():
                    columns.extend([f"{feat}_{idx}" for idx in range(8)])
                columns.extend(["gesture", "subject", "take"])
                df=pd.DataFrame(columns=columns)
            
            temp_df = pd.DataFrame(features, columns = columns[:-3])
            temp_df["gesture"] = [gesture]*temp_df.shape[0]

            if self.version == "v1":
                temp_df["iteration"] = [iteration]*temp_df.shape[0]
            if self.version == "v2":
                temp_df["timestamp"] = [timestamp]*temp_df.shape[0]

            temp_df["subject"] = [subject]*temp_df.shape[0]
            temp_df["take"] = [take]*temp_df.shape[0]

            df = pd.concat([df, temp_df], ignore_index=True)

        if self.save_df:
            df.to_csv(f"../data/features_{self.version}.csv", index=None)

        return df
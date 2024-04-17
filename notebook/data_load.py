data_dir = "C:/Users/fents/Documents/iLimb/DataCollection/data"
window_size = 250
step_size = 50

from model import Signal, GestureRecognitionModel
from model import GestureRecognitionDataset

import seaborn as sns
import numpy as np 
import ruptures as rpt
import pandas as pd
import json
import bottleneck as bn
import matplotlib.pyplot as plt

from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix
from tqdm import tqdm 
from glob import glob 
from pathlib import PureWindowsPath
from sklearn.svm import SVC  # Support Vector Classifier

from scipy.interpolate import interp1d

class Signal():
    def __init__(self, path_to_file=None, signal=None, step=10, window_size = 50):
        """
        Args:
        path_to_file (str): path to file containing numpy array for the signal
        signal (np.array): numpy array containing the signal
        Both arguments are optional. 
        This class can be instantiated to just use the functions.
        """
        if path_to_file:
            self.signal = np.load(path_to_file)
            
        else:
            if len(np.array(signal).shape) == 1:
                self.signal = np.expand_dims(np.array(signal), axis =1)
            else: 
                self.signal = np.array(signal)

        self.n_samples = self.signal.shape[0]
        self.step = step
        self.window_size = window_size
        if self.n_samples < 50:
            raise Exception ("signal too short! minimum size = 50 samples")
    
        self.features = {"mav":self.mav, "rms":self.rms, "ssc":self.ssc, "wl":self.wl, "var":self.var, 
                         "iasd":self.iasd, "iatd":self.iatd}
        self.n_features = self.signal.shape[1]*len(self.features)
        
    def get_features(self, list_features = "all", remove_transition=False):
        if remove_transition:
            self.remove_transition()
        features = np.empty((0, self.n_features))
        for idx in range(0, self.n_samples-self.window_size, self.step):
            x = self.signal[idx:idx+self.window_size, :]
            features = np.concatenate((features, self.get_features_window(x, list_features)))
        return features
    
    def get_features_window(self, x, list_features="all"):
        features = np.empty((1, 0))
        for f in self.features.values():
            features = np.concatenate((features, f(x).reshape(1, x.shape[1])), axis=1)
        return features

    def remove_transition(self):
        # detection using the mean  
        s = self.get_features(list_features=["mav"]).sum(axis=1)
        trans_idx = 0
        for i in range(0, len(s)-window_size):
            window = s[i:i+window_size]
            if (np.mean(window) > np.mean(s)) :
                break
            trans_idx+=1
        trans_idx = trans_idx*self.step
        self.signal = self.signal[trans_idx:, :]
        self.n_samples = self.signal.shape[0]
    

    # ops 
    def mav(self, x):                                      # mean absolute value
        return sum(abs(x)) / x.shape[0]

    def rms(self, x):                                      # root mean square
        return ((sum(x ** 2)) / x.shape[0]) ** (1 / 2)

    def wl (self, x):                                      # waveform length
        return sum(abs(x[:-1] - x[1:]))

    def ssc(self, x, delta = 1):                           # slope of sign change
        f = lambda x: (x >= delta).astype(float)
        return sum(f(-(x[1:-1, :] - x[:-2, :])*(x[1:-1] - x[2:])))
    
    def var(self, x):                                      # variance
        return sum((x ** 2)) / (x.shape[0] - 1)
    
    def derivative(self, x):
        return x[1:] - x[:-1]
    
    def iasd(self, x):                                     # integrated absolute of second derivative
        return (sum(abs(self.derivative(self.derivative(x)))))
    
    def iatd(self, x):                                     # integrated absolute of third derivative
        return (sum(abs(self.derivative(self.derivative(self.derivative(x))))))
    
    def sliding_avg(self, x, w=1):                         # sliding average with window size w
        return np.convolve(x, np.ones(w), 'valid') / w
    
    def sliding_var(self, x, w=1):
        return bn.move_var(x, window=w)
    
    #vis    
    def display(self, attr = "energy", w = 5):
        plt.figure()
        """
        w (window size) only applicable for functions of energy
        """
        if attr == "energy":
            energy = self.get_features(list_features=["mav"])
            energy = energy.sum(axis = 1)
            plt.plot(energy)
            plt.xlabel("time (samples)")
            plt.ylabel("Energy")
        if attr == "avg_slope_energy":
            energy = self.get_features(list_features=["mav"])
            energy = energy.sum(axis = 1)
            slope = energy[1:] - energy[:-1]
            avg_slope = self.sliding_avg(slope, w)
            plt.plot(avg_slope)
            plt.xlabel("time (samples)")
            plt.ylabel(f"Avg. slope (window = {w})")
        if attr == "sliding_var_energy":
            energy = self.get_features(list_features=["mav"])
            energy = energy.sum(axis = 1)
            mov_var = bn.move_var(energy, window=w)
            mov_avg = bn.move_mean(energy, window=w)
            plt.plot(mov_var/mov_avg)
            plt.xlabel("time (samples)")
            plt.ylabel(f"Moving Variance of Energy (window = {w})")
        ylim_bottom = min(0, plt.gca().get_ylim()[0])
        plt.gca().set_ylim(bottom=ylim_bottom)


def parse_ossur_sensor_recording(fpath):
    with open(fpath) as f:
        data = json.load(f)
    emg_shape = min(len(data['1264']['value1']), len(data['1265']['value1'])), 4
    emg = np.zeros(emg_shape)
    emg[:,0] = data['1264']['value1'][:emg_shape[0]] 
    emg[:,1] = data['1264']['value2'][:emg_shape[0]]
    emg[:,2] = data['1265']['value1'][:emg_shape[0]]
    emg[:,3] = data['1265']['value2'][:emg_shape[0]]

    disp_shape = min(len(data['528']['value1']), len(data['529']['value1'])), 4
    disp = np.zeros(disp_shape)
    disp[:,0] = data['528']['value1'][:disp_shape[0]]
    disp[:,1] = data['528']['value2'][:disp_shape[0]]
    disp[:,2] = data['529']['value1'][:disp_shape[0]]
    disp[:,3] = data['529']['value2'][:disp_shape[0]]

    return emg, disp

def resample(signal, n):
    # Initialize the array to hold the resampled signal
    resampled_signal_2d = np.zeros((n, signal.shape[1]))
    
    # New sample points for resampling
    new_positions = np.linspace(0, 1, n, endpoint=False)
    
    # Original positions corresponding to the existing samples
    original_positions = np.linspace(0, 1, signal.shape[0], endpoint=False)
    
    # Apply resampling for each row
    for i in range(signal.shape[1]):
        resampled_signal_2d[:, i] = np.interp(new_positions, original_positions, signal[:,i])
    
    return resampled_signal_2d

def data_load(subject):

    gestures_to_use = ["hand_close", "hand_open", 
                        "wrist_supin", "wrist_pron","thumb_abd", "thumb_add", "pinch", "lateral", "point"] 

    window_size = 250
    step_size = 50

    data_dir = "C:/Users/fents/Documents/DataCollection/data"

    files = glob(f"{data_dir}/{subject}*/*.npy")
    df = None

    for fpath in tqdm(files):

        win_fpath = PureWindowsPath(fpath)
        subject = win_fpath.parts[-2][:-1]
        take = win_fpath.parts[-2][-1]
        
        gesture = win_fpath.stem[:-20]
        timestamp = win_fpath.stem[-19:]
        
        emg, disp = parse_ossur_sensor_recording(fpath)
        emg = Signal(signal = emg, window_size=window_size, step=step_size)

        emg_features = emg.get_features()
        disp_features = resample(disp, emg_features.shape[0])
        features = np.concatenate([emg_features, disp_features], axis=1)

        if df is None:
            columns = []
            for feat in emg.features.keys():
                columns.extend([f"{feat}_{idx}" for idx in range(4)])
            
            columns.extend(["disp1","disp2", "disp3", "disp4", "gesture", "subject", "take"])
            df=pd.DataFrame(columns=columns)
        
        temp_df = pd.DataFrame(features, columns = columns[:-3])
        temp_df["gesture"] = [gesture]*temp_df.shape[0]

        
        temp_df["timestamp"] = [timestamp]*temp_df.shape[0]

        temp_df["subject"] = [subject]*temp_df.shape[0]
        temp_df["take"] = [take]*temp_df.shape[0]

        df = pd.concat([df, temp_df], ignore_index=True)

    ds= GestureRecognitionDataset(data_dir)
    df = ds.timestamp_to_iter(df)

    return df
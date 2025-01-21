import os
window_size = 250
step_size = 50

import numpy as np 
import bottleneck as bn
import matplotlib.pyplot as plt
import ruptures as rpt
import json
import matplotlib.pyplot as plt 
from data import GestureRecognitionDataset

from tqdm import tqdm 
from glob import glob 
from pathlib import PureWindowsPath
import pandas as pd
from model import GestureRecognitionModel


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
        s = self.get_features(list_features=["mav"]).sum(axis=1)
        # detection
        algo = rpt.Binseg(model="l2").fit(s)
        result = algo.predict(n_bkps=1)[0]
        if result > len(s)//2:
            result = 20
        trans_idx = result*self.step
        if self.signal.shape[0]>2*trans_idx:
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

def df_idx_to_emg_window(df_idx):
    return (df_idx*step_size, df_idx*step_size + step_size)


def get_recording(subject, gesture, iteration):
    files = glob(f"{data_dir}/{subject}*/{gesture}*.npy")
    files.sort(key=os.path.getmtime)
    return files[iteration-1]

    
def visualise_errors(subject, error_dict):
    # reshape the error dict to {fpath: list of errors}
    file_errors = {}
    for iteration, errors in error_dict.items():
        for gesture in errors.keys():
            fpath = get_recording(subject, gesture, iteration)
            file_errors[fpath] = (errors[gesture][0], gesture, errors[gesture][1])

    plt.figure(figsize = (10, len(file_errors)))
    for idx, (fpath, errors) in enumerate(sorted(file_errors.items(), key=lambda a:a[1][1])):
        emg, disp = parse_ossur_sensor_recording(fpath)
        flex = np.mean(emg[:, :2], axis=1)
        ext = np.mean(emg[:, 2:], axis=1)
        disp_flex = disp[:, 3]
        disp_ext = disp[:, 1]
        error_idxs = []
        for i in errors[0]:
            error_idxs.extend(list(range(*df_idx_to_emg_window(i))))
        error_flex = [f if i in error_idxs else np.nan for i, f in  enumerate(flex)]
        error_ext = [e if i in error_idxs else np.nan for i, e in enumerate(ext)]
        plt.subplot(len(file_errors), 5, idx*5+1)
        plt.plot(flex)
        if errors:
            plt.plot(error_flex, c='r')
        plt.xticks([])
        plt.ylim(-2000, 2000)
        if idx == 0:
            plt.title("extensor emg")
        plt.subplot(len(file_errors), 5, idx*5+2)
        plt.plot(ext)
        if errors:
            plt.plot(error_ext, c='r')
        plt.yticks([])
        plt.xticks([])
        plt.ylim(-2000, 2000)
        if idx == 0:
            plt.title("flexor emg")
        plt.subplot(len(file_errors), 5, idx*5+3)
        plt.plot(disp_flex, c="orange")
        plt.yticks([])

        plt.xticks([])
        plt.ylim(-2000, 2000)
        if idx == 0:
            plt.title("extensor disp")
        plt.subplot(len(file_errors), 5, idx*5+4)
        plt.plot(disp_ext, c="orange")
        plt.yticks([])
        plt.xticks([])
        plt.ylim(-2000, 2000)
        if idx == 0:
            plt.title("flexor disp")
        ax = plt.subplot(len(file_errors), 5, idx*5+5)
        plt.text(0.5, 0.8, errors[1],  horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, fontdict={"fontsize":13, 'color':'g'})
        plt.text(0.5, 0.3, errors[2],  horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, fontdict={"fontsize":9, 'color':'r'}, wrap=True)
        plt.axis("off")
    plt.suptitle(subject)
    plt.tight_layout()

def get_gesturewise_errors(y, preds):
    y, preds = np.array(y), np.array(preds)
    errors = {}
    for i in list(set(y)):
        gesture = int_to_gesture[i]
        gesture_idxs = np.where(np.array(y) == i)[0]
        gesture_ys = y[gesture_idxs]
        gesture_preds = preds[gesture_idxs]
        error_idxs = np.where(gesture_ys!=gesture_preds)
        wrong_preds = ", ".join(int_to_gesture[i] for i in set(gesture_preds[error_idxs]))
        errors[gesture] = (error_idxs[0], wrong_preds)
    return errors


data_dir = "C:/Users/rajsu/Documents/iLimb/DataCollection/OssurSensor/data"
protocol = "ossur"
subjects = os.listdir(data_dir)
for subject in tqdm(subjects):
    files = glob(f"{data_dir}/{subject}*/*.npy")
    df = None
    for fpath in files:
        win_fpath = PureWindowsPath(fpath)
        sub = win_fpath.parts[-2][:-1]
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

        temp_df["subject"] = [sub]*temp_df.shape[0]
        temp_df["take"] = [take]*temp_df.shape[0]

        df = pd.concat([df, temp_df], ignore_index=True)
    ds= GestureRecognitionDataset(data_dir)
    df = ds.timestamp_to_iter(df)

    gestures_to_use = ["hand_close", "hand_neutral", "hand_open", 
                        "wrist_supin", "wrist_pron","thumb_abd", "thumb_add", "pinch", "lateral", "point"] 

    accuracies = []
    preds = []
    ys=[]
    df = df[df.gesture.isin(gestures_to_use)]
    gestures = df.gesture.unique()
    gesture_to_int = {gesture:i for i, gesture in enumerate(gestures)}
    int_to_gesture = {val:key for key, val in gesture_to_int.items()}

    errors = {}
    if protocol == "ossur":
        for i in range(1,6):
            df_train = df[df["iteration"] != i]
            df_test = df[df["iteration"] == i]
            model = GestureRecognitionModel()
            features_to_keep = df_train.columns[:-4]
            X_train, y_train = df_train[features_to_keep], [gesture_to_int[gesture] for gesture in df_train.gesture]
            X_test, y_test = df_test[features_to_keep], [gesture_to_int[gesture] for gesture in df_test.gesture]
            """rus = RandomUnderSampler()
            X_test, y_test = rus.fit_resample(X_test, y_test)"""
            model.train(X_train, y_train)
            accuracies.append(model.evaluate(X_test, y_test))
            pred = model.predict(X_test)
            preds.extend(pred)
            ys.extend(y_test)
            errors[i] = get_gesturewise_errors(y_test, pred)

    #filter to files with over 3 errors
    for i in errors.keys():
        errors[i] = {key:value for key, value in errors[i].items() if len(value[0])>3}

    visualise_errors(subject, errors)
    plt.savefig(f"error_analysis\{subject}.png")
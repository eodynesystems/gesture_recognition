import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import json
from model import Signal, GestureRecognitionModel
from imblearn.under_sampling import RandomUnderSampler
from scipy.interpolate import interp1d
import sys
import os
from notebook.data_load import data_load


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

def failure_plot(subject,data_dir = "C:/Users/fents/Documents/iLimb/DataCollection/data"):

    gestures_to_use = ["hand_close", "hand_open", 
                        "wrist_supin", "wrist_pron","thumb_abd", "thumb_add", "pinch", "lateral", "point"] 
    
    df = data_load(subject)

    plot_first_row_only = False


    df = df[df.gesture.isin(gestures_to_use)]
    df.reset_index(drop=True, inplace=True)
    gestures = df.gesture.unique()
    gesture_to_int = {gesture: i for i, gesture in enumerate(gestures)}
    int_to_gesture = {val: key for key, val in gesture_to_int.items()}

    df = df[df.gesture.isin(gestures_to_use)]
    takes = df['take'].tolist()
    unique_values = list(set(takes))

    for take in unique_values: 

        ratios = {} 
        begin_end_numbers_by_set = {}
        misclassified_indices_by_set = {}
        indices_by_set = {}
        model=GestureRecognitionModel()

        df_take = df[df['take'] == take]

        for gesture in gestures_to_use:

            df_gesture = df_take[df_take['gesture'] == gesture]

            for i in range(1,6):
                df_train = df_take[df_take["iteration"] != i]
                df_test = df_gesture[df_gesture["iteration"] == i]
                features_to_keep = df_train.columns[:-4]
                X_train, y_train = df_train[features_to_keep], [gesture_to_int[gesture] for gesture in df_train.gesture]
                X_test = df_test[features_to_keep]
                y_test = [gesture_to_int[gesture] for gesture in df_test.gesture]

                model.train(X_train, y_train)
                preds = model.predict(X_test)
                ys = y_test

                misclassified_indices = [idx for idx, (pred, true) in enumerate(zip(preds, ys)) if pred != true]
                misclassified_indices_by_set[i] = misclassified_indices

                consecutive_sequences = []
                sequence = []

                for idx in misclassified_indices:
                    if len(sequence) == 0 or sequence[-1] == idx - 1:
                        sequence.append(idx)
                    else:
                        if len(sequence) >= 2:
                            consecutive_sequences.append((sequence[0], sequence[-1]))
                        sequence = [idx]

                if len(sequence) >= 2:
                    consecutive_sequences.append((sequence[0], sequence[-1]))

                begin_end_numbers_by_set = [num for seq in consecutive_sequences for num in seq]

                ratios[i] = [x / len(X_test) for x in begin_end_numbers_by_set]

            files = glob(f"{data_dir}/{subject}{take}/*{gesture}*.npy")
            titles = ["Movement 1", "Movement 2", "Movement 3", "Movement 4", "Movement 5"]

            plt.figure(figsize=(20, 10))

            for idx, fpath in enumerate(files):

                ratios_set = ratios[idx+1]

                emg, disp = parse_ossur_sensor_recording(fpath)
                flex = emg[:, :2]
                ext = emg[:, 2:]
                disp_flex = disp[:, 3]
                disp_ext = disp[:, 1]

                flex_signal = np.mean(flex, axis=1)
                ext_signal = np.mean(ext, axis=1)

                original_time_points = np.arange(len(disp_ext))
                new_time_points = np.linspace(0, len(disp_ext) - 1, len(flex_signal))

                interp_func = interp1d(original_time_points, disp_ext, kind='linear')
                disp_ext = interp_func(new_time_points)

                interp_func2 = interp1d(original_time_points, disp_flex, kind='linear')
                disp_flex = interp_func2(new_time_points)

                for i, (signal, ylabel) in enumerate(zip([flex_signal, ext_signal, disp_flex, disp_ext],
                                                        ["Ext EMG", "Flex EMG", "Ext FMG", "Flex FMG"])):

                    if plot_first_row_only and i != 0:  
                        continue  

                    ratios_converted = []
                    if i == 0:
                        ratios_converted = [x * len(flex_signal) for x in ratios_set]
                    elif i == 1:
                        ratios_converted = [x * len(ext_signal) for x in ratios_set]
                    elif i == 2:
                        ratios_converted = [x * len(disp_flex) for x in ratios_set]
                    elif i == 3:
                        ratios_converted = [x * len(disp_ext) for x in ratios_set]

                    plt.subplot(4, len(files), i * len(files) + idx + 1)
                    
                    if i > 1:
                        plt.plot(signal, c="blue", linewidth=2.5)
                    else:
                        plt.plot(signal, c="blue")

                    for j in range(0, len(ratios_converted), 2):
                        start = int(ratios_converted[j])
                        end = int(ratios_converted[j + 1])
                        x_values = range(start, end) 
                        y_values = signal[start:end]

                        if len(x_values) > 0: 
                            plt.plot(x_values, y_values, c="red",linewidth=3)

                    plt.ylim(-2000, 2000)
                    plt.xticks([] if i < 3 else None)

                    if idx == 0:
                        plt.ylabel(ylabel)
                        plt.yticks([-2000, 0, 2000])
                    else:
                        plt.yticks([])

                    if i == 0:
                        plt.title(titles[idx])

            plt.suptitle(subject +" "+ gesture + " take " + str(take))
            plt.tight_layout()

            save_path = os.path.join(data_dir, f"{subject}_{take}_{gesture}.pdf")

            plt.savefig(save_path)
            plt.show()  
            plt.close()  

if __name__ == "__main__":
    if len(sys.argv) != 3: 
        raise ValueError("Usage: script.py subject plot_first_row_only")
    subject = sys.argv[1]
    plot_first_row_only = bool(sys.argv[2])
    failure_plot(subject, plot_first_row_only)
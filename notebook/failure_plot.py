import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import json
from model import Signal, GestureRecognitionModel
from imblearn.under_sampling import RandomUnderSampler
from scipy.interpolate import interp1d

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

def failure_plot(df, gesture_to_int, subject="Sergio",gesture="wrist_supin",data_dir="C:/Users/fents/Documents/iLimb/DataCollection/data"):
 
    window_size = 250
    step_size = 50

    df =  df[(df['subject'] == subject)]
    files = glob(f"{data_dir}/{subject}*/{gesture}*.npy")

    accuracies = []
    ratios_dict = {}
    incorrect_predictions_by_set = {}

    for i in range(1, 6):
        df_train = df[df["iteration"] != i]
        df_test = df[df["iteration"] == i]
        model = GestureRecognitionModel()
        features_to_keep = df_train.columns[:-4]
        X_train, y_train = df_train[features_to_keep], [gesture_to_int[gesture] for gesture in df_train.gesture]
        X_test, y_test = df_test[features_to_keep], [gesture_to_int[gesture] for gesture in df_test.gesture]
        rus = RandomUnderSampler()
        X_test, y_test = rus.fit_resample(X_test, y_test)
        model.train(X_train, y_train)
        accuracies.append(model.evaluate(X_test, y_test))
        preds = model.predict(X_test)
        ys = y_test

        incorrect_predictions = [idx for idx in range(len(preds)) if preds[idx] != ys[idx]]
        
        incorrect_predictions_by_set[i] = incorrect_predictions

    for test_set_index in range(1, 6):
        incorrect_indices = incorrect_predictions_by_set[test_set_index]

        consecutive_sequences = []
        current_sequence = []
        min_length = 2

        for number in incorrect_indices:
            if not current_sequence or number == current_sequence[-1] + 1:
                current_sequence.append(number)
            else:
                if len(current_sequence) >= min_length:
                    consecutive_sequences.extend(current_sequence)
                current_sequence = [number]

        if len(current_sequence) >= min_length:
            consecutive_sequences.extend(current_sequence)

        selected_numbers = []
        current_sequence = []

        for number in consecutive_sequences:
            if not current_sequence or number == current_sequence[-1] + 1:
                current_sequence.append(number)
            else:
                if len(current_sequence) >= 2:
                    selected_numbers.extend([current_sequence[0], current_sequence[-1]])
                current_sequence = [number]

        if len(current_sequence) >= 2:
            selected_numbers.extend([current_sequence[0], current_sequence[-1]])

        df2 = df[(df['iteration'] == test_set_index)]
        df2 = df2.reset_index(drop=True)

        ratios = [x / len(df2) for x in selected_numbers]
        
        ratios_dict[test_set_index] = ratios

    titles = ["Movement 1", "Movement 2", "Movement 3", "Movement 4", "Movement 5"]

    plt.figure(figsize=(20, 10))

    for idx, fpath in enumerate(files):

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

        ratios = ratios_dict[idx+1]

        for i, (signal, ylabel) in enumerate(zip([flex_signal, ext_signal, disp_flex, disp_ext],
                                                ["Ext EMG", "Flex EMG", "Ext FMG", "Flex FMG"])):
            if i == 0:
                ratios_converted = [x * len(flex_signal) for x in ratios]
            if i == 1:
                ratios_converted = [x * len(ext_signal) for x in ratios]
            if i == 2:
                ratios_converted = [x * len(disp_flex) for x in ratios]
            if i == 3:
                ratios_converted = [x * len(disp_ext) for x in ratios]

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

    plt.suptitle(gesture)
    plt.tight_layout()
    plt.show()
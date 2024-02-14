# Gesture Recognition Documentation

## Overview

This Python project, "Gesture Recognition," is designed for processing and analyzing gesture signal data, extracting relevant features, and training machine learning models for recognition. The project consists of three main modules:

1. **GestureRecognitionDataset.py**: This module is responsible for loading gesture signal data, processing it, extracting features, and saving them as CSV files. It also provides methods for filtering out invalid recordings and converting timestamp data to iteration numbers.

2. **GestureRecognitionModel.py**: This module contains the machine learning model for recognizing gestures. Currently, it supports XGBoost as the machine learning algorithm.

3. **Signal.py**: This module is dedicated to handling signal processing tasks. It provides methods for extracting various signal features, such as Mean Absolute Value (MAV), Root Mean Square (RMS), Slope of Sign Change (SSC), Waveform Length (WL), Variance (VAR), Integrated Absolute of Second Derivative (IASD), and Integrated Absolute of Third Derivative (IATD).

## GestureRecognitionDataset.py

### GestureRecognitionDataset Class

#### `__init__(self, path_to_dataset, version="v1", save_df=True, remove_transition=False)`

- **Parameters**:
    - `path_to_dataset` (str): The path to the dataset containing gesture signal recordings.
    - `version` (str, optional): The dataset version (default is "v1").
    - `save_df` (bool, optional): Whether to save the processed dataset as a CSV file (default is True).
    - `remove_transition` (bool, optional): Whether to remove transitions in the signal data (default is False).

#### `recording_ok(self, fpath)`

- **Parameters**:
    - `fpath` (str): File path to a gesture signal recording.

- **Returns**: 
    - `True` if the recording is valid; `False` otherwise.

#### `timestamp_to_iter(self, df)`

- **Parameters**:
    - `df` (DataFrame): DataFrame containing gesture signal data with timestamps.

- **Returns**: 
    - DataFrame with timestamps converted to iteration numbers.

#### `get_features(self)`

- **Returns**:
    - DataFrame containing extracted signal features for gesture recognition.

## GestureRecognitionModel.py

### GestureRecognitionModel Class

#### `__init__(self, model_name="xgboost")`

- **Parameters**:
    - `model_name` (str, optional): The name of the machine learning model (default is "xgboost").

#### `train(self, x, y)`

- **Parameters**:
    - `x` (array-like): Input features for training.
    - `y` (array-like): Target labels for training.

#### `evaluate(self, x, y)`

- **Parameters**:
    - `x` (array-like): Input features for evaluation.
    - `y` (array-like): Target labels for evaluation.

- **Returns**:
    - The accuracy score of the trained model on the evaluation data.

#### `predict(self, x)`

- **Parameters**:
    - `x` (array-like): Input features for prediction.

- **Returns**:
    - Predicted labels based on the input features.

## Signal.py

### Signal

#### `__init__(self, path_to_file=None, signal=None, step=10, window_size=50)`

- **Parameters**:
    - `path_to_file` (str, optional): Path to a file containing a numpy array for the signal data.
    - `signal` (np.array, optional): Numpy array containing the signal data.
    - `step` (int, optional): Step size for signal processing (default is 10).
    - `window_size` (int, optional): Size of the window for feature extraction (default is 50).

#### `get_features(self, list_features="all", remove_transition=False)`

- **Parameters**:
    - `list_features` (str or list, optional): List of features to extract ("all" or a list of feature names).
    - `remove_transition` (bool, optional): Whether to remove transitions in the signal data (default is False).

- **Returns**:
    - Numpy array containing extracted signal features.

#### `get_features_window(self, x, list_features="all")`

- **Parameters**:
    - `x` (np.array): Input signal data within a specific window.
    - `list_features` (str or list, optional): List of features to extract ("all" or a list of feature names).

- **Returns**:
    - Numpy array containing extracted signal features within the window.

#### `remove_transition(self, algorithm, w, min_s)`
Removes transitions in the signal using a change-point detection algorithm. Depending on parameter 'algorithm', one could choose the standard implication of the "Binseg" algorithm, or the "Pelt" algorithm which is adaptive in parameters:

- `w`: the window size of the implemented sliding_avg function.
- `min_s`: the minimal number of data points between two change points that the algorithm considers valid, playing a crucial role in the sensitivity of the algorithm.

#### Signal Processing Functions

The Signal class provides various signal processing functions, including:

- `mav`: Mean Absolute Value
- `rms`: Root Mean Square
- `wl`: Waveform Length
- `ssc`: Slope of Sign Change
- `var`: Variance
- `iasd`: Integrated Absolute of Second Derivative
- `iatd`: Integrated Absolute of Third Derivative

These functions can be used to extract specific signal features.

#### Visualization Functions

- `display(self, attr="energy", w=5)`: Visualizes signal features. Supported attributes include "energy," "avg_slope_energy," and "sliding_var_energy."


# Audio emotion recognition

## Description

The goal of this project is to provide script to verify our emotion recognition approach.

### Requirements

* numpy
* sklearn
* librosa (for pitch estimation, optional)
* keras

### Project structure
The code consist of 3 main .py files.

* model.py - containt keras implementation of LSTM, MPC neural networks, sklearn models

 Main methods to use in emotion recognition:
 * train_mpc(train_x, train_y, test_x, test_y) - train multilayer perceptron. 
 * train_lstm(train_x, train_y, test_x, test_y) - train LSTM NN
 * train_rfc(train_x, train_y, options) - train Random Forest classifier

 All methods return Model object and also show precision and loss on test sample to inline validation. 

* calculate_features.py - script for features estimation

 Method `calculate_features(signal, freq, options)` returns features set for all 0.2 sec frames in signal. Freq is a signal framerate. If use_derivatives flag is true, method also include 1st and 2nd time deltas of features.
 To calculate features cf.py code is used. It based on https://github.com/tyiannak/pyAudioAnalysis/blob/master/audioFeatureExtraction.py with minor improvements. 

 As default returns 32 mfcc, spectral and chromagram features

* code_yan_lstm.py - implementation of LSTM NN in emotion recognition. 

 The code can be divided into 3 blocks
  * Data reading 
  * Data preprocessing
  * The main part includes model buildg and validation
 
 To get the data successfully, choose regime from ['aibo4', 'aibo5', 'iemocap'] and correct paths to wavs, labels, etc. aibo4 and aibo5 regimes correspondings to different labels over AIBO database.
 
 Data preprocessing consist of normalization, padding sequences, transfromation labels from categorical to vector, balancing and resampling.

 The main part is a 5-fold cross-validation procedure over readed database. It splits sample into train and test parts, trains models on train and validates on test. By ending cross-validation procedure it plots confusion matrix aomparing prediction and expected output.

## Running

To run the code, enter `python code_yan_lstm.py` in a command line or IDE. In addition to correction paths and settings, create folder for feature samples in the working directory. 

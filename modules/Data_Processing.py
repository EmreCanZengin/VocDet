import numpy as np
import pandas as pd

import librosa
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA

N_MFCC= 13

class FeatureExtractionWithLibrosa(TransformerMixin, BaseEstimator):
    def __init__(self, *, sr= 22050, hop_length= 512, n_fft= 512):
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = []
        for file in X:
            signal, sr = librosa.load(file, sr= self.sr)
            mel_spectrogram = librosa.feature.melspectrogram(y= signal, hop_length= self.hop_length, n_fft= self.n_fft, window= "hann")

            mfcc = librosa.feature.mfcc(y= signal, sr= self.sr, S = librosa.power_to_db(mel_spectrogram), n_mfcc= N_MFCC)
            delta_mfcc = librosa.feature.delta(mfcc)
            delta_delta_mfcc = librosa.feature.delta(mfcc, order= 2)
            # their shapes depends on the data
            concatenated = np.concatenate([mfcc, delta_mfcc, delta_delta_mfcc], axis= 1)
            transformed_X.append(concatenated)


        return transformed_X


class PCAUncompatibleShapes(TransformerMixin, BaseEstimator):
    def __init__(self, row_mfcc= 1):
        self.row_mfcc = row_mfcc

    def fit(self, X, y= None):
        return self
    
    def transform(self, X: list):
        X_new = X.copy()
        n = len(X_new)
        pca = PCA(self.row_mfcc)

        X_transformed = np.zeros(shape=(n, self.row_mfcc, self.row_mfcc))
        for idx, ins in enumerate(X_new):
            X_transformed[idx] = pca.fit_transform(ins)
        return X_transformed

def flattenLastTwoDim(X:np.ndarray):
    return X.reshape(-1, X.shape[1] * X.shape[2])
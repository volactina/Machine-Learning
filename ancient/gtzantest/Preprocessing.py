import numpy as np
import pandas as pd
import scipy as sp
import sklearn
from gtzantest import datasets
import librosa
import pickle
import os

def extract_features(sample, n_chroma=12, n_octaves=7, n_mfcc=20, n_fft=2048, hop_length=512, n_bands=6):
    waveform = sample['waveform']
    sampling_rate = sample['sampling_rate']
    genre = sample['genre']

    chroma_cens_mean_fields = [
        "chroma_cens_mean_{}".format(j) for j in range(n_chroma)]
    chroma_cens_std_fields = [
        "chroma_cens_std_{}".format(j) for j in range(n_chroma)]
    mfcc_mean_fields = ["mfcc_mean_{}".format(j) for j in range(n_mfcc)]
    mfcc_std_fields = ["mfcc_std_{}".format(j) for j in range(n_mfcc)]
    spectral_contrast_mean_fields = [
        "spectral_contrast_mean_{}".format(j) for j in range(n_bands + 1)]
    spectral_contrast_std_fields = [
        "spectral_contrast_std_{}".format(j) for j in range(n_bands + 1)]
    tonnetz_mean_fields = ["tonnetz_mean_{}".format(j) for j in range(6)]
    tonnetz_std_fields = ["tonnetz_std_{}".format(j) for j in range(6)]

    x = {}

    waveform_harmonic, waveform_percussive = librosa.effects.hpss(waveform)

    chroma_cens = librosa.feature.chroma_cens(
        waveform_harmonic, sr=sampling_rate, hop_length=hop_length, n_chroma=n_chroma, n_octaves=n_octaves).T
    x.update(zip(chroma_cens_mean_fields, np.mean(chroma_cens, axis=0)))
    x.update(zip(chroma_cens_std_fields, np.std(chroma_cens, axis=0)))

    mfcc = librosa.feature.mfcc(
        waveform_harmonic, sr=sampling_rate, n_mfcc=n_mfcc).T
    x.update(zip(mfcc_mean_fields, np.mean(mfcc, axis=0)))
    x.update(zip(mfcc_std_fields, np.std(mfcc, axis=0)))

    rmse = librosa.feature.rmse(
        waveform, frame_length=n_fft, hop_length=hop_length).T
    x['rmse_mean'] = np.asscalar(np.mean(rmse, axis=0))
    x['rmse_std'] = np.asscalar(np.std(rmse, axis=0))

    spectral_centroid = librosa.feature.spectral_centroid(
        y=waveform, sr=sampling_rate, n_fft=n_fft, hop_length=hop_length).T
    x['spectral_centroid_mean'] = np.asscalar(
        np.mean(spectral_centroid, axis=0))
    x['spectral_centroid_std'] = np.asscalar(np.std(spectral_centroid, axis=0))
    x['spectral_centroid_skew'] = np.asscalar(sp.stats.skew(spectral_centroid))

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=waveform, sr=sampling_rate, n_fft=n_fft, hop_length=hop_length).T
    x['spectral_bandwidth_mean'] = np.asscalar(
        np.mean(spectral_bandwidth, axis=0))
    x['spectral_bandwidth_std'] = np.asscalar(
        np.std(spectral_bandwidth, axis=0))

    spectral_contrast = librosa.feature.spectral_contrast(
        y=waveform_harmonic, sr=sampling_rate, n_fft=n_fft, hop_length=hop_length).T
    x.update(zip(spectral_contrast_mean_fields,
                 np.mean(spectral_contrast, axis=0)))
    x.update(zip(spectral_contrast_std_fields,
                 np.std(spectral_contrast, axis=0)))

    spectral_flatness = librosa.feature.spectral_flatness(
        y=waveform, n_fft=n_fft, hop_length=hop_length).T
    x['spectral_flatness_mean'] = np.asscalar(
        np.mean(spectral_flatness, axis=0))
    x['spectral_flatness_std'] = np.asscalar(np.std(spectral_flatness, axis=0))

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=waveform, sr=sampling_rate, n_fft=n_fft, hop_length=hop_length).T
    x['spectral_rolloff_mean'] = np.asscalar(np.mean(spectral_rolloff, axis=0))
    x['spectral_rolloff_std'] = np.asscalar(np.std(spectral_rolloff, axis=0))
    x['spectral_rolloff_skew'] = np.asscalar(sp.stats.skew(spectral_rolloff))

    tonnetz = librosa.feature.tonnetz(waveform, sr=sampling_rate).T
    x.update(zip(tonnetz_mean_fields, np.mean(tonnetz, axis=0)))
    x.update(zip(tonnetz_std_fields, np.std(tonnetz, axis=0)))

    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        y=waveform_harmonic, frame_length=n_fft, hop_length=hop_length).T
    x['zero_crossing_rate_mean'] = np.asscalar(
        np.mean(zero_crossing_rate, axis=0))
    x['zero_crossing_rate_std'] = np.asscalar(
        np.std(zero_crossing_rate, axis=0))
    x['zero_crossing_rate_skew'] = np.asscalar(
        sp.stats.skew(zero_crossing_rate))

    tempo, beats = librosa.beat.beat_track(
        y=waveform_harmonic, sr=sampling_rate, hop_length=hop_length)
    x['tempo'] = np.asscalar(tempo)
    x['beats_mean'] = np.asscalar(np.mean(beats, axis=0))
    x['beats_std'] = np.asscalar(np.std(beats, axis=0))

    return x, genre

def extract_features_gtzan(name='gtzan', datasets_path='C:\MLData', load_subset=False, load_object=True):
    if load_subset:
        name = "{}_subset".format(name)

    features_object_path = os.path.join(
        datasets_path, "{}.features".format(name))
    if load_object and os.path.isfile(features_object_path):
        print("==> Extracting features from dataset '{}', with results:".format(name))
        print("Saved features from dataset '{}' exists, loading...".format(name))
        X, y = pickle.load(open(features_object_path, 'rb'))
    else:
        datasets.fetch_dataset('gtzan', 'http://opihi.cs.uvic.ca/sound/genres.tar.gz')
        dataset = datasets.load_dataset_gtzan(load_subset=load_subset)

        print("==> Extracting features from dataset '{}', with results:".format(name))
        X = []
        y = []
        for sample in dataset:
            x, label = extract_features(sample)
            X.append(x)
            y.append(label)

        print("Saving features from dataset '{}' to '{}'...".format(
            name, features_object_path))
        pickle.dump((X, y), open(features_object_path, 'wb'))
        print("Saved features from dataset '{}'.".format(name))

    print("==> Extracted features from dataset '{}'.\n".format(name))
    return X, y

def load_data(n_mfcc=20, n_fft=2048, hop_length=512, n_bands=6):
    X, y = extract_features_gtzan(load_subset=True)
    return pd.DataFrame(X), pd.DataFrame(y)

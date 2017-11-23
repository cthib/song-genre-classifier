# Feature Extraction

import numpy as np
import librosa
from scipy import stats
import sys

def feature_stats(values):
  out = []
  out.extend(np.mean(values,axis = 1))
  out.extend(np.std(values,axis = 1))
  out.extend(stats.skew(values,axis = 1))
  out.extend(stats.kurtosis(values,axis = 1))
  out.extend(np.median(values,axis = 1))
  out.extend(np.min(values,axis = 1))
  out.extend(np.max(values,axis = 1))
  return out

def compute_features(audio):
  
  array = []

  x, samplerate = librosa.load(audio, sr=None, mono = True)

  #stft = short-term Fourier Transform, y -> input signal, hop_length -> # of audio frams between stft columns
  stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
  assert stft.shape[0] == 1 + 2048 // 2

  #S = spectrogram magnitude

  f = librosa.feature.spectral_centroid(S=stft)
  array.extend(feature_stats(f))

  f = librosa.feature.spectral_bandwidth(S=stft)
  array.extend(feature_stats(f))

  f = librosa.feature.spectral_rolloff(S=stft)
  array.extend(feature_stats(f))

  f = librosa.feature.rmse(S=stft)
  array.extend(feature_stats(f))

  f = librosa.feature.zero_crossing_rate(x, frame_length = 2048, hop_length = 512)
  array.extend(feature_stats(f))

  return array


def main():
  print('Starting feature extraction...')

  input_audio = sys.argv[1]

  feature_array = compute_features(input_audio)

  print(feature_array)

if __name__ == "__main__":
    main()


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

def compute_features(fn):
	print("Extracing features from downloaded song...")
	array = []
	x, samplerate = librosa.load(fn, sr=None, mono = True)

	# Short Fourier Transform, y -> input signal, hop_length -> # of audio frams between stft columns
	print("- Short Fourier Transform")
	stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
	assert stft.shape[0] == 1 + 2048 // 2

	# Audio features
	print("- Spectral Centroid")
	f = librosa.feature.spectral_centroid(S=stft)
	array.extend(feature_stats(f))

	print("- Spectral Bandwidth")
	f = librosa.feature.spectral_bandwidth(S=stft)
	array.extend(feature_stats(f))

	print("- Spectral Rolloff")
	f = librosa.feature.spectral_rolloff(S=stft)
	array.extend(feature_stats(f))

	print("- Root Mean Square Error (RMSE)")
	f = librosa.feature.rmse(S=stft)
	array.extend(feature_stats(f))

	print("- Zero Cross Rate (ZCR)")
	f = librosa.feature.zero_crossing_rate(x, frame_length = 2048, hop_length = 512)
	array.extend(feature_stats(f))

	return array


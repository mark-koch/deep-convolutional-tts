"""
Adapted from Kyubyong Park's code. See https://github.com/Kyubyong/dc_tts/blob/master/utils.py
"""

import librosa
import scipy
import copy
import os
import numpy as np
from config import Config


def extract_spectrogram(path):
    """
    Loads the audio file in 'path' and returns a corresponding normalized melspectrogram
    and a linear spectrogram.
    """
    y, _ = librosa.load(path, sr=Config.sample_rate)

    # Remove leading and trailing silence
    y, _ = librosa.effects.trim(y)

    # Preemphasis (upscale frequencies and downscale them later to reduce noise)
    y = np.append(y[0], y[1:] - Config.preemphasis*y[:-1])

    # Convert the waveform to a complex spectrogram by a short-time Fourier transform
    linear = librosa.stft(y=y, n_fft=Config.num_fft_samples, hop_length=Config.hop_length,
                          win_length=Config.window_length)

    # Only consider the magnitude of the spectrogram
    mag = np.abs(linear)

    # Compute the mel spectrogram
    mel_basis = librosa.filters.mel(Config.sample_rate, Config.num_fft_samples, Config.mel_size)
    mel = np.dot(mel_basis, mag)

    # To decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # Normalize
    mel = np.clip((mel - Config.ref_db + Config.max_db) / Config.max_db, 1e-8, 1)
    mag = np.clip((mag - Config.ref_db + Config.max_db) / Config.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)
    mag = mag.T.astype(np.float32)

    return mel, mag


def spectrogram2wav(mag):
    """ Generates wave file from linear magnitude spectrogram
    Args:
        mag: A numpy array of shape [T, 1 + num_fft_samples//2]
    Returns:
        wav: A 1-D numpy array.
    """
    # Transpose
    mag = mag.T

    # De-noramlize
    mag = (np.clip(mag, 0, 1) * Config.max_db) - Config.max_db + Config.ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag**Config.power)

    # De-preemphasis
    wav = scipy.signal.lfilter([1], [1, -Config.preemphasis], wav)

    # Remove leading and trailing silence
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def griffin_lim(spectrogram):
    """Applies Griffin-Lim's raw."""
    X_best = copy.deepcopy(spectrogram)
    for i in range(Config.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, Config.num_fft_samples, Config.hop_length, win_length=Config.window_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)
    return y


def invert_spectrogram(spectrogram):
    """ Applies inverse fft.
    Args:
        spectrogram: [1 + num_fft_samples//2, t]
    """
    return librosa.istft(spectrogram, Config.hop_length, win_length=Config.window_length, window="hann")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generates spectrograms from wav files.')
    parser.add_argument('-w', '--wav', dest='wav_path', required=False, default="wav",
                        help='Directory of the wav files')
    parser.add_argument('-m', '--mel', dest='mel_path', required=False, default="mel",
                        help='Directory for the mel spectrograms')
    parser.add_argument('-l', '--lin', dest='lin_path', required=False, default="lin",
                        help='Directory for the linear spectrograms')
    args = parser.parse_args()

    mels = []
    for file in os.listdir(args.wav_path):
        name, ext = os.path.splitext(file)
        if ext == ".wav":
            print("Processing " + file)
            mel, mag = extract_spectrogram(os.path.join(args.wav_path, name) + ".wav")
            mels.append(mel)
            np.save(os.path.join(args.mel_path, name), mel)
            np.save(os.path.join(args.lin_path, name), mag)
    np.save(os.path.join(args.mel_path, "all"), np.array(mels))
    print("Finished")

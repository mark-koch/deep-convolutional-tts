import inspect


class Config:
    # Note that when continuing to train or generating speech from a checkpoint, we will automatically load the original
    # configuration. If you wish to use different parameters, run the scripts with the flag '--cc'.
    # When training a new model, the current parameters in this file will be used.

    vocab = u"PE abcdefghijklmnopqrstuvwxyz'.?"             # P = Padding,  E = End of text
    vocab_padding_index = 0                                 # Index of the padding character
    vocab_end_of_text = "E"

    # Network dimensions
    e = 128
    d = 256
    c = 512
    F = 80          # Mel spectrogram size
    F_ = 1025       # STFT spectrogram size. The original paper uses 513 instead of 1025

    # Parameters for audio processing. Refer to https://github.com/Kyubyong/dc_tts
    sample_rate = 22050
    num_fft_samples = (F_ - 1) * 2
    frame_length = 0.05                                 # seconds, original paper uses ~11.6 ms approx 0.012 s
    frame_shift = frame_length / 4                      # seconds
    hop_length = int(sample_rate * frame_shift)
    window_length = int(sample_rate * frame_length)
    mel_size = F
    power = 1.5
    preemphasis = .97
    n_iter = 50
    max_db = 100
    ref_db = 20
    time_reduction = 4                                  # Only consider every n-th time step

    # The following parameters are only needed for training.

    # Maximum number of characters and mel frames. These will be used to ignore training samples, that are too long.
    # Can be tweaked to limit memory consumption.
    max_N, max_T = 180, 210
    g = 0.2                         # Guided attention
    dropout_rate = 0.05

    @staticmethod
    def get_config():
        attributes = inspect.getmembers(Config, lambda a: not (inspect.isroutine(a)))
        return [a for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))]

    @staticmethod
    def set_config(a):
        for name, value in a:
            setattr(Config, name, value)

# Deep Convolutional TTS [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-vePaCwBcLD2H-mGScYE6MnfN8A4uGoz#forceEdit=true&sandboxMode=true)

A PyTorch implementation of "[Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention](https://arxiv.org/abs/1710.08969)".

## Setup

Requirements:
* pytorch >= 1.3
* librosa
* scipy
* numpy
* matplotlib
* unidecode
* tqdm

Optional:
* simpleaudio and num2words, if you want to run ``realtime.py``
* nltk for better text processing


## Data

For audio preprocessing I mainly used [Kyubyong's DCTTS code](https://github.com/Kyubyong/dc_tts). I trained the model on the [LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/) 
and the german samples from the [CSS10 Dataset](https://github.com/Kyubyong/css10). You can find pretrained models below.

If you want want to train a model, you need to prepare your dataset:
1. Create a directory ``data`` for your dataset and a sub directory ``data/wav`` containing all your audio clips.
2. Run ``audio_processing.py -w data/wav -m data/mel -l data/lin``.
3. Create a text file ``data/lines.txt`` containing the transcription of the audio clips in the following format:

        my-wav-file-000|Transciption of file my-wav-file-000.wav
        my-wav-file-001|Transciption of file my-wav-file-001.wav
        ...
        
   Note that you don't need to remove umlauts or accents like ä, é, î, etc. This will be done automatically. If your transcipt 
   contains abbreviations or numbers on the other hand, you will need to spell them out. For spelling out numbers you can 
   install ``num2words`` and use ``spell_out_numbers`` from the script ``text_processing.py``.


## Training

After preparing the dataset you can start training the Text2Mel and SSRN networks. Run
* ``train_text2mel.py -d path/to/dataset``
* ``train_ssrn.py -d path/to/dataset``

By default, checkpoints will be saved every 10,000 steps, but you can also set ``-save_iter`` for a custom value.
If you want to continue training from a checkpoint, use ``-r save/checkpoint-xxxxx.pth``. 
For other options run ``train_text2mel.py -h``, ``train_ssrn.py -h`` and have a look at ``config.py``.


## Generate speech

There are two scripts for generating audio:

With ``realtime.py`` you can type sentences in the terminal and the computer will read it out aloud.
Run ``realtime.py --t2m text2mel-checkpoint.pth --ssrn ssrn-checkpoint.pth --lang en``.

With ``synthesize.py text.txt`` you can generate a wav file from a given text file. Run it with the following arguments:
* ``--t2m``, ``--ssrn``, ``-o``: paths to the saved networks and output file (optional)
* ``--max_N``: The text file will be split into chunks not longer than this length (optional). If not given, it will pick
    the value used for training in ``config.py``. Reducing this value might improve audio quality, but increases generating
    time for longer texts and introduces breaks in sentences.
* ``--max_T``: Number of mel frames to generate for each chunk (optional). If the endings of sentences are cut off, increase
    this value.  
* ``--lang``: Language of the text (optional). Defaults to ``en`` and will be used to spell out numbers occuring in the text.


## Samples
See [here](http://mark-koch.github.io/deep-convolutional-tts/index.html). All samples were generated with the models below.



## Pretrained models

Lanuage|Dataset|Text2Mel|SSRN|
|--|--|--|--|
English|LJ Speech|[350k steps](https://drive.google.com/open?id=12KvCJkID75Rgcg-Q_DLIwI-iHM_mHej4)|[350k steps](https://drive.google.com/open?id=1hcxs_zgPdAxAwtEsr4GDaPtZVgLDU5nh)|
German|CSS10|[150k steps](https://drive.google.com/open?id=15ZusRQiqK2HyagWDLgtVF6GLLKUK7iVB)|[100k steps](https://drive.google.com/open?id=17VyjKSMYFmIqArQr6yYcfl_NdYRqesL4)


## Notes

* I use layer norm, dropout and learning rate decay during training.
* The audio quality seems to deteriorate at the end of generated audio samples. A workaround would be to set a low value
    for ``--max_N`` to reduce the length for each sample.


## Acknowledgement

* The audio preprocessing uses [Kyubyong's DCTTS code](https://github.com/Kyubyong/dc_tts). This repo also helped me with
    some difficulties I had during the implementation.
* Also see this other [PyTorch implementation](https://github.com/chaiyujin/dctts-pytorch).


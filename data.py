import torch.utils.data
import torch.nn.functional as F
import os
import io
import text_processing
import numpy as np
from collections import OrderedDict
from random import shuffle
from config import Config


class TTSDataset(torch.utils.data.Dataset):
    """
    Text to speech dataset.

    Args:
        text_path: Path to the text file containing the lines.
        mel_dir: Directory with all the mel spectrograms.
        lin_dir: Directory with all the linear spectrograms. Set 'None' for the text2mel training, because only mel
            spectrograms are needed there.
    """
    def __init__(self, text_path, mel_dir, lin_dir, data_in_memory=True):
        self.data = []
        self.data_in_memory = data_in_memory
        if data_in_memory and os.path.exists(os.path.join(mel_dir, "all.npy")):
            mels = np.load(os.path.join(mel_dir, "all.npy"), allow_pickle=True)
        else:
            mels = None
        with io.open(text_path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.split("|", maxsplit=1)
                text = line[1]
                text = text_processing.normalize(text)
                text = text + Config.vocab_end_of_text
                # Skip if text is too long
                if len(text) > Config.max_N:
                    print("Warning: Text with id '{}' is too long! Line will be skipped".format(line[0]))
                    continue
                text = text_processing.vocab_lookup(text)
                text = torch.tensor(text, dtype=torch.long)
                if data_in_memory:
                    if mels is not None:
                        mel = mels[i]
                        t = mel.shape[0]  # Needed for lin padding
                        mel = self._process_mel(mel)
                    else:
                        mel_path = os.path.join(mel_dir, line[0]) + ".npy"
                        mel = np.load(mel_path)
                        t = mel.shape[0]  # Needed for lin padding
                        mel = self._process_mel(np.load(mel_path))
                    # Skip if mel is too long
                    if mel.shape[0] > Config.max_T:
                        print("Warning: Mel with id '{}' is too long! Line will be skipped".format(line[0]))
                        continue
                    self.data.append({"name": line[0], "text": text, "mel": mel, "t": t})
                else:
                    self.data.append({"name": line[0], "text": text})
        self.mel_dir = mel_dir
        self.lin_dir = lin_dir

    @staticmethod
    def _process_mel(mel):
        mel = torch.tensor(mel)
        t = mel.shape[0]
        # Marginal padding for reduction shape sync. Needed for SSRN training to make up-sampling from T to 4T match up
        # with lin size
        num_paddings = Config.time_reduction - (t % Config.time_reduction) if t % Config.time_reduction != 0 else 0
        mel = F.pad(input=mel, pad=[0, 0, 0, num_paddings], mode='constant', value=0)
        # Time reduction
        mel = mel[::Config.time_reduction]
        return mel

    @staticmethod
    def _process_lin(lin, t):
        lin = torch.tensor(lin)
        # Marginal padding for reduction shape sync. Needed for SSRN training to make lin match up with up-sampled mel
        # from T to 4T.
        num_paddings = Config.time_reduction - (t % Config.time_reduction) if t % Config.time_reduction != 0 else 0
        _lin = F.pad(input=lin, pad=[0, 0, 0, num_paddings], mode='constant', value=0)
        # print(lin.shape[0], t, num_paddings, _lin.shape[0])
        return _lin

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        text = self.data[idx]["text"]  # Already converted to tensor

        if self.data_in_memory:
            mel = self.data[idx]["mel"]
            t = self.data[idx]["t"]
        else:
            mel_path = os.path.join(self.mel_dir, self.data[idx]["name"]) + ".npy"
            mel = np.load(mel_path)
            t = mel.shape[0]
            mel = self._process_mel(mel)

        if self.lin_dir is not None:
            lin_path = os.path.join(self.lin_dir, self.data[idx]["name"]) + ".npy"
            lin = self._process_lin(np.load(lin_path), t)
            return {"mel": mel, "lin": lin, "text": text}

        else:
            return {"mel": mel, "text": self.data[idx]["text"]}


def collate_fn(data):
    """
    Creates mini-batch tensors from the list maps.

    Args:
        data: List of string maps containing the training data ("text", "mel", "lin").
    """
    texts = [d["text"] for d in data]
    mels = [d["mel"] for d in data]
    max_text_len = max(text.shape[0] for text in texts)
    max_mel_len = max(mel.shape[0] for mel in mels)

    # Prepare zero padding for texts, mels and lins
    mel_pads = torch.zeros(len(mels), max_mel_len, mels[0].shape[-1])
    text_tensor = torch.zeros(len(texts), max_text_len, dtype=torch.long)

    if "lin" in data[0]:
        lins = [d["lin"] for d in data]
        max_lin_len = max(lin.shape[0] for lin in lins)
        lin_pads = torch.zeros(len(lins), max_lin_len, lins[0].shape[-1])
        for i in range(len(texts)):
            mel_pads[i, :mels[i].shape[0]] = mels[i]
            text_tensor[i, :texts[i].shape[0]] = texts[i]
            lin_pads[i, :lins[i].shape[0]] = lins[i]
        return {"text": text_tensor, "mel": mel_pads, "lin": lin_pads}
    else:
        for i in range(len(texts)):
            mel_pads[i, :mels[i].shape[0]] = mels[i]
            text_tensor[i, :texts[i].shape[0]] = texts[i]
        return {"text": text_tensor, "mel": mel_pads}


class BucketBatchSampler(torch.utils.data.Sampler):
    """
    Groups inputs into buckets of equal length and samples batches out of these buckets. This way all inputs in a batch
    will have the same size and no padding is needed.
    Adapted from https://discuss.pytorch.org/t/tensorflow-esque-bucket-by-sequence-length/41284

    Args:
        inputs: A 1d array containing the feature that should be used for bucketing (indices in the same order as in the
            data set). In our case we want to bucket text sizes, so 'inputs' will be the list of all texts.

        batch_size: Maximum batch size (note that some batches can be smaller if buckets are not large enough).

        bucket_boundaries: int list, increasing non-negative numbers. The edges of the buckets to use when bucketing
            tensors.  Two extra buckets are created, one for `input_length < bucket_boundaries[0]` and one for
            `input_length >= bucket_boundaries[-1]`.
    """
    def __init__(self, inputs, batch_size, bucket_boundaries):
        self.batch_size = batch_size
        # Add bucket for smaller and larger inputs
        self.bucket_boundaries = [-1] + bucket_boundaries + []
        ind_n_len = []
        for i, p in enumerate(inputs):
            ind_n_len.append((i, p.shape[0]))
        self.ind_n_len = ind_n_len
        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)

    def _generate_batch_map(self):
        # Shuffle all of the indices first so they are put into buckets differently
        shuffle(self.ind_n_len)
        # Organize lengths, e.g., batch_map[10] = [30, 124, 203, ...] <= indices of sequences of length 10
        batch_map = OrderedDict()
        for idx, length in self.ind_n_len:
            # Find corresponding bucket
            if length < self.bucket_boundaries[0]:
                bucket = -1
            elif length > self.bucket_boundaries[-1]:
                bucket = self.bucket_boundaries[-1] + 1
            else:
                for i in range(len(self.bucket_boundaries)):
                    if length == self.bucket_boundaries[i]:
                        bucket = i
                        break
                    if length < self.bucket_boundaries[i]:
                        bucket = i - 1
                        break
            # Save index in bucket
            if bucket not in batch_map:
                batch_map[bucket] = [idx]
            else:
                batch_map[bucket].append(idx)
        # Use batch_map to split indices into batches of equal size
        # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
        batch_list = []
        for length, indices in batch_map.items():
            for group in [indices[i:(i + self.batch_size)] for i in range(0, len(indices), self.batch_size)]:
                batch_list.append(group)
        return batch_list

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return len(self.ind_n_len)

    def __iter__(self):
        self.batch_list = self._generate_batch_map()
        # shuffle all the batches so they arent ordered by bucket size
        shuffle(self.batch_list)
        for i in self.batch_list:
            yield i


class _RepeatSampler(object):
    """ Sampler that repeats forever. """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.dataloader.DataLoader):
    """
    DataLoader that pretends, that the epoch never ends. This improves performance on windows, because the process
    spawning at the start of each epoch will be avoided. See https://github.com/pytorch/pytorch/issues/15849
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

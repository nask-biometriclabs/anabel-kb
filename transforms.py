from enum import Enum
import torch
import numpy as np


class RandomSubstring():
    def __init__(self, minpart=0.6):
        self.minpart = minpart

    def __call__(self, seq):
        seq_len = seq.shape[0]
        if seq_len > 3:
            rand_seq_len = torch.randint(int(self.minpart * seq_len), seq_len, size=(1,)).item()
            start = torch.randint(0, seq_len - rand_seq_len + 1, size=(1,)).item()
            return seq[start:start + rand_seq_len, :]
        else:
            return seq


class KVCTransform(object):
    """
    Transform for KVC dataset. Returns the Key value, Hold Latency (HL)
    Press Latency (PL), Inter-key Latency (IL) and Release Latency (RL)
    """

    def __init__(self, seq_len, time_diff_scale=1/1E4, hold_time_scale=1/1E3, ascii_scale=1/256):
        self.seq_len = seq_len
        self.time_diff_scale = time_diff_scale
        self.hold_time_scale = hold_time_scale
        self.ascii_scale = ascii_scale

    def __call__(self, seq):
        key = np.reshape(seq[:, 2]/256, newshape=(np.shape(seq)[0], 1))
        key_next = np.concatenate((key[1:], [[0]]), axis=0)

        # Hold Latency (HL)
        hl = np.reshape((seq[:, 1] - seq[:, 0]) * self.hold_time_scale, newshape=(np.shape(seq)[0], 1))

        # Press Latency (PL)
        pl = np.zeros((np.shape(seq)[0], 1))
        pl_ = np.reshape([(seq[i + 1, 0] - seq[i, 0]) * self.time_diff_scale for i in range(len(seq[:, 0]) - 1)],
                         newshape=(np.shape(pl)[0] - 1, 1))
        pl[:len(pl_)] = pl_

        # Inter-key Latency (IL)
        il = np.zeros((np.shape(seq)[0], 1))
        il_ = np.reshape([(seq[i + 1, 0] - seq[i, 1]) * self.time_diff_scale for i in range(len(seq[:, 0]) - 1)],
                         newshape=(np.shape(pl)[0] - 1, 1))
        il[:len(il_)] = il_

        # Release Latency (RL)
        rl = np.zeros((np.shape(seq)[0], 1))
        rl_ = np.reshape([(seq[i + 1, 1] - seq[i, 1]) * self.time_diff_scale for i in range(len(seq[:, 0]) - 1)],
                         newshape=(np.shape(pl)[0] - 1, 1))
        rl[:len(rl_)] = rl_

        output = np.concatenate((key, key_next, hl, pl, il, rl), axis=1)

        if self.seq_len is not None:
            output = np.concatenate((output, np.zeros((self.seq_len, 6))))[:self.seq_len]
        return output


class NumpyToTensor(object):
    def __call__(self, x) -> torch.Tensor:
        return torch.Tensor(x)


class FeatureSet(Enum):
    ONLY_KEYS = 0
    ALL_LATENCIES = 1
    HOLD_LATENCY = 2
    INTERKEY_LATENCY = 3
    PRESSPRESS_LATENCY = 4
    RELEASERELEASE_LATENCY = 5
    HOLD_AND_INTERKEY = 6
    HOLD_AND_PRESS = 7
    KEY_HOLD_AND_INTERKEY = 8
    KEYS_HOLD_AND_INTERKEY = 9


class SelectFeatures:
    def __init__(self, ftset=FeatureSet.HOLD_AND_INTERKEY):
        self.ftset = ftset

    def __call__(self, seq):
        if self.ftset == FeatureSet.ALL_LATENCIES:
            return seq[:, 2:]
        elif self.ftset == FeatureSet.HOLD_AND_PRESS:
            return seq[:, 2:4]
        elif self.ftset == FeatureSet.HOLD_AND_INTERKEY:
            return seq[:, [2, 4]]
        elif self.ftset == FeatureSet.KEY_HOLD_AND_INTERKEY:
            return seq[:, [0, 2, 4]]
        elif self.ftset == FeatureSet.KEYS_HOLD_AND_INTERKEY:
            return seq[:, [0, 1, 2, 4]]
        else:
            return seq[:, self.ftset.value]


def featureset2channels(featureset):
    if featureset == FeatureSet.ALL_LATENCIES:
        return 5
    elif featureset == FeatureSet.HOLD_AND_PRESS:
        return 2
    elif featureset == FeatureSet.HOLD_AND_INTERKEY:
        return 2
    elif featureset == FeatureSet.KEY_HOLD_AND_INTERKEY:
        return 3
    elif featureset == FeatureSet.KEYS_HOLD_AND_INTERKEY:
        return 4
    else:
        return 1

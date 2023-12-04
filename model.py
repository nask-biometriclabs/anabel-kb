import transforms
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention


class AnabelKa(nn.Module):
    def __init__(self, n_classes=0, feature_set=transforms.FeatureSet.KEY_HOLD_AND_INTERKEY, seq_len=100,
                 n_kernels=10000, representation_size=256):
        super(AnabelKa, self).__init__()
        self.feature_set = feature_set
        self.n_channels = transforms.featureset2channels(self.feature_set) - 2
        self.seq_len = seq_len
        self.n_kernels = n_kernels
        conv1 = nn.Conv1d(self.n_channels, n_kernels, kernel_size=3, stride=1, groups=self.n_channels)
        bn1 = nn.BatchNorm1d(n_kernels)
        relu1 = nn.ReLU(inplace=True)
        self.extractor = nn.Sequential(conv1, bn1, relu1)
        conv2 = nn.Conv1d(in_channels=1, out_channels=n_kernels, kernel_size=3, stride=1)
        bn2 = nn.BatchNorm1d(n_kernels)
        relu2 = nn.ReLU(inplace=True)
        self.key_extractor = nn.Sequential(conv2, bn2, relu2)
        self.token_len = n_kernels * (seq_len - 1)
        self.merger = MultiheadAttention(n_kernels, num_heads=8, batch_first=True)
        self.n_features = n_kernels * (seq_len - 2)
        self.representation_size = representation_size
        self.n_classes = n_classes
        self.encoder, self.classifier = self.init_classifier(self.n_features, self.n_classes, self.representation_size)

    @staticmethod
    def init_classifier(n_features, n_classes, representation_size):
        layers = []
        layers += [nn.BatchNorm1d(n_features)]
        linear = nn.Linear(n_features, representation_size)
        nn.init.constant_(linear.weight.data, 0)
        nn.init.constant_(linear.bias.data, 0)
        layers += [linear]
        layers += [nn.LayerNorm(representation_size)]
        encoder = nn.Sequential(*layers)
        clinear = nn.Linear(in_features=representation_size, out_features=n_classes)
        classifier = nn.Sequential(clinear)
        return encoder, classifier

    def forward(self, keys, x, orig_seq_len=None):
        y = self.encode(keys, x, orig_seq_len)
        y = self.classifier(y)
        return y

    def enroll(self, keys, x, orig_seq_len=None):
        y = self.encode(keys, x, orig_seq_len)
        return y

    def encode(self, keys, x, orig_seq_len=None):
        device = x.device
        if len(x.shape) < 3:
            x = x.unsqueeze(-1)
            keys = keys.unsqueeze(-1)
        x = x.permute(0, 2, 1)
        keys = keys.permute(0, 2, 1)
        keys = self.key_extractor(keys)
        y = self.extractor(x)
        mask = torch.zeros((y.shape[0], y.shape[-1]), dtype=torch.bool).to(device)
        if orig_seq_len is not None:
            for k, osq in enumerate(orig_seq_len):
                mask[k, osq:] = True
        keys = keys.permute(0, 2, 1)
        y = y.permute(0, 2, 1)
        y, _ = self.merger(keys, keys, y, key_padding_mask=mask)
        y = y.reshape(y.shape[0], -1)
        y = self.encoder(y)
        return y

    def save(self, filename, epoch=None):
        torch.save({
            'epoch': epoch,
            'state_dict': self.state_dict(),
            'n_classes': self.n_classes,
            'n_channels': self.n_channels,
            'seq_len': self.seq_len,
            'n_kernels': self.n_kernels,
            'feature_set': self.feature_set,
            'representation_size': self.representation_size,
            'n_features': self.n_features
        }, filename)


def create_model(n_classes, feature_set, seq_len, n_kernels, representation_size):
    model = AnabelKa(n_classes, feature_set, seq_len, n_kernels, representation_size)
    return model


def load_model(filename):

    import pickle
    with open(filename, "rb") as tf:
        chdict = pickle.load(tf)
        fs = transforms.FeatureSet[chdict['featureset']]
        model = AnabelKa(chdict['n_classes'], fs, chdict['seqlenght'],
                         chdict['n_kernels'], chdict['representation_size'])
        src_dict = chdict['state_dict']
        model.load_state_dict(src_dict)
    return model

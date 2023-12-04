import torch

from model import load_model
import transforms
import numpy as np

def preprocess(x):
    seqlen = torch.IntTensor([len(x)])
    tlist = [transforms.KVCTransform(seq_len=66),
                          transforms.SelectFeatures(transforms.FeatureSet.KEYS_HOLD_AND_INTERKEY),
                          transforms.NumpyToTensor()]
    for t in tlist:
        x = t(x)
    keys = x[:, :1].unsqueeze(0)
    times = x[:, 2:].unsqueeze(0)
    return keys,times,seqlen

if __name__ == '__main__':

    checkpoint =  "data/checkpoints/AnabelKA_desktop_2023-10-29-224950-nfix.pkl"
    data_path= "data/desktop/desktop_test_sessions.npy"

    data = np.load(data_path, allow_pickle=True).item()




    model = load_model(checkpoint)
    model.eval()

    session_id, sample = next(iter(data.items()))


    keys,x,seqlen = preprocess(sample)
    embedding = model.enroll(keys,x, orig_seq_len=seqlen)


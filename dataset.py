from torch.utils.data import Dataset
import numpy as np
import os


class KVCDataset(Dataset):
    def __init__(self, filename, fold_file=None, transform=None):
        self.filename = filename
        self.fold_file = fold_file
        self.transform = transform
        self.samples, self.users, self.user_indexes, self.session_ids, self.orig_seq_lens = self.load_samples(filename)

    def get_nclasses(self):
        n_classes = len(np.unique(self.users))
        return n_classes

    def load_samples(self, filename):
        samples = []
        user_ids = []
        user_indexes = []
        session_ids = []
        orig_seq_lens = []
        if filename.endswith('.npy'):
            data = np.load(filename, allow_pickle=True).item()
            if self.fold_file is not None:
                userids = []
                with open(self.fold_file, 'r') as fp:
                    for line in fp.readlines():
                        userids.append(line.strip())
            else:
                userids = data.keys()
            for user_idx, user_id in enumerate(set(userids)):
                sessions = data[user_id]

                for session_id, session_data in sessions.items():
                    user_ids.append(user_id)
                    user_indexes.append(user_idx)
                    session_ids.append(session_id)
                    samples.append(session_data)
                    orig_seq_lens.append(len(session_data))
        else:
            raise NotImplementedError(
                'Data loader for data type of {} is not implemented'.format(os.path.splitext(filename)[-1]))
        print('Session data loaded - avg seq lenght = {} max = {} min = {} median = {}'.format(np.mean(orig_seq_lens),
                                                                                               np.max(orig_seq_lens),
                                                                                               np.min(orig_seq_lens),
                                                                                               np.median(
                                                                                                   orig_seq_lens)))
        return samples, user_ids, user_indexes, session_ids, orig_seq_lens

    def __getitem__(self, index):
        element = {}
        sample = self.samples[index]
        if self.users is not None:
            element['user'] = self.users[index]
        if self.transform is not None:
            for transformation in self.transform:
                sample = transformation(sample)
        element['data'] = sample
        element['user_index'] = self.user_indexes[index]
        element['session_id'] =  self.session_ids[index]
        element['orig_seq_len'] =  self.orig_seq_lens[index]
        return element

    def __len__(self):
        return len(self.samples)

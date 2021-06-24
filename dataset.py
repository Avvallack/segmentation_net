import re
import numpy as np
import pandas as pd
import torch
import torch.utils.data as dt
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict


class SegmentationDataset(dt.Dataset):
    def __init__(self, query, conn, user_col='halo_id', target_col='first_seg', text_col='full_text', min_frequency=0):
        self.df = pd.read_sql(query, conn)
        self.user_col = user_col
        self.target_col = target_col
        self.text_col = text_col
        self.df['split_segs'] = self.df['segment_id'].apply(self.__get_cats)
        self.df[self.target_col] = self.df['split_segs'].apply(lambda x: x[0])
        self.df[self.text_col] = self.df['partner_site_host'] + self.df['partner_site_path'] + self.df['visit_referrer']
        self.df[self.text_col] = self.df[self.text_col].apply(self.__process_site_list)
        self.df['lens'] = self.df[self.text_col].apply(lambda x: len(x))
        self.df = self.df.copy().query('lens > 0')
        self.__build_vocab(min_frequency)
        self.df[self.text_col + '_idx'] = self.df[self.text_col].apply(
            lambda x: [self.clean_vocabulary[token] for token in x])
        self.df[self.user_col + '_idx'] = self.df[self.user_col].apply(lambda x: self.clean_vocabulary[x])
        self.df[self.target_col + '_idx'] = self.df[self.target_col].apply(lambda x: self.clean_vocabulary[x])
        self.texts = self.df[self.text_col + '_idx'].values
        self.users = self.df[self.user_col + '_idx'].values
        self.target = self.df[self.target_col + '_idx'].values
        self.classes_idx = self.df[self.target_col + '_idx'].unique()
        del self.df

    @staticmethod
    def __get_cats(x):
        return x[16:].split('_GT_')

    @staticmethod
    def __process_site_list(site_text):
        return re.findall(r"[\w']+", site_text)

    def __build_vocab(self, min_frequency=0):
        tokens = []
        target_tokens = [token for token_list in self.df[self.text_col] for token in token_list]
        tokens += target_tokens
        tokens += list(set(self.df[self.target_col].values))
        tokens += list(set(self.df[self.user_col].values))
        vocabulary = defaultdict(int)
        for token in tokens:
            vocabulary[token] += 1
        self.clean_vocabulary = defaultdict(int)
        self.clean_vocabulary['PAD'] = 0
        for key, value in vocabulary.items():
            if value > min_frequency and len(key) > 1:
                self.clean_vocabulary[key] = len(self.clean_vocabulary) + 1
        self.back_vocab = {value: key for key, value in self.clean_vocabulary.items()}

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        return (torch.from_numpy(np.array(self.texts[index])),
                torch.from_numpy(np.array([self.users[index]])),
                torch.from_numpy(np.array([self.target[index]]))
                )


def padding(data):
    text, user, target = zip(*data)
    text = pad_sequence(text, batch_first=True)

    return text, torch.cat(user, 0), torch.cat(target, 0)


def padded_data_loader(data, workers, batch_size=32):
    return dt.DataLoader(dataset=data, batch_size=batch_size, collate_fn=padding, num_workers=workers)


if __name__ == '__main__':
    import pickle
    from data_load import query, conn

    dataset = SegmentationDataset(query, conn)
    with open('dataset.pkl', 'wb') as out_file:
        pickle.dump(dataset, out_file)


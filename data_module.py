import torch
import pickle
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split, DataLoader
from argparse import ArgumentParser


class SegmentationDataModule(pl.LightningDataModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default="")
        return parser

    def __init__(self, batch_size=32, workers=8, data_dir=''):
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.data_dir = data_dir
        with open(self.data_dir + 'dataset.pkl', 'rb') as f:
            self.dataset = pickle.load(f)
        self.input_dim = len(self.dataset.clean_vocabulary)
        self.classes_idx = self.dataset.classes_idx

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_set, self.test_set = random_split(self.dataset, [592220, 148351],
                                                         generator=torch.Generator().manual_seed(42))

            self.dims = tuple(self.train_set[0][0].shape)

    @staticmethod
    def padding(data):
        text, user, target = zip(*data)
        text = pad_sequence(text, batch_first=True)

        return text, torch.cat(user, 0), torch.cat(target, 0)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, collate_fn=self.padding,
                          num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, collate_fn=self.padding,
                          num_workers=self.workers)


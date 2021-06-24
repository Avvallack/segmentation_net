from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics


class TripletEmbeddings(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--adam_eps", type=float, default=1e-7)
        parser.add_argument("--adam_beta1", type=float, default=0.9)
        parser.add_argument("--adam_beta2", type=float, default=0.99)
        parser.add_argument("--embedding_size", type=int, default=300)
        return parser

    def __init__(self, vocab_size, classes_idx, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size
        self.num_classes = len(classes_idx)
        self.classes_idx = torch.from_numpy(classes_idx)
        self.back_classes_dict = {v: k for k, v in enumerate(classes_idx)}
        self.embeddings = nn.Embedding(self.vocab_size, self.hparams.embedding_size, padding_idx=0)

        self.f1_metric = torchmetrics.FBeta(num_classes=self.num_classes, beta=1, average='weighted')
        self.recall = torchmetrics.Recall(self.num_classes, average='weighted')

    def forward(self, text, user):
        self.emb = self.embeddings(text).mean(axis=1)
        self.user = self.embeddings(user)
        return self.emb * self.user

    def training_step(self, batch, batch_idx):
        texts, users, tags = batch
        outputs = self.forward(texts, users)
        targets = self.embeddings(tags)
        wrong_targets = targets[:, torch.randperm(targets.size()[1])]

        loss = F.triplet_margin_loss(outputs, targets, wrong_targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, data_batch, batch_nb):
        texts, users, tags = data_batch
        outputs = self.forward(texts, users)
        targets = self.embeddings(tags)
        wrong_targets = targets[:, torch.randperm(targets.size()[1])]

        loss_val = F.triplet_margin_loss(outputs, targets, wrong_targets)
        truth = self.get_categories_embeddins()
        prediction = torch.matmul(outputs, truth.transpose(0, 1)).argmax(1)
        simple_idx_tags = torch.tensor([self.back_classes_dict[i] for i in tags.numpy()])
        f1_score = self.f1_metric(simple_idx_tags, prediction)
        recall_score = self.recall(simple_idx_tags, prediction)
        similarity = nn.CosineSimilarity()
        cos = similarity(outputs, targets).mean()
        output = OrderedDict({
            'val_loss': loss_val,
            'val_f1': f1_score,
            'val_recall': recall_score,
            'val_similarity': cos
        })
        self.log_dict(output)
        return output

    def get_categories_embeddins(self):
        return self.embeddings(self.classes_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.hparams.learning_rate,
                                      betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
                                      eps=self.hparams.adam_eps,
                                      weight_decay=self.hparams.weight_decay,
                                      )
        return optimizer

    def return_embeddins(self):
        return self.embeddings


if __name__ == '__main__':
    import pickle
    import torch.utils.data as dt
    from dataset import SegmentationDataset, padded_data_loader
    from data_module import SegmentationDataModule

    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, default="triplets")
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=0)
    parser = TripletEmbeddings.add_model_specific_args(parser)
    parser = SegmentationDataModule.add_model_specific_args(parser)
    args = parser.parse_args()

    dm = SegmentationDataModule(batch_size=args.batch_size, workers=args.workers, data_dir=args.data_dir)

    model = TripletEmbeddings(dm.input_dim, dm.classes_idx, **vars(args))
    logger = pl.loggers.TensorBoardLogger(save_dir='lightning_logs', name='triplets')
    early_stop = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_f1',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='max'
    )

    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=[early_stop])
    trainer.fit(model, dm)


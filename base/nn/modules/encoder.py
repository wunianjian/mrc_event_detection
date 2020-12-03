import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import pytorch_lightning as pl

import base.data.dataset as dataset
from torch.utils.data import DataLoader, Sampler
from base.data.processor import *
from base.nn.utils import *
import argparse
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


class EntityLMEncoder:
    def __init__(self, mode, encoder=None, embedding=None, entity_tokens=None, nhid=None):
        """[summary]

        Args:
            encoder ([type]): [description]
            embedding ([type]): [description]
            entity_tokens ([type]): [description]
            mode ([type]): [description]
        """
        assert type(entity_tokens) is torch.tensor and len(entity_tokens.shape) == 2
        assert mode in ['encoder', 'pretrained', 'random']
        if mode == 'encoder':
            assert callable(encoder)
        elif mode == 'pretrained':
            assert callable(encoder)
            with torch.no_grad():
                self.embedding = nn.Embedding(len(entity_tokens), nhid)
                self.embedding.weight.data.copy_(encoder(entity_tokens)[:, 0, :].detach())
        elif mode == 'random':
            assert type(nhid) is int
            self.embedding = nn.Embedding(len(entity_tokens), nhid)
        else:
            self.embedding = embedding
        self.mode = mode
        self.encoder = encoder
        self.entity_tokens = entity_tokens

    def __encoding_by_encoder(self):
        return self.encoder(entity_tokens)[:, 0, :]

    def __encoding_by_embedding(self):
        return self.embedding.weight

    @property
    def encoding(self):
        if self.mode == 'encoder':
            return self.__encoding_by_encoder()
        elif self.mode in ['pretrained', 'random']:
            return self.embedding.weight

    def parameters(self):
        if self.mode == 'encoder':
            # if mode is encoder, 
            # the encoder should be optimized somewhere else,
            # by its own agenda.
            return []
        else:
            return self.embedding.parameters()

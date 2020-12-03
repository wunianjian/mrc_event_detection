from abc import ABC, abstractmethod
from collections.abc import Iterable
from argparse import Namespace

import seqeval.metrics as metrics
import sklearn.metrics as sk_metrics
import torch
import numpy
from IPython import embed

def get_last_valid_column(masks):
    """ Return the index of the column after which, inclusively,
    all columns are all False (invalid).
    """
    has_valid_token = torch.sum(masks, 0) > 0
    ridx = masks.shape[1] - 1
    while ridx >= 0:
        if not has_valid_token[ridx]:
            ridx -= 1
            continue
        break
    return ridx + 1

class BatcherMixin:
    def __init__(self):
        self.batch_states = {'train': None, 'val': None, 'test': None}

    @abstractmethod
    def batch_states_values(self):
        raise NotImplementedError

    @abstractmethod
    def sequence_mask_names(self):
        raise NotImplementedError

    @abstractmethod
    def process_batch(self):
        raise NotImplementedError

class SequenceBatcherMixin(BatcherMixin):
    def __init__(self):
        self.batch_states = {'train': None, 'val': None, 'test': None}

    def process_batch(self, batch, mode='train'):
        batch_states = Namespace()
        for name, value in zip(self.batch_states_values(), batch):
            setattr(batch_states, name, value)
        return batch_states


class SequenceTruncateMixin(BatcherMixin):
    def __init__(self):
        self.batch_states = {'train': None, 'val': None, 'test': None}

    def process_batch(self, batch, mode='train'):
        sequence_mask_names = self.sequence_mask_names()
        if isinstance(sequence_mask_names, list):
            sequence_names, mask_names = sequence_mask_names
        elif isinstance(sequence_mask_names, dict):
            sequence_names, mask_names = [
                sequence_mask_names['sequence'],
                sequence_mask_names['mask']
            ]
        else:
            raise TypeError("sequence_mask_names should be a list or a dict.")

        self.batch_states[mode] = Namespace()
        batch_states = self.batch_states[mode]
        for name, value in zip(self.batch_states_values(), batch):
            setattr(batch_states, name, value)

        masks = [getattr(batch_states, name) for name in mask_names]
        max_valid_idx = max(map(get_last_valid_column, masks))
        for name in sequence_names:
            setattr(batch_states, name, 
                getattr(batch_states, name)[:, :max_valid_idx]
            )
        for name in mask_names:
            setattr(batch_states, name, 
                getattr(batch_states, name)[:, :max_valid_idx]
            )

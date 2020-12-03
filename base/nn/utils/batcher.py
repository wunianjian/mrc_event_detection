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

class SequenceTruncateMixin(BatcherMixin):

    def process_batch(self, batch, val_names, excepts=[], mask_names=['attention_mask']):

        batch = {
            x: y for x, y in zip(val_names, batch)
        }

        masks = [batch[x] for x in mask_names]
        max_valid_idx = max(map(get_last_valid_column, masks))

        for x, t in batch.items():
            if x in excepts:
                continue
            if isinstance(t, torch.Tensor) and len(t.shape) == 2:
                batch[x] = t[:, :max_valid_idx]
        return batch
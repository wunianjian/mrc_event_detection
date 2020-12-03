import seqeval.metrics as metrics
import torch
import numpy

__function_map = {
    'f1': 'f1_score',
    'recall': 'recall_score',
    'precision': 'precision_score'
}

def filter_sequences_by_mask(seqs, masks):
    #seqs = numpy.array(seqs)
    #masks = numpy.array(masks).astype(bool)
    def __filter(seq, mask):
        return [x for x, y in zip(seq, mask) if y]
    return [__filter(seq, mask) for seq, mask in zip(seqs, masks)]

def masked_score(pred, label, masks, metric):
    pred = filter_sequences_by_mask(pred, masks)
    label = filter_sequences_by_mask(label, masks)
    return getattr(metrics, __function_map[metric])(label, pred)

def tensor_to_list(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, numpy.ndarray):
        return x.tolist()
    elif isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy().tolist()
    else:
        raise TypeError("Unknown type {} to convert to list.".format(
            type(x)
        ))

class AccumEvaluator:
    def restart(self):
        self.preds = []
        self.labels = []
        self.masks = []
        self.niters = 0

    def __init__(self):
        self.restart()

    def accumulate(self, preds, labels, masks):
        """
        add samples to accumulators. 
        preds, labels, masks should have the same length.
        """
        assert len(preds) == len(labels) == len(masks)
        self.preds += tensor_to_list(preds)
        self.labels += tensor_to_list(labels)
        self.masks += tensor_to_list(masks)
        self.niters += 1

    def metric(self, metric, n_last_samples=-1):
        """
        Arguments:
            metric: 
                one of 'precision', 'recall', 'f1_score'. 
            n_last_samples: 
                the number of samples used to compute the score.
                the last n_last_samples will be used.
                if n_last_samples <= 0, all samples will be used. 
        """
        if self.niters == 0:
            return 0.0
        if n_last_samples <= 0:
            n_last_samples = len(self.preds)
        return masked_score(
            *[x[-n_last_samples:] for x in [self.preds, self.labels, self.masks]], 
            metric=metric
        )
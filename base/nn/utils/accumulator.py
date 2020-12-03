from abc import ABC, abstractmethod
from collections.abc import Iterable
from argparse import Namespace

import seqeval.metrics as metrics
import sklearn.metrics as sk_metrics
import torch
import numpy
import json
import numpy as np

__function_map = {
    'f1': 'f1_score',
    'recall': 'recall_score',
    'precision': 'precision_score'
}

class reading_comprehension_metrics:
    @classmethod
    def tp(cls, labels, preds):
        labels = np.array(labels).astype(int)
        preds = np.array(preds).astype(int)
        return labels[preds==1].sum()

    @classmethod
    def fp(cls, labels, preds):
        labels = np.array(labels).astype(int)
        preds = np.array(preds).astype(int)
        return preds[labels==0].sum()

    @classmethod
    def tn(cls, labels, preds):
        labels = np.array(labels).astype(int)
        preds = np.array(preds).astype(int)
        return (1-preds)[labels==0].sum()

    @classmethod
    def fn(cls, labels, preds):
        labels = np.array(labels).astype(int)
        preds = np.array(preds).astype(int)
        return labels[preds==0].sum()

    @classmethod
    def precision_score(cls, labels, preds):
        tp = cls.tp(labels, preds)
        fp = cls.fp(labels, preds)
        if tp + fp == 0.0:
            return 0.0
        return tp / (tp + fp)

    @classmethod
    def recall_score(cls, labels, preds):
        tp = cls.tp(labels, preds)
        fn = cls.fn(labels, preds)
        if tp + fn == 0.0:
            return 0.0
        return tp / (tp + fn)

    @classmethod
    def f1_score(cls, labels, preds):
        p = cls.precision_score(labels, preds)
        r = cls.recall_score(labels, preds)
        if p+r == 0.0:
            return 0.0
        return 2 * (p * r) / (p + r)
        
def filter_sequences_by_mask(seqs, masks):
    # seqs = numpy.array(seqs)
    # masks = numpy.array(masks).astype(bool)
    def __filter(seq, mask):
        return [x for x, y in zip(seq, mask) if y]

    return [__filter(seq, mask) for seq, mask in zip(seqs, masks)]


def num2seq(seqs):
    num_map = {0: 'O', 1: 'I', 2: 'B'}
    return [[num_map[num] for num in seq] for seq in seqs]


def seq_score(pred, label, metric):
    '''
    try:
        if not isinstance(type(label[0][0]), str):
            pred = num2seq(pred)
            label = num2seq(label)
    except IndexError:
        pass
    '''
    return getattr(metrics, __function_map[metric])(label, pred)


def score(pred, label, metric, average='binary', **kwargs):
    return getattr(sk_metrics, __function_map[metric])(label, pred, average=average, **kwargs)

def binary_score(pred, label, metric, average=None):
    return getattr(reading_comprehension_metrics, __function_map[metric])(label, pred)

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

    def __init__(self, reverse_label_map=None, add_bio_scheme=False):
        self.restart()
        self.reverse_label_map = reverse_label_map
        self.add_bio_scheme = add_bio_scheme

    def add_bio(self, tags):
        inlabel = False
        tags_ = []
        for x in tags:
            if x == 'O' or x == 'Other':
                tags_.append(x)
                inlabel = False
            elif inlabel:
                tags_.append("I-" + x)
            else:
                tags_.append("B-" + x)
                inlabel = True
        return tags_

    def accumulate(self, preds, labels, masks):
        """
        add samples to accumulators. 
        preds, labels, masks should have the same length.
        """
        assert len(preds) == len(labels) == len(masks)
        if self.reverse_label_map is not None:
            rm = self.reverse_label_map
            preds, labels, masks  = [[[rm[int(x)] for x in y] for y in z] for z in [preds, labels, masks]]
        if self.add_bio_scheme:
            self.preds += list(map(self.add_bio, filter_sequences_by_mask(tensor_to_list(preds), masks)))
            self.labels += list(map(self.add_bio, filter_sequences_by_mask(tensor_to_list(labels), masks)))
        else:
            self.preds += filter_sequences_by_mask(tensor_to_list(preds), masks)
            self.labels += filter_sequences_by_mask(tensor_to_list(labels), masks)
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
        return seq_score(
            *[x[-n_last_samples:] for x in [self.preds, self.labels]],
            metric=metric
        )

class SimpleAccumEvaluator:
    def __init__(self, name=''):
        self.restart()
        self.name = name

    def restart(self):
        self.preds = []
        self.labels = []
        self.logits = []
        self.niters = 0

    def accumulate(self, preds, labels, logits=None, masks=None):
        """
        add samples to accumulators.
        preds, labels, masks should have the same length.
        """
        assert len(preds) == len(labels)
        if masks is not None:
            if masks.shape != preds.shape:
                masks = masks.unsqueeze(-1).expand_as(preds)
            preds = preds & masks
        preds = preds.flatten()
        labels = labels.flatten()
        self.preds += tensor_to_list(preds)
        self.labels += tensor_to_list(labels)
        if logits is not None:
            self.logits += tensor_to_list(logits)
        self.niters += 1

    def metric(self, metric, n_last_samples=-1, average='binary'):
        """
        Arguments:
            metric:
                one of 'precision', 'recall', 'f1_score'.
        """
        if self.niters == 0:
            return 0.0
        if n_last_samples <= 0:
            n_last_samples = len(self.preds)
        return score(self.preds[-n_last_samples:], self.labels[-n_last_samples:], metric, average)

    def save(self):
        assert self.logits is not None
        with open('%s_record.json' % self.name, 'w') as fp:
            json.dump([self.logits, self.labels], fp)

class BinaryAccumEvaluator:
    def __init__(self, name=''):
        self.restart()
        self.name = name

    def restart(self):
        self.preds = []
        self.labels = []
        self.logits = []
        self.niters = 0

    def accumulate(self, preds, labels, logits=None):
        """
        add samples to accumulators.
        preds, labels, masks should have the same length.
        """
        assert len(preds) == len(labels)
        self.preds += tensor_to_list(preds)
        self.labels += tensor_to_list(labels)
        if logits is not None:
            self.logits += tensor_to_list(logits)
        self.niters += 1

    def metric(self, metric, n_last_samples=-1, average='binary'):
        """
        Arguments:
            metric:
                one of 'precision', 'recall', 'f1_score'.
        """
        if self.niters == 0:
            return 0.0
        if n_last_samples <= 0:
            n_last_samples = len(self.preds)
        #return binary_score(self.preds[-n_last_samples:], self.labels[-n_last_samples:], metric, average)
        return score(self.preds[-n_last_samples:], self.labels[-n_last_samples:], metric, average, labels=[1])

    def save(self):
        assert self.logits is not None
        with open('%s_record.json' % self.name, 'w') as fp:
            json.dump([self.logits, self.labels], fp)


class TupleAccumulator:
    def __init__(self, data, event_map, argument_map):
        self.pred_tuples = set()
        self.label_tuples = set()
        self.data = data
        for si, sent in enumerate(data):
            for event_mention in sent['event-mentions']:
                event = event_map[event_mention['event_type']]
                for ei, argument in enumerate(event_mention['arguments']):
                    if argument != 'None':
                        self.label_tuples.add((si, event, ei, argument_map[argument]))

    def restart(self):
        self.pred_tuples = set()

    def accumulate(self, preds, tuples):
        preds = preds.cpu().detach()
        inst_num = preds.shape[0]
        tuples = tensor_to_list(tuples)
        if len(preds.shape) == 1:
            for row in range(inst_num):
                if preds[row] == 1:
                    self.pred_tuples.add(tuple(tuples[row]))
        elif len(preds.shape) == 2:
            for row in range(inst_num):
                for col in range(preds.shape[1]):
                    if preds[row, col] == 1:
                        si, event, argument = tuples[row]
                        self.pred_tuples.add((si, event, col, argument))

    def metric(self, metric):
        TP = len(self.pred_tuples & self.label_tuples)
        FP = len(self.pred_tuples - self.label_tuples)
        FN = len(self.label_tuples - self.pred_tuples)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        if metric == 'precision':
            return precision
        if metric == "recall":
            return recall
        if metric == "f1":
            if precision + recall != 0:
                return 2 * precision * recall / (precision + recall)
            else:
                return 0
        raise NotImplementedError()


class AnalyzeAccumulator:
    def __init__(self, mode):
        self.restart()
        self.mode = mode

    def restart(self):
        self.event_logits = []
        self.event_preds = []
        self.event_labels = []
        self.event_preds_with_conf = []

    def accumulate(self, event_logits, event_preds, event_labels, argument_confidence):
        event_logits = tensor_to_list(event_logits)
        event_preds = tensor_to_list(event_preds)
        event_labels = tensor_to_list(event_labels)
        self.event_logits.extend(event_logits)
        self.event_labels.extend(event_labels)
        argument_confidence = tensor_to_list(argument_confidence)
        index = 0
        for logit, pred, label in zip(event_logits, event_preds, event_labels):
            if pred == 1:
                self.event_preds_with_conf.append([logit, pred, label, argument_confidence[index]])
                index += 1
        assert index == len(argument_confidence)

    def save(self):
        with open('%s event_record.json' % self.mode, 'w') as fp:
            json.dump([self.event_logits, self.event_labels], fp)
        with open('%s event_with_arg_conf.json' % self.mode, 'w') as fp:
            json.dump(self.event_preds_with_conf, fp)

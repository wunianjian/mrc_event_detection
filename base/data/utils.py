import json
import transformers
from collections import Iterable
import copy
from nltk.tokenize import word_tokenize
import numpy as np
import unicodedata
import os

def apply_to_sequences(fn, sequence, **kwargs):
    """
    Apply fn to sequence if it is a single sequence, or
    each item inside sequence if it is a sequence of sequences.
    """
    assert isinstance(sequence, Iterable), "Sequence itself should be at least Iterable."
    if isinstance(sequence[0], Iterable) and type(sequence) is not str:
        return list(map(lambda x: fn(x, **kwargs), sequence))
    else:
        return fn(sequence, **kwargs)

def extract_property(data, *keywords):
    for key in keywords:
        data = [x[key] for x in data]
    return data

def masks_from_lens(lens, max_len):
    masks = np.stack([np.arange(max_len) < x for x in lens])
    return masks

def debio(sequence):
    """
    Strip a BIO-tagging sequence of the BIO prefix.

    Args:
        sequence ([type]): sequence of BIO tags.

    Returns:
        List: sequence without the BIO prefix.
    """
    sequence = [x.split('-') for x in sequence]
    for i, x in enumerate(sequence):
        if len(x) == 1: # Other
            sequence[i] = 'Other'
        else:
            sequence[i] = '-'.join(x[1:])
    return sequence


def pad_sequences_(sequences, max_len, pad_token=0):
    lens = [len(x) for x in sequences]
    for i, x in enumerate(sequences):
        tp = type(x)
        if len(x) < max_len:
            sequences[i] = tp(list(x) + [pad_token
                for _ in range(max_len-len(x))])
        else:
            sequences[i] = sequences[i][len(x)-max_len:]
    return masks_from_lens(lens, max_len)

def pad_sequences(sequences, max_len, pad_token=0):
    sequences_ = copy.deepcopy(sequences)
    masks = pad_sequences_(sequences_, max_len, pad_token)
    return sequences_, masks

def assert_types(types, target, name=''):
    if isinstance(types, Iterable):
        expected_str = ' or '.join([str(x.__name__) for x in types])
    else:
        expected_str = types
        types = [types]
    assert_types(str, name, 'target variable')
    name = ' for ' + name
    assert type(target) in types, \
        "Expected {}{}, received {} instead.".format(
            expected_str, name, type(target).__name__
        )

def isiterable(target):
    return isinstance(target, Iterable)

def prepare_indexizer(entities, path):
    def __make_maps(entities, path):
        map_ = {t: i for i, t in enumerate(entities)}
        with open(path, 'w') as f:
            json.dump(map_, f)
        return map_

    def __load_maps(path):
        with open(path, 'r') as f:
            return json.load(f)

    if not os.path.exists(path):
        try:
            os.makedirs(os.path.dirname(path))
        except FileExistsError:
            pass
        return __make_maps(entities, path)
    else:
        return __load_maps(path)

def find_subsequence(target, sequence):
    return [(i, i+len(target)) for i in range(len(sequence)) if sequence[i:i+len(target)] == target][0]

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

def make_few_shots(dataset, nsamps, category_by='label'):
    return_values = dataset.register_returns()
    if type(category_by) is str:
        assert category_by in return_values
        labels = dataset.category_by
        categories = set(labels)
        category_idx = [
            np.random.choice(
                np.arange(len(labels))[labels==i], 
                nsamps
            )
            for i in categories
        ]
    else:
        categories = set(category_by)
        category_idx = [
            np.random.choice(
                np.arange(len(category_by))[category_by==i], 
                nsamps
            )
            for i in categories
        ]

    few_shot_data = copy.deepcopy(dataset)
    return_values_samp = [
            [np.array(x)[idx] for idx in category_idx]
            for x in [
                getattr(dataset, y) for y in return_values
            ]
    ]
    return_values_samp = [
        [x for t in y for x in t] for y in return_values_samp
    ]

    for val_name, vals in zip(return_values, return_values_samp):
        setattr(few_shot_data, val_name, vals)
    return few_shot_data

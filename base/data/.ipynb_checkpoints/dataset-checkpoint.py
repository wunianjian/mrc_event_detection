""" dataset.py: classes for ACE data. """

__author__ = "Rui Feng"

import json
import abc
from transformers import AutoTokenizer
from collections import Iterable
import os

import base.data.utils as utils
from base.data.processor import *
from base.data.tokenizer import Tokenizer


class Dataset:
    """
        Abstract class for ACE dataset. An inheritance from this class must implement
        the following methods:
            1. register_preprocessors(self):
                Returns: Dict[class, argv]. Return a dictionary of preprocessor classes and
                corresponding arguments.
            2. register_returns(self):
                Returns: List[str]. Return a list of attributes to be retrieved by the class.
                Each attribute must be attached by the dataset by one of the preprocessors.
        The abstract class's constructor takes as input the path to the .json file containing
        target data.

        Properties:
            valid_attributes: the set of valid attributes attached by
                preprocessors as return values.

        Example:
            class SimpleData(Dataset):
                def __init__(self, input_file, max_len):
                    # all attributes needed in register_preprocessors needs to be
                    # declared before super's __init__ is called.
                    self.max_len = max_len
                    super(SimpleData, self).__init__(input_file)

                def register_preprocessors(self):
                    return {TokensPreprocessor: {'max_len': self.max_len},
                            EventsBIOPreprocessor: {'max_len': self.max_len}}

                def register_returns(self):
                    return ['sentences', 'masks', 'event_labels', 'event_label_masks']

            Implementing these methods will create a subclass that process data
            according to the two preprocessor classes and return the specified four values
            when indexed:

            data = SimpleData('input_json_file.json', 'auto')
            sentence, mask, event_label, event_label_mask = data[101]

    """

    def __init__(self, input_file):
        with open(input_file, 'r') as f:
            self.data = json.load(f)
        self.__valid_attributes = set()
        self.init_preprocessors(self.register_preprocessors())
        self.return_validity_check()
        self.preprocess()

    @property
    def valid_attributes(self):
        return self.__valid_attributes

    @abc.abstractclassmethod
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        attrs = self.register_returns()
        for attr in attrs:
            assert attr in self.__valid_attributes
        return [getattr(self, x)[index] for x in attrs]

    def init_preprocessors(self, preprocessor_args_dict):
        self.__preprocessors = {}
        for p, argv in preprocessor_args_dict.items():
            self.__preprocessors[p] = p(self, **argv)
        for _, p in self.__preprocessors.items():
            self.__valid_attributes = self.valid_attributes.union(list(p.attributes().keys()))

    def preprocess(self):
        for _, p in self.__preprocessors.items():
            p.preprocess_and_attach_attrs()

    @property
    def preprocessors(self):
        return self.__preprocessors

    @abc.abstractmethod
    def register_preprocessors(self):
        raise NotImplementedError

    @abc.abstractmethod
    def register_returns(self):
        raise NotImplementedError

    def return_validity_check(self):
        attrs = self.register_returns()
        for attr in attrs:
            assert attr in self.__valid_attributes, \
                "Specified return attribute {} not attached by any preprocessor.".format(attr)

class BasicDataset(Dataset):
    def __init__(self, input_file):
        super(BasicDataset, self).__init__(input_file)

    def register_preprocessors(self):
        return {TokensPreprocessor: {},
                EventsBIOPreprocessor: {}}

    def register_returns(self):
        return ['sentences', 'masks', 'event_labels', 'event_label_masks']

class ArgumentDataset(Dataset):
    def __init__(self, input_file, use_augment=False, use_desc=False):
        self.use_augment = use_augment
        self.use_desc = use_desc
        super(ArgumentDataset, self).__init__(input_file)
        self.nevents = len(self.preprocessors[EventTokensPreprocessor].event_map)
        self.nroles = len(self.preprocessors[ArgumentPreprocessor].argument_map)
        self.seq_len = self.sentences.shape[1]

    def register_preprocessors(self):
        items = [(TokensPreprocessor, {}),
                (SentenceLabelPreprocessor, {}),
                (EventTokensPreprocessor, {'ignore_other': True, 'use_desc': self.use_desc}),
                (ArgumentPreprocessor, {})]
        if self.use_augment:
            items.insert(0, (DataGenerationPreprocessor, {}))
        return dict(items)

    def register_returns(self):
        return ['sentences', 'masks', 'sentence_labels', 'event_indexes',
                'argument_labels', 'argument_label_masks', 'trigger_masks']

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        ret = super(ArgumentDataset, self).__getitem__(index)
        event_indexes, ret_arg, ret_arg_masks = ret[3], ret[4], ret[5]
        ret[4] = np.zeros([self.nevents, self.seq_len, self.nroles], dtype=np.int64)
        assert len(ret_arg) == len(event_indexes)
        if len(ret_arg) != 0:
            for index, arg in zip(event_indexes, ret_arg):
                ret[4][index] |= arg
            ret[4][ret[4] == 3] = 1
        return ret

class ArgumentEntityTokensDataset(Dataset):
    def __init__(self, input_file, use_augment=False, use_desc=False):
        self.use_augment = use_augment
        self.use_desc = use_desc
        super(ArgumentEntityDataset, self).__init__(input_file)
        self.nevents = len(self.preprocessors[EventTokensPreprocessor].event_map)
        self.nroles = len(self.preprocessors[ArgumentEntityPreprocessor].argument_map)
        self.seq_len = self.sentences.shape[1]

    def register_preprocessors(self):
        items = [(TokensPreprocessor, {}),
                (EventsBIOPreprocessor, {'ignore_tags': True}),
                (EventTokensPreprocessor, {'ignore_other': True, 'use_desc': self.use_desc}),
                (ArgumentEntityPreprocessor, {}),
                (ArgumentTokensPreprocessor, {})]
        if self.use_augment:
            items.insert(0, (DataGenerationPreprocessor, {}))
        return dict(items)

    def register_returns(self):
        return ['sentences', 'masks', 'event_labels', 'event_label_masks', 'entities', 'entity_masks',
                'argument_labels', 'trigger_masks']

    def __len__(self):
        return len(self.sentences)

class ArgumentEntityDataset(Dataset):
    def __init__(self, input_file, use_augment=False, event_query_template=None):
        self.use_augment = use_augment
        self.event_query_template = event_query_template
        super(ArgumentEntityDataset, self).__init__(input_file)
        self.nevents = len(self.preprocessors[EventTokensPreprocessor].event_map)
        self.nroles = len(self.preprocessors[ArgumentEntityPreprocessor].argument_map)
        self.seq_len = self.sentences.shape[1]

    def register_preprocessors(self):
        items = [(TokensPreprocessor, {}),
                (SentenceLabelPreprocessor, {}), 
                (EventTokensPreprocessor, {'ignore_other': True, 
                    'query_template': self.event_query_template}),
                (ArgumentEntityPreprocessor, {}),
                (ArgumentTokensPreprocessor, {})]
        if self.use_augment:
            items.insert(0, (DataGenerationPreprocessor, {}))
        return dict(items)

    def register_returns(self):
        return ['sentences', 'masks', 'sentence_labels', 'entities', 'entity_masks',
                'argument_labels', 'trigger_masks']

    def __len__(self):
        return len(self.sentences)


class CustomDataset(Dataset):
    def __init__(self, input_file, preprocessor_argvs, returns, length_by):
        self.preprocessor_argvs = preprocessor_argvs
        self.returns = returns
        self.length_by = length_by
        super(CustomDataset, self).__init__(input_file)

    def register_preprocessors(self):
        return self.preprocessor_argvs

    def register_returns(self):
        return self.returns

    def __len__(self):
        return len(getattr(self, self.length_by))


class AugmentDataset(Dataset):
    def __init__(self, input_file):
        super(AugmentDataset, self).__init__(input_file)

    def register_preprocessors(self):
        return {DataGenerationPreprocessor: {},
                TokensPreprocessor: {},
                EventsBIOPreprocessor: {}}

    def register_returns(self):
        return ['sentences', 'masks', 'event_labels', 'event_label_masks']

class RCArgumentDataset(Dataset):
    def __init__(self, 
        input_file, 
        argument_query_template='ARG_EQ1',
        tokenizer='bert-base-uncased',
        negative_sampling='all', 
        use_entity_type=True, 
        is_eval_data=False,
        query_position='prefix'):

        self.argument_query_template = argument_query_template
        self.tokenizer = tokenizer
        self.negative_sampling = negative_sampling
        self.use_entity_type = use_entity_type
        self.is_eval_data = is_eval_data
        self.query_position = query_position

        super().__init__(input_file)

    def register_preprocessors(self):
        return {
            RCArgumentPreprocessor: {
                'argument_query_template': self.argument_query_template,
                'negative_sampling': self.negative_sampling,
                'use_entity_type': self.use_entity_type,
                'is_eval_data': self.is_eval_data,
                'tokenizer': self.tokenizer,
                'query_position': self.query_position,
            }
        }

    def register_returns(self):
        return ['query_tokens', 'query_masks', 'query_labels']

    def __len__(self):
        return len(self.query_tokens)

def create_dataset_template(preprocessor_argvs, returns, length_by):
    if type(preprocessor_argvs) is list:
        preprocessor_argvs = {x: {} for x in preprocessor_argvs}

    class _CustomDataset(Dataset):
        def __init__(self, input_file):
            self.preprocessor_argvs = preprocessor_argvs
            self.returns = returns
            super(_CustomDataset, self).__init__(input_file)

        def register_preprocessors(self):
            return self.preprocessor_argvs

        def register_returns(self):
            return self.returns

        def __len__(self):
            return len(getattr(self, length_by))

    return _CustomDataset

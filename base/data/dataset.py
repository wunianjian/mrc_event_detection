""" dataset.py: classes for ACE data. """

__author__ = "Rui Feng"

import json
import abc
from transformers import AutoTokenizer
from collections import Iterable
import os
import numpy as np

from .processor import *
from .rc_processor import *
from .tokenizer import Tokenizer


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
            assert attr in self.__valid_attributes, (attr, self.__valid_attributes)
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
    def __init__(self, input_file, use_augment=False, event_query_template=None, event_label_processor=None, use_event_description='none'):
        self.use_augment = use_augment
        self.event_query_template = event_query_template
        self.event_label_processor = event_label_processor
        self.use_event_description = use_event_description
        super(ArgumentEntityDataset, self).__init__(input_file)
        self.nevents = len(self.preprocessors[EventTokensPreprocessor].event_map)
        self.nroles = len(self.preprocessors[ArgumentEntityPreprocessor].argument_map)
        self.seq_len = self.sentences.shape[1]

    def register_preprocessors(self):
        items = [(TokensPreprocessor, {}),
                (SentenceLabelPreprocessor, {}), 
                (EventTokensPreprocessor, {'ignore_other': True, 
                    'query_template': self.event_query_template,
                    'event_label_processor': self.event_label_processor,
                    'use_event_description': self.use_event_description}),
                (ArgumentEntityPreprocessor, {}),
                (ArgumentTokensPreprocessor, {})]
        if self.use_augment:
            items.insert(0, (DataGenerationPreprocessor, {}))
        return dict(items)

    def make_few_shot(self, nsamps):
        labels = np.array(self.sentence_labels)
        labels_unique = labels.unique()
        label_indices = [
            np.arange(len(labels))[labels==i] for i in labels_unique
        ] 
        label_index_samps = [np.random.choice(x, nsamps) for x in label_indices]
        labels = [x for t in label_indices for x in t]
        for attr in self.register_returns():
            setattr(
                attr, getattr(self, x)[sample_indices]
            )

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

class RCEventDataset(Dataset):
    def __init__(self,
        input_file,
        event_label_processor='IdentityEventProcessor',
        event_query_template="EQ1",
        use_event_description='none',
        tokenizer='bert-base-uncased',
        query_position='prefix',
        use_sep=True,
        nsamps='all',
        max_desc_sentences=10,
        span_detection=False
    ):
        self.reverse_query = query_position == 'postfix'
        self.tokenizer = tokenizer
        self.event_label_processor = event_label_processor
        self.event_query_template = event_query_template
        self.use_event_description = use_event_description
        self.use_sep = use_sep
        self.nsamps = nsamps
        self.max_desc_sentences = max_desc_sentences
        self.span_detection = span_detection
        self.preprocessor_class = RCEventSpanPreprocessor if span_detection else RCEventPreprocessor

        super().__init__(input_file)

    def register_preprocessors(self):
        return {
            self.preprocessor_class: {
                'tokenizer': self.tokenizer,
                'event_query_template': self.event_query_template,
                'event_label_processor': self.event_label_processor,
                'use_event_description': self.use_event_description,
                'reverse_query': self.reverse_query,
                'nsamps': self.nsamps,
                'max_desc_sentences': self.max_desc_sentences
            }
        }

    def reindex_with_sentence(self, attr_name):
        attr = getattr(self, attr_name)
        new_attr = [
            [attr[t] for t in range(x[0], x[1])] 
            for x in self.sentence_inst_idx_span
        ]
        new_attr = [x for t in new_attr for x in t]
        setattr(self, attr_name, new_attr)

    def grouped_shuffle(self):
        np.random.shuffle(self.sentence_inst_idx_span)
        for x in self.register_returns():
            if (not isinstance(x, Iterable)) or len(x)!=len(self):
                continue
            self.reindex_with_sentence(x)
        cur_idx = 0
        self.sentence_inst_idx_span_ = []
        for x in self.sentence_inst_idx_span:
            self.sentence_inst_idx_span_.append(
                (cur_idx, cur_idx + (x[1] - x[0]))
            )
            cur_idx += x[1] - x[0]
        self.sentence_inst_idx_span = self.sentence_inst_idx_span_

    def register_returns(self):
        returns = list(self.preprocessors[self.preprocessor_class].tok_returns)
        if self.span_detection:
            returns += ['start_labels', 'end_labels']
            returns += ['token_head_ids']
        else:
            returns += ['labels']
        return returns

    def __len__(self):
        return len(self.input_ids)


class RCArgumentDataset(Dataset):
    def __init__(self, 
        input_file, 
        argument_query_template='ARG_EQ1',
        desc_type=None,
        tokenizer='bert-base-uncased',
        negative_sampling='all',
        n_sample = 'all',
        few_shot_type='argument',
        use_entity_type=True, 
        is_eval_data=False,
        query_position='prefix',
        event_preds=None,
        train_without_guide=True,
    ):
        self.argument_query_template = argument_query_template
        self.desc_type = desc_type
        self.tokenizer = tokenizer
        self.negative_sampling = negative_sampling
        self.n_sample = n_sample
        self.few_shot_type = few_shot_type
        self.use_entity_type = use_entity_type
        self.is_eval_data = is_eval_data
        self.query_position = query_position
        self.use_entity = getattr(qa, argument_query_template)().use_entity
        self.event_preds = event_preds
        self.train_without_guide = train_without_guide

        super(RCArgumentDataset, self).__init__(input_file)

    def register_preprocessors(self):
        return {
            RCArgumentPreprocessor: {
                'argument_query_template': getattr(qa, self.argument_query_template),
                'desc_type': self.desc_type,
                'negative_sampling': self.negative_sampling,
                'n_sample': self.n_sample,
                'few_shot_type': self.few_shot_type,
                'use_entity_type': self.use_entity_type,
                'is_eval_data': self.is_eval_data,
                'tokenizer': self.tokenizer,
                'query_position': self.query_position,
                'event_preds': self.event_preds,
                'query_without_guide': (not self.is_eval_data) and self.train_without_guide
            }
        }

    def register_returns(self):
        returns = ['input_ids', 'token_type_ids', 'attention_mask', 'query_labels', 'id_tuples', 'pred_masks', 'label_masks']
        if not self.use_entity:
            returns += ['argument_start_labels', 'argument_end_labels', 'entity_spans', 'possible_entity_masks']
        return returns

    def __getitem__(self, item):
        rets = super(RCArgumentDataset, self).__getitem__(item)
        for i, ret in enumerate(rets):
            try:
                single_shape = ret.shape
            except AttributeError as e:
                print(e)
                print(self.register_returns()[i])
                exit(0)
        return rets

    def __len__(self):
        return len(self.query_labels)


class RCArgumentWithoutEntitiesDataset(Dataset):
    def __init__(self, input_path, **kwarg):
        self.args = dict(kwarg)
        for key, value in dict(kwarg).items():
            setattr(self, key, value)
        super().__init__(input_path)

    def register_preprocessors(self):
        return {
            RCArgumentWithoutEntitiesPreprocessor: self.args
        }

    def register_returns(self):
        return ['query_tokens', 'query_masks', 'token_type_ids', 'starts', 'ends', 'sentence_indices']

    def __len__(self):
        return len(self.query_tokens)

class RCEntailEventsDataset(Dataset):
    def __init__(self, input_path, **kwargs):
        self.args = dict(kwargs)
        for key, value in dict(kwargs).items():
            setattr(self, key, value)
        super().__init__(input_path)

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

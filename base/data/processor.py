import os
import abc
import math
import json
import copy
import heapq
from collections import defaultdict

import torch
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertForMaskedLM, BertModel, BertTokenizer

from . import utils, qa
from .tokenizer import Tokenizer, encode_pretty
from tqdm import tqdm

class Preprocessor:
    """
    Base class for preprocessors. To create an custom preprocessor, inherit from this class.
    This class stores a Dataset object as self.data_ref, from which the raw data self.data and
    other attributes produced by previous preprocessors can be accessed.

    A subclass must implement these following functions:
        1. preprocess(self):
            Returns: None.
            Customized function to preprocess data. Do whatever work is required here, and
            store resulting attributes in class.
        2. attributes(self):
            Returns: Dict[str, str].
            Return a dictionary specifying the attributes to attach to accompanying dataset class.
            Keys of the dictionary are the attribute names for the dataset class.
            Values are the attribute names in the preprocessor class, produced by preprocess().

    Example:
        class MagicPreprocessor(Preprocessor):
            def preprocess(self):
                self.magic_results = magic_trick(self.data_ref)

            def attributes(self):
                return {'processed_data': 'magic_results'}

    """

    def __init__(self, data_ref):
        self.data_ref = data_ref

    @abc.abstractmethod
    def preprocess(self):
        raise NotImplementedError

    def preprocess_and_attach_attrs(self):
        self.preprocess()
        self.attach_attributes()

    @abc.abstractmethod
    def attributes(self):
        pass

    def attach_attributes(self):
        for tgt, src in self.attributes().items():
            setattr(self.data_ref, tgt, getattr(self, src))

class TokensPreprocessor(Preprocessor):
    """
    Tokens preprocessor. Tokenizes sentences in data according to the given tokenizer.
    Arguments:
        max_len: Union(Str, Int): Either 'auto' or an int of max sentence length.
            If 'auto' is received, will be the maximum sentence length in data.
        tokenizer: Str: either a path to a .json token-to-index dict file,
            or a Transformer tokenizer name, such as 'bert-base-uncased'.

    This preprocessor attaches the following attributes:
        sentences: tokenized, indexized, and padded sentences.
        masks: masks.
    """

    def __init__(self, data_ref, tokenizer='bert-base-uncased'):
        super(TokensPreprocessor, self).__init__(data_ref)
        self.tokenizer_type = tokenizer
        self.init_tokenizer()

    def attributes(self):
        return {'sentences': 'sentences',
                'masks': 'masks',
                'tokenize_index_map': 'tokenize_index_map',
                'tokenizer': 'tokenizer'}

    def init_tokenizer(self):
        self.tokenizer = Tokenizer(self.tokenizer_type)

    def tokenize_sentences(self):
        # self.sentences = self.tokenizer.encode(self.sentences)
        tokenized = []
        self.tokenize_index_map = []
        for sent in self.sentences:
            encoded, index_map = encode_pretty(sent, self.tokenizer.encode, self.tokenizer.decode)
            encoded = [int(x) for x in encoded]  # make sure IndexedToken converts to int
            tokenized.append(encoded)
            self.tokenize_index_map.append(index_map)
        self.sentences = tokenized

    def pad(self):
        max_len = max(map(len, self.sentences))
        self.masks = utils.pad_sequences_(self.sentences, max_len, pad_token=0)

    def preprocess(self):
        data = self.data_ref.data
        self.sentences = utils.extract_property(data, 'words')
        self.tokenize_sentences()
        self.pad()
        self.masks = np.array(self.masks)
        self.sentences = np.array(self.sentences)

class EventsBIOPreprocessor(Preprocessor):
    def __init__(self, data_ref, ignore_subwords=True, ignore_tags=False, ignore_other=False):
        super(EventsBIOPreprocessor, self).__init__(data_ref)
        self.ignore_subwords = ignore_subwords
        self.ignore_tags = ignore_tags
        self.last_other = ignore_other

    def pad(self):
        max_len = max(map(len, self.event_labels))
        utils.pad_sequences_(self.event_labels, max_len,
                             pad_token=self.event_tag_map['Other'])

    def attributes(self):
        return {'event_labels': 'event_labels',
                'event_label_masks': 'masks',
                'reverse_label_map': 'rev_event_tag_map'}

    def prepare_event_tag_maps(self):
        self.event_tags = utils.extract_property(self.data_ref.data, 'event-labels')
        if self.ignore_tags:
            self.event_tag_map_path = 'res/maps/event_map.json'
            event_tag_tokens = set([x.split('-', 1)[-1] for t in self.event_tags for x in t])
        else:
            self.event_tag_map_path = 'res/maps/event_tag_map.json'
            event_tag_tokens = set([x for t in self.event_tags for x in t])
        self.event_tag_map = utils.prepare_indexizer(event_tag_tokens, self.event_tag_map_path)
        if self.last_other:
            self.event_tag_map[list(self.event_tag_map.keys())[list(self.event_tag_map.values()).index(len(self.event_tag_map) - 1)]] = self.event_tag_map['Other']
            self.event_tag_map['Other'] = len(self.event_tag_map) - 1
        self.rev_event_tag_map = {t: i for i, t in self.event_tag_map.items()}
        self.null_index = self.event_tag_map['Other']

    def preprocess(self):
        self.prepare_event_tag_maps()
        event_labels_og = utils.extract_property(self.data_ref.data, 'event-labels')
        if self.ignore_tags:
            event_labels_og = [[self.event_tag_map[x.split('-', 1)[-1]] for x in y] for y in event_labels_og]
        else:
            event_labels_og = [[self.event_tag_map[x] for x in y] for y in event_labels_og]
        event_labels = [[self.null_index for _ in range(self.data_ref.sentences.shape[1])]
                        for _ in event_labels_og]
        event_label_masks = [[False for _ in range(self.data_ref.sentences.shape[1])]
                             for _ in event_labels_og]
        for i, (label_og, index_map) in enumerate(zip(event_labels_og, self.data_ref.tokenize_index_map)):
            # for j, t in enumerate(label_og):
            for og_idx, tgt_idx in index_map.items():
                if og_idx == len(label_og):
                    continue
                head_idx = tgt_idx[0]
                tail_idx = tgt_idx[1]
                if self.ignore_subwords:
                    event_labels[i][head_idx] = label_og[og_idx]
                    event_label_masks[i][head_idx] = True
                else:
                    for k in range(head_idx, tail_idx):
                        event_labels[i][k] = label_og[og_idx]
                        event_label_masks[i][k] = True
        self.event_labels = np.array(event_labels)
        self.masks = np.array(event_label_masks)

class ArgumentPreprocessor(EventsBIOPreprocessor):
    def __init__(self, data_ref, ignore_subwords=True):
        super(ArgumentPreprocessor, self).__init__(data_ref, ignore_tags=True, ignore_other=True)
        self.ignore_subwords = ignore_subwords

    def pad(self):
        max_len = max(map(len, self.event_labels))
        utils.pad_sequences_(self.event_labels, max_len,
                             pad_token=self.argument_map['Other'])

    def attributes(self):
        return {'event_indexes': 'event_indexes',
                'argument_labels': 'argument_labels',
                'argument_label_masks': 'argument_masks',
                'trigger_masks': 'trigger_masks'}

    def prepare_argument_maps(self):
        event_mentions_list = utils.extract_property(self.data_ref.data, 'event-mentions')
        self.argument_map_path = 'res/maps/argument_map.json'
        arguments_list = [event_mention["arguments"] for event_mentions in event_mentions_list for event_mention in event_mentions]
        argument_tokens = set([argument['role'] for arguments in arguments_list for argument in arguments])
        self.argument_map = utils.prepare_indexizer(argument_tokens, self.argument_map_path)
        self.rev_argument_map = {t: i for i, t in self.argument_map.items()}

    def preprocess(self):
        self.prepare_event_tag_maps()
        self.prepare_argument_maps()
        del self.event_tag_map['Other']
        event_mentions_list = utils.extract_property(self.data_ref.data, 'event-mentions')
        event_indexes = [[] for _ in event_mentions_list]
        argument_labels = [[] for _ in event_mentions_list]
        argument_label_masks = [[False for _ in range(self.data_ref.sentences.shape[1])]
                                for _ in event_mentions_list]
        trigger_masks = np.zeros([len(event_mentions_list), len(self.event_tag_map), self.data_ref.sentences.shape[1]])
        for i, (event_mentions, index_map) in enumerate(zip(event_mentions_list, self.data_ref.tokenize_index_map)):
            for og_idx, tgt_idx in index_map.items():
                head_idx = tgt_idx[0]
                tail_idx = tgt_idx[1]
                if self.ignore_subwords:
                    argument_label_masks[i][head_idx] = True
                else:
                    for k in range(head_idx, tail_idx):
                        argument_label_masks[i][k] = True
            for event_mention in event_mentions:
                trigger = event_mention['trigger']
                trigger_masks[i, self.event_tag_map[event_mention['event_type']], trigger['start']: trigger['end']] = True
                arguments = event_mention['arguments']
                argument_label = [[0 for _ in range(len(self.argument_map))] for _ in range(self.data_ref.sentences.shape[1])]
                for argument in arguments:
                    start, end = argument['start'], argument['end']
                    role = self.argument_map[argument['role']]
                    for idx in range(start, end):
                        head_idx, tail_idx = index_map[idx]
                        if self.ignore_subwords:
                            argument_label[head_idx][role] = 1 if idx != start else 2
                            for k in range(head_idx + 1, tail_idx):
                                argument_label[k][role] = -100
                        else:
                            for k in range(head_idx, tail_idx):
                                argument_label[k][role] = 1 if idx != start else 2
                event_indexes[i].append(self.event_tag_map[event_mention['event_type']])
                argument_labels[i].append(argument_label)
        self.event_indexes = np.array(event_indexes)
        self.argument_labels = np.array(argument_labels)
        self.argument_masks = np.array(argument_label_masks)
        self.trigger_masks = trigger_masks

class ArgumentEntityPreprocessor(EventsBIOPreprocessor):
    def __init__(self, data_ref, ignore_subwords=True):
        super(ArgumentEntityPreprocessor, self).__init__(data_ref, ignore_tags=True, ignore_other=True)
        self.ignore_subwords = ignore_subwords

    def attributes(self):
        return {'entities': 'entities',
                'entity_masks': 'entity_masks',
                'argument_labels': 'argument_labels',
                'trigger_masks': 'trigger_masks'}

    def prepare_argument_maps(self):
        event_mentions_list = utils.extract_property(self.data_ref.data, 'event-mentions')
        self.argument_map_path = 'res/maps/argument_map.json'
        arguments_list = [event_mention["arguments"] for event_mentions in event_mentions_list for event_mention in event_mentions]
        argument_tokens = set([argument for arguments in arguments_list for argument in arguments])
        self.argument_map = utils.prepare_indexizer(argument_tokens, self.argument_map_path)
        self.rev_argument_map = {t: i for i, t in self.argument_map.items()}

    def preprocess(self):
        self.prepare_event_tag_maps()
        self.prepare_argument_maps()
        del self.event_tag_map['Other']
        event_mentions_list = utils.extract_property(self.data_ref.data, 'event-mentions')
        entities_list = utils.extract_property(self.data_ref.data, 'entities')
        max_entity_num = max(map(len, entities_list))
        max_seq_len = self.data_ref.sentences.shape[1]
        entities = np.zeros([len(event_mentions_list), max_entity_num, max_seq_len])
        entity_masks = np.zeros([len(event_mentions_list), max_entity_num], dtype=np.bool)
        argument_labels = np.zeros([len(event_mentions_list), len(self.event_tag_map), max_entity_num, len(self.argument_map)])
        trigger_masks = np.zeros([len(event_mentions_list), len(self.event_tag_map), max_seq_len])
        for i, (event_mentions, index_map, entity) in enumerate(zip(event_mentions_list, self.data_ref.tokenize_index_map, entities_list)):
            entity_masks[i][:len(entity)] = True
            for j, en in enumerate(entity):
                entities[i][j][index_map[en['head']['start']][0]: index_map[en['head']['end']-1][1]] = True
            for event_mention in event_mentions:
                trigger = event_mention['trigger']
                trigger_masks[i, self.event_tag_map[event_mention['event_type']], trigger['start']: trigger['end']] = True
                arguments = event_mention['arguments']
                event_type = self.event_tag_map[event_mention['event_type']]
                for j, role in enumerate(arguments):
                    if role != 'None':
                        argument_labels[i][event_type][j][self.argument_map[role]] = True
        self.entities = entities
        self.entity_masks = entity_masks
        self.argument_labels = argument_labels
        self.trigger_masks = trigger_masks

class SentenceArgumentPreprocessor(ArgumentPreprocessor):
    def __init__(self, data_ref):
        super(SentenceArgumentPreprocessor, self).__init__(data_ref)

    def attributes(self):
        return {'sentence_argument_labels': 'sentence_argument_labels'}

    def preprocess(self):
        self.prepare_argument_maps()
        event_mentions_list = utils.extract_property(self.data_ref.data, 'event-mentions')
        sentence_argument_labels = [[False for _ in range(len(self.argument_map))]
                                    for _ in event_mentions_list]
        for i, event_mentions in enumerate(event_mentions_list):
            for event_mention in event_mentions:
                arguments = event_mention['arguments']
                for argument in arguments:
                    role = self.argument_map[argument['role']]
                    sentence_argument_labels[i][role] = True
        self.sentence_argument_labels = np.array(sentence_argument_labels)

class SentenceLabelPreprocessor(EventsBIOPreprocessor):
    def __init__(self, data_ref):
        super(SentenceLabelPreprocessor, self).__init__(data_ref, ignore_tags=True, ignore_other=True)

    def attributes(self):
        return {'sentence_labels': 'sentence_labels'}

    def preprocess(self):
        super(SentenceLabelPreprocessor, self).preprocess()
        event_labels = [[(x, i) for i, x in enumerate(y) if x != self.null_index] for y in self.event_labels]
        sentence_labels = []
        del self.event_tag_map['Other']
        for event_label in event_labels:
            sentence_label = np.zeros(len(self.event_tag_map), dtype=np.bool)
            trigger_index = []
            trigger_label = []
            for (label_id, index) in event_label:
                if label_id == len(self.event_tag_map):
                    continue
                sentence_label[label_id] = True
                trigger_index.append(index)
                trigger_label.append(label_id)
            sentence_labels.append(sentence_label)
        self.sentence_labels = np.array(sentence_labels)

class EventTokensPreprocessor(Preprocessor):
    def __init__(self, data_ref, ignore_other=True, query_template=None, event_label_processor=None, use_event_description=None):
        super(EventTokensPreprocessor, self).__init__(data_ref)
        self.ignore_other = ignore_other
        if query_template is None:
            query_template = "IdentityQuery"
        self.query_template = getattr(qa, query_template)
        self.event_label_processor = getattr(qa, event_label_processor)
        self.use_event_description = use_event_description
        self.query_maker = qa.EventQueryMaker(
            self.event_label_processor,
            self.query_template,
            self.use_event_description
        )

    def prepare_event_maps(self):
        self.event_map_path = 'res/maps/event_map.json'
        self.event_tags = utils.extract_property(self.data_ref.data, 'event-labels')
        event_tokens = list(set(utils.debio(set([x for t in self.event_tags for x in t]))))
        self.event_map = utils.prepare_indexizer(event_tokens, self.event_map_path)
        self.null_index = self.event_map['Other']
        if self.ignore_other:
            self.event_map[list(self.event_map.keys())[list(self.event_map.values()).index(len(self.event_map) - 1)]] = self.event_map['Other']
            del self.event_map['Other']
        self.rev_event_map = {t: i for i, t in self.event_map.items()}
        self.event_tokens = [self.rev_event_map[x]
            for x in range(len(self.rev_event_map))]

    def tokenize_and_pad(self):
        tokenizer = self.data_ref.preprocessors[TokensPreprocessor].tokenizer
        self.event_tokens_og = []
        for i, x in enumerate(self.event_tokens):
            self.event_tokens[i] = [int(t) for t in tokenizer.encode(self.query_maker.encode(x))]
            self.event_tokens_og.append(
                [int(t) for t in tokenizer.encode(x)]
            )

        max_len = max(map(len, self.event_tokens))
        self.masks = utils.pad_sequences_(self.event_tokens, max_len, pad_token=0)
        self.event_tokens = np.array(self.event_tokens)
        self.masks = np.array(self.masks)

        max_len = max(map(len, self.event_tokens_og))
        self.og_masks = utils.pad_sequences_(self.event_tokens_og, max_len, pad_token=0)
        self.event_tokens_og = np.array(self.event_tokens_og)
        self.og_masks = np.array(self.og_masks)

    def prepare_event_matrices(self):
        events = utils.extract_property(self.data_ref.data, 'event-mentions')
        for i, x in enumerate(events):
            events[i] = [t['event_type'] for t in x]
        self.event_matrices = np.zeros((len(events), len(self.event_map)))
        for i, x in enumerate(events):
            if len(x) == 0 and not self.ignore_other:
                self.event_matrices[i][self.null_index] = 1
            else:
                for event in x:
                    self.event_matrices[i][self.event_map[event]] = 1

    def preprocess(self):
        self.prepare_event_maps()
        self.tokenize_and_pad()
        self.prepare_event_matrices()

    def attributes(self):
        return {'event_tokens': 'event_tokens',
                'event_token_masks': 'masks',
                'event_matrices': 'event_matrices',
                'event_tokens_og': 'event_tokens_og',
                'event_token_og_masks': 'og_masks',
                'event_map': 'event_map',
                'rev_event_map': 'rev_event_map'}


class ArgumentTokensPreprocessor(Preprocessor):
    def __init__(self, data_ref):
        super(ArgumentTokensPreprocessor, self).__init__(data_ref)

    def prepare_argument_maps(self):
        event_mentions_list = utils.extract_property(self.data_ref.data, 'event-mentions')
        self.argument_map_path = 'res/maps/argument_map.json'
        arguments_list = [event_mention["arguments"] for event_mentions in event_mentions_list for event_mention in event_mentions]
        argument_tokens = set([argument for arguments in arguments_list for argument in arguments])
        argument_tokens.remove('None')
        self.argument_map = utils.prepare_indexizer(argument_tokens, self.argument_map_path)
        self.rev_argument_map = {t: i for i, t in self.argument_map.items()}
        self.argument_tokens = [self.rev_argument_map[x] for x in range(len(self.rev_argument_map))]

    def tokenize_and_pad(self):
        tokenizer = self.data_ref.preprocessors[TokensPreprocessor].tokenizer
        for i, x in enumerate(self.argument_tokens):
            self.argument_tokens[i] = [int(t) for t in tokenizer.encode(x)]

        max_len = max(map(len, self.argument_tokens))
        self.masks = utils.pad_sequences_(self.argument_tokens, max_len, pad_token=0)
        self.argument_tokens = np.array(self.argument_tokens)
        self.masks = np.array(self.masks)

    def preprocess(self):
        self.prepare_argument_maps()
        self.tokenize_and_pad()

    def attributes(self):
        return {'argument_tokens': 'argument_tokens',
                'argument_token_masks': 'masks'}


class IndependentArgumentPreprocessor(ArgumentPreprocessor):
    def __init__(self, dataref):
        super(IndependentArgumentPreprocessor, self).__init__(dataref)

    def attributes(self):
        return {'sentences': 'sentences',
                'masks': 'masks',
                'trigger_masks': 'trigger_masks',
                'argument_labels': 'argument_labels',
                'argument_label_masks': 'argument_label_masks',
                'event_indexes': 'event_indexes'}

    def preprocess(self):
        self.prepare_event_tag_maps()
        self.prepare_argument_maps()
        del self.event_tag_map['Other']
        event_mentions_list = utils.extract_property(self.data_ref.data, 'event-mentions')
        sentences = []
        masks = []
        trigger_masks = []
        event_indexes = []
        argument_labels = []
        argument_label_masks = []
        for i, (event_mentions, index_map) in enumerate(zip(event_mentions_list, self.data_ref.tokenize_index_map)):
            argument_label_mask = [False for _ in range(self.data_ref.sentences.shape[1])]
            for og_idx, tgt_idx in index_map.items():
                head_idx = tgt_idx[0]
                tail_idx = tgt_idx[1]
                if self.ignore_subwords:
                    argument_label_mask[head_idx] = True
                else:
                    for k in range(head_idx, tail_idx):
                        argument_label_mask[k] = True
            for event_mention in event_mentions:
                trigger = event_mention['trigger']
                arguments = event_mention['arguments']
                argument_label = np.zeros([self.data_ref.sentences.shape[1], len(self.argument_map)])
                trigger_mask = np.zeros(self.data_ref.sentences.shape[1])
                trigger_mask[trigger['start']: trigger['end']] = True
                for argument in arguments:
                    start, end = argument['start'], argument['end']
                    role = self.argument_map[argument['role']]
                    for idx in range(start, end):
                        head_idx, tail_idx = index_map[idx]
                        if self.ignore_subwords:
                            argument_label[head_idx][role] = 1 if idx != start else 2
                            for k in range(head_idx + 1, tail_idx):
                                argument_label[k][role] = -100
                        else:
                            for k in range(head_idx, tail_idx):
                                argument_label[k][role] = 1 if idx != start else 2
                sentences.append(self.data_ref.sentences[i])
                masks.append(self.data_ref.masks[i])
                trigger_masks.append(trigger_mask)
                event_indexes.append(self.event_tag_map[event_mention['event_type']])
                argument_labels.append(argument_label)
                argument_label_masks.append(argument_label_mask)
        self.sentences = np.array(sentences)
        self.masks = np.array(masks)
        self.trigger_masks = np.array(trigger_masks)
        self.event_indexes = np.array(event_indexes)
        self.argument_labels = np.array(argument_labels)
        self.argument_label_masks = np.array(argument_label_masks)


class DataGenerationPreprocessor(Preprocessor):
    def __init__(self, data_ref, m=0.4, n=1.0, lam=0.5,
                 elmo_weight_file="../elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                 elmo_options_file="../elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                 bert_model='../ACE_LM', device=0):
        super(DataGenerationPreprocessor, self).__init__(data_ref)
        self.save_path = 'res/augment/augment_data.json'
        self.m, self.n, self.lam = m, n, lam
        self.elmo_model = Elmo(options_file=elmo_options_file,
                               weight_file=elmo_weight_file,
                               num_output_representations=1)
        self.bert_masked_model = BertForMaskedLM.from_pretrained(bert_model)
        self.bert_model = BertModel.from_pretrained(bert_model)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.device = device
        if device >= 0:
            self.elmo_model.cuda(device)
            self.bert_model.cuda(device)
            self.bert_masked_model.cuda(device)
        self.elmo_model.eval()
        self.bert_model.eval()
        self.bert_masked_model.eval()
        self.role_map = defaultdict(dict)
        self.orig_events = []
        self.gen_events = []
        self.final_augment_data = []

    def preprocess(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r') as fp:
                self.final_augment_data = json.load(fp)
                self.data_ref.data += self.final_augment_data
                return
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.collect_argument()
        self.generate_event()
        self.scoring()

    def attributes(self):
        return {}

    def bert_sentence_encode(self, inputs):
        inputs = self.bert_tokenizer.encode_plus(inputs, return_tensors="pt")
        for key in inputs:
            inputs[key] = inputs[key].to(self.bert_model.device)
        return self.bert_model(inputs, )[0][0][0].detach().cpu().numpy().tolist()

    def collect_argument(self):
        def overlap(arg, args):
            for tmp in args:
                if tmp != arg and not (arg['end'] <= tmp['start'] or arg['start'] >= tmp['end']):
                    args.remove(tmp)
                    return True
            return False

        def elmo_encode(inputs):
            char_ids = batch_to_ids([inputs])
            if self.device >= 0:
                char_ids = char_ids.cuda(self.device)
            return np.array(self.elmo_model.forward(char_ids)['elmo_representations'][0][0].detach().cpu())

        role_map_path = 'res/augment/role_map.json'
        event_feature_list_path = 'res/augment/event_list.json'
        if os.path.exists(event_feature_list_path):
            assert os.path.exists(role_map_path)
            with open(role_map_path, 'r') as fp:
                self.role_map = json.load(fp)
            with open(event_feature_list_path, 'r') as fp:
                self.orig_events = json.load(fp)
            return
        events_list = utils.extract_property(self.data_ref.data, 'event-mentions')
        words_list = utils.extract_property(self.data_ref.data, 'words')
        event_labels_list = utils.extract_property(self.data_ref.data, 'event-labels')
        for i, (event_labels, words, events) in enumerate(zip(event_labels_list, words_list, events_list), 1):
            print("\rcollect arguments %d/%d" % (i, len(words_list)), end='')
            words_embeddings = elmo_encode(words)
            for event in events:
                all_arg_list = event['arguments']
                arg_list = copy.deepcopy(
                    list(filter(lambda x: not overlap(x, all_arg_list + [event['trigger']]), all_arg_list)))
                for argument in arg_list:
                    argument_text = words[argument['start']: argument['end']]
                    if ' '.join(argument_text) not in self.role_map[argument['role']]:
                        self.role_map[argument['role']][' '.join(argument_text)] = np.mean(elmo_encode(argument_text),
                                                                                           axis=0).tolist()
                    argument['embedding'] = np.mean(words_embeddings[argument['start']: argument['end']],
                                                    axis=0).tolist()
                event['replace_argument'] = arg_list
                event['words'] = words
                event['event-labels'] = event_labels
                self.orig_events.append(event)
        print()
        with open(role_map_path, 'w') as fp:
            json.dump(self.role_map, fp)
        with open(event_feature_list_path, 'w') as fp:
            json.dump(self.orig_events, fp)

    def generate_event(self):
        gen_event_path = 'res/augment/gen_event.json'
        if os.path.exists(gen_event_path):
            with open(gen_event_path, 'r') as fp:
                self.gen_events = json.load(fp)
            return
        for i in range(4):
            events = copy.deepcopy(self.orig_events)
            for j, new_event in enumerate(events, 1):
                print("\rgenerate events %d/%d" % (j + i * len(events), 4 * len(events)), end='')
                # ----replace arguments----
                for argument in new_event['replace_argument']:
                    if np.random.random() < 0.2:
                        continue
                    arg_words = new_event["words"][argument['start']: argument['end']]
                    arg_words_str = argument['text']
                    role_args = self.role_map[argument['role']]
                    similarity_tuple = []
                    for role_arg, embedding in role_args.items():
                        if role_arg != arg_words_str:
                            similarity_tuple.append(
                                (role_arg, cosine_similarity([argument['embedding']], [embedding])[0][0]))
                    if len(similarity_tuple) == 0:
                        continue
                    n = math.ceil(0.1 * len(role_args))
                    top_n = heapq.nlargest(n, similarity_tuple, key=lambda x: x[1])
                    exp_n = np.exp([x[1] for x in top_n])
                    softmax_n = exp_n / np.sum(exp_n)
                    random_select_index = np.random.choice(n, p=softmax_n)
                    random_select = top_n[random_select_index][0].split(' ')
                    random_select_text = ' '.join(random_select)
                    random_select.reverse()
                    del new_event['words'][argument['start']: argument['end']]
                    for word in random_select:
                        new_event['words'].insert(argument['start'], word)
                    offset = len(random_select) - len(arg_words)
                    if offset >= 0:
                        for _ in range(offset):
                            new_event['event-labels'].insert(argument['start'], 'Other')
                    else:
                        del new_event['event-labels'][argument['start']: argument['start'] - offset]
                    if argument['start'] < new_event['trigger']['start']:
                        new_event['trigger']['start'] += offset
                        new_event['trigger']['end'] += offset
                    for arg in new_event['replace_argument'] + new_event['arguments']:
                        if arg['start'] > argument['start']:
                            arg['start'] += offset
                            arg['end'] += offset
                        elif arg['start'] == argument['start']:
                            arg['end'] += offset
                            arg['text'] = random_select_text
                del new_event['replace_argument']

                # ----replace adjunct tokens----
                adjunct_tokens = set(range(len(new_event['words']))) - \
                                 set(range(new_event['trigger']['start'], new_event['trigger']['end']))
                for arg in new_event['arguments']:
                    adjunct_tokens -= set(range(arg['start'], arg['end']))
                adjunct_tokens = np.array(list(adjunct_tokens))
                random_index = np.random.choice(len(adjunct_tokens), int(self.m * len(adjunct_tokens)), replace=False)
                step_size = int(len(adjunct_tokens) * 0.15) or 1
                num_step = math.ceil(len(random_index) / step_size)
                words = np.array([word.lower() for word in new_event['words']])
                probability = 0
                for step in range(num_step):
                    cur_tokens_indexes = adjunct_tokens[random_index[step * step_size: (step + 1) * step_size]]
                    words[cur_tokens_indexes] = '[MASK]'
                    inputs = self.bert_tokenizer.encode_plus(words.tolist(), return_tensors="pt")
                    if self.device >= 0:
                        for key in inputs:
                            inputs[key] = inputs[key].cuda(self.device)
                    predictions = torch.softmax(self.bert_masked_model(**inputs)[0], dim=2)
                    predicted_index = torch.argmax(predictions[0, :], dim=1)
                    for index in cur_tokens_indexes:
                        probability += predictions[0, index + 1, predicted_index[index + 1]].item()
                    words[cur_tokens_indexes] = np.array(self.bert_tokenizer.convert_ids_to_tokens(predicted_index))[
                        cur_tokens_indexes + 1]
                new_event['words'] = words.tolist()
                if len(random_index) != 0:
                    probability /= len(random_index)
                else:
                    probability = 0
                new_event['probability'] = probability
                self.gen_events.append(new_event)
        print()
        with open(gen_event_path, 'w') as fp:
            json.dump(self.gen_events, fp)

    def scoring(self):
        q_list = [event['probability'] for event in self.gen_events]
        sorted_index = np.argsort(-np.array(q_list))
        reserve_num = int(self.n / 4 * len(self.gen_events))
        self.gen_events = np.array(self.gen_events)[sorted_index[:reserve_num]].tolist()
        for event in self.gen_events:
            sample = {'words': event['words'], 'event-mentions': [event], 'event-labels': event['event-labels']}
            try:
                del event['words'], event['event-labels'], event['probability'], event['embedding']
            except:
                pass
            self.final_augment_data.append(sample)
        self.data_ref.data += self.final_augment_data
        with open(self.save_path, 'w') as fp:
            json.dump(self.final_augment_data, fp)


class PostProcessor:
    """ TODO: a class that post-processes data batches.
    """

    def __init__(self):
        raise NotImplementedError

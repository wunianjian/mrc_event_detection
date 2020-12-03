from base.data.processor import *
import os
import abc
import math
import json
import copy
import heapq
import nltk
from collections import defaultdict
from collections.abc import Iterable

import torch
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertForMaskedLM, BertModel, BertTokenizer, AutoTokenizer

from . import utils, qa
from .tokenizer import Tokenizer, encode_pretty
from .processor import *
from tqdm import tqdm


class RCArgumentPreprocessor(ArgumentEntityPreprocessor):
    def __init__(self,
                 data_ref,
                 argument_query_template,
                 desc_type=None,
                 negative_sampling='all',
                 n_sample='all',
                 few_shot_type='argument',
                 use_entity_type=True,
                 is_eval_data=False,
                 tokenizer='bert-base-uncased',
                 query_position='prefix',
                 argument_description=r'res/descriptions/argument_descriptions.json',
                 event_map='res/maps/event_map_without_other.json',
                 event_preds=None,
                 query_without_guide=False
                 ):
        super(RCArgumentPreprocessor, self).__init__(data_ref)
        if type(argument_query_template) is str:
            argument_query_template = getattr(qa, argument_query_template)
        self.query_template = argument_query_template()
        self.desc_type = desc_type
        self.use_entity = self.query_template.use_entity
        self.use_trigger = self.query_template.use_trigger

        self.negative_sampling = negative_sampling != 'all'
        if self.negative_sampling:
            self.negative_k = negative_sampling
        self.n_sample = n_sample
        self.few_shot_type = few_shot_type
        self.is_eval_data = is_eval_data
        self.event_preds = event_preds
        self.use_event_preds = self.is_eval_data and self.event_preds is not None
        if self.is_eval_data:
            # overwrite negative sampling setting
            self.negative_sampling = False
        self.use_entity_type = use_entity_type
        with open(argument_description, 'r', encoding='utf-8') as fin:
            self.argument_desc = json.load(fin)
        with open(event_map, 'r', encoding='utf-8') as fin:
            self.event_map = json.load(fin)
        self.rev_event_map = {i: ev for ev, i in self.event_map.items()}
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.query_position = query_position
        self.query_without_guide = query_without_guide
        self.tok_returns = list(self.tokenizer.encode_plus([1], [1]).keys())

    def preprocess(self):
        # Each instance is event/role/entity/sentence pair.
        print("Preparing argument maps...", )
        self.prepare_argument_maps()
        print("Done. ")
        # remove 'None'
        if 'None' in self.argument_map:
            self.argument_map.pop('None')
        self.argument_types = list(self.argument_map.keys())
        """
        Combined data should contain in each line:
        sentence, mask, event, entity, entity_type, argument_type, label
        """

        print("Tokenizing sentences...")
        sentences = utils.extract_property(
            self.data_ref.data, 'words'
        )
        self.sentences, self.token_pos_map = [], []
        for sent in sentences:
            token_ids = []
            index_map = {}
            for i, x in enumerate(sent):
                index_map[i] = len(token_ids)
                token_ids += self.tokenizer.encode(x, add_special_tokens=False)
            index_map[len(sent)] = len(token_ids)
            self.sentences.append(token_ids)
            self.token_pos_map.append(index_map)
        print("Done.")

        self.entities = utils.extract_property(
            self.data_ref.data,
            'entities'
        )
        self.events = utils.extract_property(
            self.data_ref.data,
            'event-mentions'
        )
        if not self.use_entity:
            self.prepare_entities()

        print("Preparing queries...", )
        data_, labels, self.id_tuples = self.prepare_queries(
            self.sentences, self.entities, self.events
        )
        self.labels = labels
        print("Done.")

        print("Combining...")
        self.combined_text = []
        self.pred_masks = []
        self.label_masks = []

        if not self.use_entity:
            self.entity_spans = np.zeros([len(data_), self.max_entity_num, 2], dtype=np.int)
            self.possible_entity_masks = np.zeros([len(data_), self.max_entity_num])
        total_outs = {x: [] for x in self.tok_returns}
        for di, entry in enumerate(tqdm(data_)):
            if self.use_entity:
                [sentence, event, entity, entity_type, argument_type, template, is_pred, is_label] = entry
                query_text = self.query_template.encode(
                    event, argument_type, entity, entity_type, template
                )
            else:
                [sentence, event, entity_span, possible_entity_mask, argument_type, template, trigger, is_pred, is_label] = entry
                if trigger is None:
                    query_text = self.query_template.encode(event, argument_type, None, None, template)
                else:
                    query_text = self.query_template.encode(event, argument_type, trigger)
                self.entity_spans[di] = entity_span
                self.possible_entity_masks[di, :len(possible_entity_mask)] = possible_entity_mask

            self.pred_masks.append(is_pred)
            self.label_masks.append(is_label)
            outs = {x: [] for x in self.tok_returns}
            query_text = self.tokenizer.encode(query_text, add_special_tokens=False)
            combined, start_index = self.combine_query_and_sentence(
                query_text, sentence
            )
            if not self.use_entity:
                self.entity_spans[di] += start_index
            for k, v in combined.items():
                outs[k].append(v)
            self.combined_text.append(combined)
            for x in outs:
                total_outs[x] += outs[x]
        if not self.use_entity:
            max_seq_len = max(map(lambda comb: len(comb['input_ids']), self.combined_text))
            argument_start_labels = np.zeros([len(self.combined_text), max_seq_len])
            argument_end_labels = np.zeros([len(self.combined_text), max_seq_len])
            for i, label in enumerate(labels):
                for ei, labl in enumerate(label):
                    if labl == 1:
                        argument_start_labels[i, self.entity_spans[i, ei, 0]] = 1
                        argument_end_labels[i, self.entity_spans[i, ei, 1]] = 1
            self.argument_start_labels = argument_start_labels
            self.argument_end_labels = argument_end_labels
        for x, y in total_outs.items():
            setattr(self, x, y)
        self.pad()
        for v in self.attributes().values():
            if type(getattr(self, v)) is list:
                setattr(self, v, np.array(getattr(self, v)))
        # pred_correct_masks = self.pred_masks & self.label_masks
        # TP = pred_correct_masks.sum()
        # FP = (self.pred_masks & ~pred_correct_masks).sum()
        # FN = (self.label_masks & ~pred_correct_masks).sum()
        # precision = TP / (TP + FP)
        # recall = TP / (TP + FN)
        # print(TP, FN, FP)
        # print('pr:', precision)
        # print('re:', recall)
        # print('f1:', 2*precision*recall/(precision+recall))
        print("Done.")

    def prepare_negative_samples_for_entity(
            self,
            sentence,
            event,
            entity,
            entity_type,
            exclude=None,
    ):
        if self.negative_sampling:
            neg_args = np.random.choice(
                self.argument_types,
                self.negative_k
            )
        else:
            neg_args = self.argument_types
        data_ = []
        for narg in neg_args:
            if ((exclude and narg != exclude) or
                    isinstance(exclude, list) and narg not in exclude
            ):
                continue
            data_.append(
                [sentence, event, entity, entity_type, narg, 0]
            )
        return data_

    def prepare_positive_samples_for_event(
            self,
            sentence,
            event,
            entities,
            entity_types,
            argument_label
    ):
        data_ = []
        for ent, etype, arg in zip(entities, entity_types, argument_label):
            data_.append(
                [sentence, event, ent, etype, arg, 1]
            )
        return data_

    @staticmethod
    def is_same_entity_type(ent_in_data, ent_in_guide):
        try:
            super_ent, sub_ent = ent_in_data.lower().split(":")
        except ValueError:
            super_ent, sub_ent = None, ent_in_data.lower()
        return ent_in_guide.lower() in (sub_ent, super_ent)

    def contain_entity_type(self, entity_type_list, entity_type):
        for ent in entity_type_list:
            if self.is_same_entity_type(entity_type, ent):
                return True
        return False

    def prepare_queries_for_event_with_entities(self, si, sentence, event_id, entity_mentions, argument_labels, is_pred=True, is_label=True):
        event = self.rev_event_map[event_id]
        event_supertype, event_subtype = event.split(':')
        args_desc = self.argument_desc[event_supertype][event_subtype]
        data = []
        label = []
        id_tuple = []
        for ei, (entity, argument_label_set) in enumerate(zip(entity_mentions, argument_labels)):
            entity_type = entity['entity-type']
            entity_text = entity['head']
            for argument_label in argument_label_set:
                if argument_label != 'None':
                    super_argument_role = argument_label.split('-')[0]
                    assert super_argument_role in args_desc.keys(), "Guide wrong! event {} doesn't have argument {}, detail: \n{}\n{}".format(
                        event, super_argument_role, args_desc.keys(), sentence)
            if self.query_without_guide:
                for arg, arg_id in self.argument_map.items():
                    data.append([sentence, event, entity_text, entity_type, arg, None, is_pred, is_label])
                    label.append(arg in argument_label_set)
                    id_tuple.append([si, event_id, ei, arg_id])
            else:
                for arg, desc in args_desc.items():
                    if self.contain_entity_type(desc['entity-type'], entity_type):
                        if self.desc_type is None:
                            template = None
                        else:
                            template = desc[self.desc_type]
                        if arg == 'Time':
                            for time_arg in ["Time-Holds", "Time-Starting", "Time-Ending", "Time-Before", "Time-Within", "Time-At-Beginning", "Time-After", "Time-At-End"]:
                                data.append([sentence, event, entity_text, entity_type, time_arg, template, is_pred, is_label])
                                label.append(time_arg in argument_label_set)
                                id_tuple.append([si, event_id, ei, self.argument_map[time_arg]])
                        else:
                            data.append([sentence, event, entity_text, entity_type, arg, template, is_pred, is_label])
                            label.append(arg in argument_label_set)
                            id_tuple.append([si, event_id, ei, self.argument_map[arg]])
        return data, label, id_tuple

    def prepare_queries_for_event_without_entities(self, si, sentence, event_id, entity_mention, entity_span, argument_labels, trigger=None, is_pred=True, is_label=True):
        event = self.rev_event_map[event_id]
        event_supertype, event_subtype = event.split(':')
        args_desc = self.argument_desc[event_supertype][event_subtype]
        data = []
        label = []
        id_tuple = []
        for arg, desc in args_desc.items():
            template = desc[self.desc_type] if self.desc_type is not None else None
            possible_entity_mask = [self.contain_entity_type(desc['entity-type'], entity['entity-type']) for entity in entity_mention]
            if arg == 'Time':
                for time_arg in ["Time-Holds", "Time-Starting", "Time-Ending", "Time-Before", "Time-Within", "Time-At-Beginning", "Time-After", "Time-At-End"]:
                    data.append([sentence, event, entity_span, possible_entity_mask, time_arg, template, trigger, is_pred, is_label])
                    label.append([argument_label == time_arg for argument_label in argument_labels])
                    id_tuple.append([si, event_id, self.argument_map[time_arg]])
            else:
                data.append([sentence, event, entity_span, possible_entity_mask, arg, template, trigger, is_pred, is_label])
                label.append([arg in argument_label_set for argument_label_set in argument_labels])
                id_tuple.append([si, event_id, self.argument_map[arg]])
        return data, label, id_tuple

    def prepare_queries(self, sentences, entities, events):
        data_ = []
        label = []
        id_tuples = []
        label_to_samp_idx = defaultdict(set)
        if self.use_trigger:
            # if self.is_eval_data and self.event_preds is not None:
            #     events = [[event for i, event in enumerate(event_mentions) if self.event_preds[si, self.event_map[event['event_type']]] == 1] for si, event_mentions in enumerate(events)]
            for si, (sentence, entity, event) in enumerate(
                    zip(sentences, entities, events)):
                s_data = []
                s_label = []
                s_id_tuple = []
                for evt in event:
                    ev = evt['event_type']
                    trigger = evt['trigger']['text']
                    arguments = evt['arguments']
                    for argument in arguments:
                        if argument != 'None':
                            if self.few_shot_type == 'argument':
                                label_to_samp_idx[argument].add(si)
                            elif self.few_shot_type == 'event-argument':
                                label_to_samp_idx[(ev, argument)].add(si)
                    queries_for_event, arg_label, id_tuple = \
                        self.prepare_queries_for_event_without_entities(si, sentence, self.event_map[ev], entity, self.entity_spans[si], arguments, trigger)
                    s_data.extend(queries_for_event)
                    s_label.extend(arg_label)
                    s_id_tuple.extend(id_tuple)
                data_.append(s_data)
                label.append(s_label)
                id_tuples.append(s_id_tuple)
        else:
            event_normalizes = [defaultdict(list) for _ in range(len(sentences))]
            event_label_mat = np.zeros([len(sentences), len(self.event_map)], dtype=np.bool)
            for si, event in enumerate(events):
                for evt in event:
                    ev, arguments = self.event_map[evt['event_type']], evt['arguments']
                    event_normalizes[si][ev] = [set() for _ in range(len(entities[si]))]
                    event_label_mat[si, ev] = True
                    for ai, argument in enumerate(arguments):
                        if argument != 'None':
                            event_normalizes[si][ev][ai].add(argument)
            if self.use_event_preds:
                # event_list = [[self.rev_event_map[i] for i, pred in enumerate(event) if pred == 1] for event in self.event_preds]
                combine_events = self.event_preds | event_label_mat
            else:
                # event_list = [set(evt['event_type'] for evt in event) for event in events]
                combine_events = event_label_mat
                self.event_preds = event_label_mat
            for si, (sentence, entity, pred_event, event, event_normalize) in enumerate(
                    zip(sentences, entities, self.event_preds, combine_events, event_normalizes)):
                s_data = []
                s_label = []
                s_id_tuple = []
                for ev_id, ev in enumerate(event):
                    if ev == 0:
                        continue
                    arguments = event_normalizes[si][ev_id] if ev_id in event_normalizes[si] \
                        else [set() for _ in range(len(entities[si]))]
                    for ai, argument_set in enumerate(arguments):
                        for argument in argument_set:
                            if argument != 'None':
                                if self.few_shot_type == 'argument':
                                    label_to_samp_idx[argument].add(si)
                                elif self.few_shot_type == 'event-argument':
                                    label_to_samp_idx[(ev_id, argument)].add(si)
                    if self.use_entity:
                        queries_for_event, arg_label, id_tuple = \
                            self.prepare_queries_for_event_with_entities(si, sentence, ev_id, entity, arguments, is_pred=pred_event[ev_id], is_label=ev_id in event_normalize)
                    else:
                        queries_for_event, arg_label, id_tuple = \
                            self.prepare_queries_for_event_without_entities(si, sentence, ev_id, entity, self.entity_spans[si], arguments, is_pred=pred_event[ev_id], is_label=ev_id in event_normalize)
                    s_data.extend(queries_for_event)
                    s_label.extend(arg_label)
                    s_id_tuple.extend(id_tuple)
                data_.append(s_data)
                label.append(s_label)
                id_tuples.append(s_id_tuple)
        if self.n_sample != 'all':
            samp_indices = list(set(
                [x for t in [np.random.choice(list(x), self.n_sample) for x in label_to_samp_idx.values() if len(x) > 0]
                 for x in t]))
            data_ = [data_[i] for i in samp_indices]
            label = [label[i] for i in samp_indices]
            id_tuples = [id_tuples[i] for i in samp_indices]
        data_ = [x for data in data_ for x in data]
        label = [x for labl in label for x in labl]
        id_tuples = [x for id_tuple in id_tuples for x in id_tuple]
        return data_, label, id_tuples

    def combine_query_and_sentence(self, query, sentence, cls='[CLS]', sep='[SEP]'):
        '''
        if self.query_position == 'prefix':
            return "{} {} {} {} {}".format(
                cls, query, sep, sentence, sep
            )
        elif self.query_position == "postfix":
            return "{} {} {} {} {}".format(
                cls, sentence, sep, query, sep
            )
        '''
        if self.query_position == 'prefix':
            out = self.tokenizer.encode_plus(query, sentence)
            start_index = len(query) + 2
        elif self.query_position == 'postfix':
            out = self.tokenizer.encode_plus(sentence, query)
            start_index = 1
        else:
            raise NotImplementedError("Query position {} is not implemented".format(self.query_position))
        return out, start_index

    def prepare_entities(self):
        self.max_entity_num = max(map(len, self.entities))
        entities = np.zeros([len(self.events), self.max_entity_num, 2], dtype=np.int)
        entity_masks = np.zeros([len(self.events), self.max_entity_num], dtype=np.bool)
        for i, (event_mentions, index_map, entity) in enumerate(
                zip(self.events, self.token_pos_map, self.entities)):
            entity_masks[i][:len(entity)] = True
            for j, en in enumerate(entity):
                entities[i, j] = index_map[en['start']], index_map[en['end']]
        self.entity_spans = entities
        self.entity_masks = entity_masks

    def tokenize(self):
        self.tokenized_text = [self.tokenizer.encode(x) for x in self.combined_text]

    def pad(self):
        # TODO: needs modified
        max_len = max(map(len, self.input_ids))
        self.masks_ = utils.pad_sequences_(self.input_ids, max_len)
        for x in self.tok_returns:
            if x == 'input_ids':
                continue
            self.masks = utils.pad_sequences_(
                getattr(self, x), max_len
            )
            assert (np.array(self.masks) != np.array(self.masks_)).sum() == 0
        if not self.use_entity:
            utils.pad_sequences_(self.labels, self.max_entity_num)
            utils.pad_sequences_(self.possible_entity_masks, self.max_entity_num)

    def attributes(self):
        attrs = {
            x: x for x in self.tok_returns
        }
        attrs.update({'query_labels': 'labels',
                      'id_tuples': 'id_tuples',
                      'tokenizer': 'tokenizer',
                      'pred_masks': 'pred_masks',
                      'label_masks': 'label_masks'})
        if not self.use_entity:
            attrs.update({'argument_start_labels': 'argument_start_labels',
                          'argument_end_labels': 'argument_end_labels',
                          'entity_spans': 'entity_spans',
                          'possible_entity_masks': 'possible_entity_masks'})
        return attrs


class RCArgumentWithoutEntitiesPreprocessor(ArgumentEntityPreprocessor):
    def __init__(self,
                 data_ref,
                 argument_query_template="ARG_EQ2",
                 event_label_processor='IdentityEventProcessor',
                 is_eval_data=False,
                 tokenizer='bert-base-uncased',
                 query_position='prefix',
                 mask_event_label=False,
                 ):
        super().__init__(data_ref)
        if type(argument_query_template) is str:
            argument_query_template = getattr(qa, argument_query_template)
        self.query_template = argument_query_template()
        assert not self.query_template.use_entity
        self.use_entity = self.query_template.use_entity

        self.tokenizer = Tokenizer(tokenizer)
        self.query_position = query_position
        self.mask_event_label = mask_event_label

        self.cls_idx = self.tokenizer.encode('[CLS]', add_special_tokens=False)[0]
        self.sep_idx = self.tokenizer.encode('[SEP]', add_special_tokens=False)[0]

    def pretokenize(self):
        self.sentences, self.token_pos_map = zip(*[
            encode_pretty(x,
                          lambda x: self.tokenizer.encode(x, add_special_tokens=False),
                          self.tokenizer.decode)
            for x in self.sentences
        ])

        self.event_id_map = {
            x: self.tokenizer.encode(x, add_special_tokens=False)
            for x in self.event_types
        }
        self.role_id_map = {
            x: self.tokenizer.encode(x, add_special_tokens=False)
            for x in self.role_types
        }

        self.pretokenize_query_template()

    def pretokenize_query_template(self):
        event_placeholder = '[EVENT]:[EVENT]'
        role_placeholder = '[ROLE]'
        empty_query = self.query_template.encode(
            event_placeholder, role_placeholder,
        )

        event_placeholder_idx = self.tokenizer.encode(
            event_placeholder, add_special_tokens=False
        )
        role_placeholder_idx = self.tokenizer.encode(
            role_placeholder, add_special_tokens=False
        )

        self.empty_query_idx = self.tokenizer.encode(empty_query, add_special_tokens=False)
        self.event_placeholder_loc = utils.find_subsequence(event_placeholder_idx, self.empty_query_idx)
        self.role_placeholder_loc = utils.find_subsequence(role_placeholder_idx, self.empty_query_idx)
        if self.mask_event_label:
            self.event_placeholder_idx = event_placeholder_idx

    def fill_query_template(self, event_label, role_label):
        event_idx = self.event_id_map[event_label]
        role_idx = self.role_id_map[role_label]
        query_idx = list(self.empty_query_idx)
        el, er = self.event_placeholder_loc
        rl, rr = self.role_placeholder_loc
        if self.mask_event_label:
            query_idx[el:er] = self.event_placeholder_idx
        else:
            query_idx[el:er] = event_idx
        query_idx[rl:rr] = role_idx
        return query_idx

    def generate_query_sentence_pair(self, sentence_idx, event_label, role_label):
        query_idx = self.fill_query_template(event_label, role_label)
        ret = [self.cls_idx]
        token_type_ids = [0]
        if self.query_position == 'prefix':
            ret += query_idx + [self.sep_idx]
            left_add = len(ret)
            ret += sentence_idx + [self.sep_idx]
            token_type_ids += [0] * (len(query_idx) + 1) + [1] * (len(sentence_idx) + 1)
        elif self.query_position == 'postfix':
            left_add = len(ret)
            ret += sentence_idx + [self.sep_idx]
            ret += query_idx + [self.sep_idx]
            token_type_ids += [0] * (len(sentence_idx) + 1) + [1] * (len(query_idx) + 1)
        else:
            raise NotImplementedError("Query position type {} is unknown.".format(self.query_position))
        return ret, token_type_ids, left_add

    def prepare_sentence_pairs(self):
        self.query_tokens = []
        self.token_type_ids = []
        self.starts = []
        self.ends = []
        self.left_adds = []
        self.indices = []
        for j, (sent, events, roles, entities) in enumerate(zip(
                self.sentences, self.sample_event_types, self.sample_role_types,
                self.sample_entities
        )):
            for i, event_ in enumerate(events):
                for role in self.role_types:
                    sent_, type_ids_, left_add_ = self.generate_query_sentence_pair(sent, event_, role)
                    self.left_adds.append(left_add_)
                    self.query_tokens.append(sent_)
                    self.token_type_ids.append(type_ids_)
                    self.indices.append(j)
                    if role in roles[i]:
                        idx = roles[i].index(role)
                        self.starts.append(left_add_ + self.token_pos_map[j][entities[idx]['start']][0])
                        self.ends.append(left_add_ + self.token_pos_map[j][entities[idx]['end']][0])
                    else:
                        self.starts.append(-1)
                        self.ends.append(-1)

    def preprocess(self):
        self.sentences = utils.extract_property(
            self.data_ref.data, "words"
        )

        self.sample_events = utils.extract_property(
            self.data_ref.data, "event-mentions"
        )
        self.sample_event_types = [[x['event_type'] for x in y] for y in self.sample_events]
        self.event_types = set([t for y in self.sample_event_types for t in y])

        self.sample_role_types = [
            [[p for p in x['arguments']] for x in y] for y in self.sample_events
        ]
        self.role_types = set([role for entry in self.sample_role_types for event in entry for role in event])
        self.role_types.remove('None')

        self.sample_entities = utils.extract_property(
            self.data_ref.data, "entities"
        )

        self.pretokenize()
        self.prepare_sentence_pairs()

        self.entity_lefts = [[self.token_pos_map[i][ent['start']][0] for ent in entities] for i, entities in
                             enumerate(self.sample_entities)]
        self.entity_rights = [[self.token_pos_map[i][ent['end']][0] for ent in entities] for i, entities in
                              enumerate(self.sample_entities)]

        self.pad()

    def pad(self):
        max_len = max(map(len, self.query_tokens))
        self.masks = utils.pad_sequences_(self.query_tokens, max_len)

    def attributes(self):
        return {
            'query_tokens': 'query_tokens',
            'query_masks': 'masks',
            'token_type_ids': 'token_type_ids',
            'starts': 'starts',
            'ends': 'ends',
            'entity_lefts': 'entity_lefts',
            'entity_rights': 'entity_rights',
            'left_adds': 'left_adds',
            'sentence_indices': 'indices',
            'tokenizer': 'tokenizer',
            'get_entity_start_pos': 'get_entity_start_pos',
            'get_entity_end_pos': 'get_entity_end_pos'
        }

    def get_entity_start_pos(self, sentence_idx, instance_idx):
        left_add = self.left_adds[instance_idx]
        return [x + left_add for x in self.entity_lefts[sentence_idx]]

    def get_entity_end_pos(self, sentence_idx, instance_idx):
        left_add = self.left_adds[instance_idx]
        return [x + left_add for x in self.entity_rights[sentence_idx]]


class RCEventPreprocessor(EventsBIOPreprocessor):
    def __init__(self, data_ref,
                 event_label_processor='IdentityEventProcessor',
                 event_query_template='EQ1',
                 use_event_description='none',
                 tokenizer='bert-base-uncased',
                 reverse_query=False,
                 nsamps='all',
                 max_desc_sentences=10,
                 ):
        super().__init__(data_ref, ignore_tags=True)

        self.event_label_processor = getattr(qa, event_label_processor)
        self.event_query_template = getattr(qa, event_query_template)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.use_event_description = use_event_description

        self.query_maker = qa.EventQueryMaker(
            self.event_label_processor,
            self.event_query_template,
            self.use_event_description
        )

        self.reverse_query = reverse_query
        self.nsamps = nsamps
        self.max_desc_sentences = max_desc_sentences

        self.tok_returns = list(self.tokenizer.encode_plus([1], [1]).keys())

    def tokenize_queries(self):
        self.event_to_query = {x: self.query_maker.encode(x, self.max_desc_sentences) for x in self.event_labels}
        self.event_to_query_idx = {
            x: self.tokenizer.encode(y, add_special_tokens=False)
            for x, y in self.event_to_query.items()
        }

    def pretokenize(self):
        self.sentences = [self.tokenizer.encode(x, add_special_tokens=False) for x in self.sentences]
        self.tokenize_queries()

    def prepare_queries_for_sentence(self, sentence, true_labels):
        outs = {x: [] for x in self.tok_returns}
        labels = []
        for event_label in self.event_labels:
            if self.reverse_query:
                out = self.tokenizer.encode_plus(sentence, self.event_to_query_idx[event_label])
            else:
                out = self.tokenizer.encode_plus(self.event_to_query_idx[event_label], sentence)
            for x in outs:
                outs[x].append(out[x])
            if event_label in true_labels:
                labels.append(1)
            else:
                labels.append(0)
        return outs, labels

    def prepare_queries(self):
        self.labels = []
        self.sentence_inst_idx_span = []
        outs = {x: [] for x in self.tok_returns}
        for isent, (sentence, event_labels) in enumerate(zip(
                self.sentences, self.raw_event_labels
        )):
            out, labels = self.prepare_queries_for_sentence(
                sentence, event_labels
            )
            self.sentence_inst_idx_span.append(
                (len(self.labels), len(self.labels) + len(labels))
            )
            self.labels += labels
            for x in outs:
                outs[x] += out[x]
        for x, y in outs.items():
            setattr(self, x, y)

    def preprocess(self):
        self.prepare_event_tag_maps()
        if 'Other' in self.event_tag_map:
            self.event_tag_map.pop('Other')
        self.event_labels = list(self.event_tag_map.keys())

        self.sentences = utils.extract_property(
            self.data_ref.data, "words"
        )
        self.sentences = [' '.join(x) for x in self.sentences]
        self.raw_event_labels = [
            [x['event_type'] for x in y['event-mentions']] for y in self.data_ref.data
        ]

        # do few-shot data sampling
        if self.nsamps != 'all':
            label_to_samp_idx = {x: [] for x in self.event_labels}
            for ix, label in enumerate(self.raw_event_labels):
                for l in label:
                    label_to_samp_idx[l].append(ix)

            samp_indices = list(set(
                [x for t in [np.random.choice(x, self.nsamps) for x in label_to_samp_idx.values() if len(x) > 0] for x
                 in t]))
            self.sentences = [self.sentences[i] for i in samp_indices]
            self.raw_event_labels = [self.raw_event_labels[i] for i in samp_indices]

        self.pretokenize()
        self.prepare_queries()
        self.pad()

        for v in self.attributes().values():
            if type(getattr(self, v)) is list:
                setattr(self, v, np.array(getattr(self, v)))

    def pad(self):
        max_len = max(map(len, self.input_ids))
        self.masks_ = utils.pad_sequences_(self.input_ids, max_len)
        for x in self.tok_returns:
            if x == 'input_ids':
                continue
            self.masks = utils.pad_sequences_(
                getattr(self, x), max_len
            )
            assert (np.array(self.masks) != np.array(self.masks_)).sum() == 0

    def attributes(self):
        attrs = {
            x: x for x in self.tok_returns
        }
        attrs['labels'] = 'labels'
        attrs['tokenizer'] = 'tokenizer'
        attrs['sentence_inst_idx_span'] = 'sentence_inst_idx_span'
        return attrs


class RCEventSpanPreprocessor(EventsBIOPreprocessor):
    def __init__(self, data_ref,
                 event_label_processor='IdentityEventProcessor',
                 event_query_template='EQ1',
                 use_event_description='none',
                 tokenizer='bert-base-uncased',
                 reverse_query=False,
                 nsamps='all',
                 max_desc_sentences=10,
                 negative_sampling='all'
                 ):
        super().__init__(data_ref, ignore_tags=True)
        self.event_label_processor = getattr(qa, event_label_processor)
        self.event_query_template = getattr(qa, event_query_template)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.use_event_description = use_event_description

        self.query_maker = qa.EventQueryMaker(
            self.event_label_processor,
            self.event_query_template,
            self.use_event_description
        )

        self.reverse_query = reverse_query
        self.nsamps = nsamps
        self.max_desc_sentences = max_desc_sentences

        self.tok_returns = list(self.tokenizer.encode_plus([1], [1]).keys())

    def tokenize_queries(self):
        self.event_to_query = {x: self.query_maker.encode(x, self.max_desc_sentences) for x in self.event_labels}
        self.event_to_query_idx = {
            x: self.tokenizer.encode(y, add_special_tokens=False)
            for x, y in self.event_to_query.items()
        }

    def pretokenize(self):
        self.tokenize_queries()
        """
        Pretokenize sentences and add label indexes. 
        """
        self.sent_token_ids = []
        self.sent_span_ids = []
        self.true_events = []
        self.sent_token_head_ids = []
        # after processing: 
        #   1. each item in sent_token_ids is tokenized sentence, 
        #   2. each item in sent_span_ids is a list of (span start idx, span end idx)
        #       for each event in the sentence.
        #   3. each item in true_events is a list of event indices for events
        #       in the sentence.
        for sent, raw_event in zip(self.sentences, self.raw_events):
            self.true_events.append(
                [self.event_tag_map[x] for x in [y['event_type'] for y in raw_event]]
            )
            token_ids = []
            og_lefts = [x['start'] for x in [y['trigger'] for y in raw_event]]
            og_rights = [x['end'] for x in [y['trigger'] for y in raw_event]]
            index_map = {}
            for i, x in enumerate(sent):
                index_map[i] = len(token_ids)
                token_ids += self.tokenizer.encode(x, add_special_tokens=False)
            og_lefts = [index_map[x] for x in og_lefts]
            index_map[len(sent)] = len(token_ids)
            og_rights = [index_map[x] for x in og_rights]
            self.sent_span_ids.append(
                list(zip(og_lefts, og_rights))
            )
            self.sent_token_ids.append(token_ids)
            self.sent_token_head_ids.append(list(index_map.values()))

    def prepare_queries_for_sentence(self, index):
        outs = {x: [] for x in self.tok_returns}
        token_ids = self.sent_token_ids[index]
        true_events = self.true_events[index]
        span_ids = self.sent_span_ids[index]
        span_start_labels = []
        span_end_labels = []
        token_head_ids = []

        def encode_plus_with_start_pos(sentence, event_id):
            query_ids = self.event_to_query_idx[self.event_labels[event_id]]
            if self.reverse_query:
                out = self.tokenizer.encode_plus(sentence, query_ids)
                start_index = 1
            else:
                out = self.tokenizer.encode_plus(query_ids, sentence)
                start_index = len(query_ids) + 2
            return out, start_index

        # add for true events
        for ie, (event_id, span) in enumerate(zip(true_events, span_ids)):
            out, start_idx = encode_plus_with_start_pos(token_ids, event_id)
            for k, v in out.items():
                outs[k].append(v)
            starts = [0 for _ in out['input_ids']]
            ends = [0 for _ in out['input_ids']]
            starts[span[0] + start_idx] = 1
            ends[span[1] + start_idx] = 1
            span_start_labels.append(starts)
            span_end_labels.append(ends)
            token_head_ids.append(
                [x + start_idx for x in self.sent_token_head_ids[index]]
            )

        for event_id in range(len(self.event_labels)):
            if event_id in true_events:
                continue
            out, start_idx = encode_plus_with_start_pos(token_ids, event_id)
            for k, v in out.items():
                outs[k].append(v)
            starts = [0 for _ in out['input_ids']]
            ends = [0 for _ in out['input_ids']]
            span_start_labels.append(starts)
            span_end_labels.append(ends)
            token_head_ids.append(
                [x + start_idx for x in self.sent_token_head_ids[index]]
            )

        return outs, span_start_labels, span_end_labels, token_head_ids

    def prepare_queries(self):
        self.start_labels = []
        self.end_labels = []
        self.sentence_inst_idx_span = []
        self.token_head_ids = []
        outs = {x: [] for x in self.tok_returns}
        for isent in range(len(self.sentences)):
            out, start_labels, end_labels, token_head_ids = self.prepare_queries_for_sentence(
                isent
            )
            self.sentence_inst_idx_span.append(
                (len(self.start_labels), len(self.start_labels) + len(start_labels))
            )
            self.start_labels += start_labels
            self.end_labels += end_labels
            self.token_head_ids += token_head_ids
            for x in outs:
                outs[x] += out[x]
        for x, y in outs.items():
            setattr(self, x, y)

    def preprocess(self):
        self.prepare_event_tag_maps()
        if 'Other' in self.event_tag_map:
            self.event_tag_map.pop('Other')
        self.event_labels, _ = zip(*sorted(
            tuple(self.event_tag_map.items()), key=lambda x: x[1]
        ))
        self.event_tag_map = {x: i for i, x in enumerate(self.event_labels)}

        self.sentences = utils.extract_property(
            self.data_ref.data, "words"
        )
        self.raw_events = [
            y['event-mentions'] for y in self.data_ref.data
        ]

        # do few-shot data sampling
        if self.nsamps != 'all':
            label_to_samp_idx = {x: [] for x in self.event_labels}
            for ix, event in enumerate(self.raw_events):
                label = [x['event_type'] for x in event]
                for l in label:
                    label_to_samp_idx[l].append(ix)

            samp_indices = list(set(
                [x for t in [np.random.choice(x, self.nsamps) for x in label_to_samp_idx.values() if len(x) > 0] for x
                 in t]))
            self.sentences = [self.sentences[i] for i in samp_indices]
            self.raw_events = [self.raw_events[i] for i in samp_indices]

        self.pretokenize()
        self.prepare_queries()
        self.pad()

        for v in self.attributes().values():
            if type(getattr(self, v)) is list:
                setattr(self, v, np.array(getattr(self, v)))

    def pad(self):
        # TODO: needs modified
        max_len = max(map(len, self.input_ids))
        self.masks_ = utils.pad_sequences_(self.input_ids, max_len)
        for x in self.tok_returns:
            if x == 'input_ids':
                continue
            self.masks = utils.pad_sequences_(
                getattr(self, x), max_len
            )
            assert (np.array(self.masks) != np.array(self.masks_)).sum() == 0
        utils.pad_sequences_(self.start_labels, max_len)
        utils.pad_sequences_(self.end_labels, max_len)
        utils.pad_sequences_(
            self.token_head_ids,
            max(map(len, self.token_head_ids))
        )

    def attributes(self):
        # TODO: needs modified
        attrs = {
            x: x for x in self.tok_returns
        }
        attrs['start_labels'] = 'start_labels'
        attrs['end_labels'] = 'end_labels'
        attrs['tokenizer'] = 'tokenizer'
        attrs['sentence_inst_idx_span'] = 'sentence_inst_idx_span'
        attrs['token_head_ids'] = 'token_head_ids'
        return attrs

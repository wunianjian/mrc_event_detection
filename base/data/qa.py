from abc import abstractmethod, abstractproperty
import re
import json
from typing import Union
import nltk

norm_map = {
    "Start": "Starting",
    "End": "Ending",
    "Declare": "Declaration"
}

#{"Target": 0, "Org": 1, "Victim": 2, "Seller": 3, "Artifact": 4, "Beneficiary": 5, "Time-Holds": 6, "Place": 7, "Money": 8, "Defendant": 9, "Time-Starting": 10, "Time-Ending": 11, "Time-Before": 12, "Entity": 13, "Time-Within": 14, "Adjudicator": 15, "Plaintiff": 16, "Buyer": 17, "Instrument": 18, "Attacker": 19, "Crime": 20, "Position": 21, "Time-At-Beginning": 22, "Recipient": 23, "Vehicle": 24, "Destination": 25, "Origin": 26, "Agent": 27, "Giver": 28, "Person": 29, "Time-After": 30, "Time-At-End": 31, "Price": 32, "Sentence": 33, "Prosecutor": 34}



def process_event_type(event_type):
    event_type = event_type.split("-")
    for i, x in enumerate(event_type):
        if x == "Org":
            event_type[i] = "Organization"
    if len(event_type) == 1:
        return event_type[0]
    assert len(event_type) == 2
    x = event_type[0]
    if x == "Be":
        return " ".join(event_type)
    if x in ["Transfer", "Merge"]:
        return " of ".join(event_type)
    if x in ["Start", "End", "Declare"]:
        return norm_map[x] + " of " + event_type[1]
    else:
        return " and ".join(event_type)


def depunctuate(event_label):
    if event_label == "Other":
        return event_label
    event_label = event_label.split(":")
    event_label[1] = process_event_type(event_label[1])
    return ":".join(event_label)


with open("res/descriptions/event_descriptions.json", "r") as f:
    event_descriptions = json.load(f)


class EventQueryTemplate:
    @property
    def nary(cls):
        return cls._nary

    @abstractmethod
    def encode(self, event_label):
        raise NotImplementedError

    def __call__(self, event_label):
        return self.encode(event_label)

    def split_label(self, event_label):
        split = event_label.replace(' ', '').split(':')
        assert len(split) == 2, "{} is not split by : into two parts".format(event_label)
        return split


class EventLabelProcessor:
    @abstractmethod
    def encode(self, event_label):
        raise NotImplementedError

    def split_label(self, event_label):
        split = event_label.replace(' ', '').split(':')
        assert len(split) == 2, "{} is not split by : into two parts".format(event_label)
        return split

    def depunctuate(self, event_label):
        if event_label == "Other":
            return event_label
        event_label = event_label.split(":")
        event_label[1] = process_event_type(event_label[1])
        return ":".join(event_label)


class IdentityEventProcessor(EventLabelProcessor):
    def encode(self, event_label):
        return event_label

class SubtypeProcessor(EventLabelProcessor):
    def encode(self, event_label):
        return event_label.split(":")[1]


class ComaProcessor(EventLabelProcessor):
    def encode(self, event_label):
        event_label = self.depunctuate(event_label)
        hypernym, subtype = self.split_label(event_label)
        return ", ".join([hypernym, subtype])


class ConjProcessor(EventLabelProcessor):
    def encode(self, event_label):
        event_label = self.depunctuate(event_label)
        hypernym, subtype = self.split_label(event_label)
        return ", and".join([hypernym, subtype])


class ProgProcessor(EventLabelProcessor):
    def encode(self, event_label):
        event_label = self.depunctuate(event_label)
        hypernym, subtype = self.split_label(event_label)
        return ", and moreover, ".join([hypernym, subtype])


class IdentityQuery(EventQueryTemplate):
    _nary = 1

    def __init__(self, token_only=True):
        self.token_only = token_only

    def encode(self, event_label):
        if self.token_only:
            return depunctuate(event_label)


class EQ1(EventQueryTemplate):
    def encode(self, event_label):
        return "Did any event about {} happen?".format(event_label)

class SpanEQ1(EventQueryTemplate):
    def encode(self, event_label):
        return "Find the trigger for event {}.".format(event_label)


class ST1(EventQueryTemplate):
    _nary = 2

    def encode(self, event_label):
        return "Hence, an event about {} happened.".format(event_label)


class ST2(EventQueryTemplate):
    _nary = 2

    def encode(self, event_label):
        return "Hence, something about {} happened.".format(event_label)


class ArgumentQueryTemplate(EventQueryTemplate):
    _use_entity = False
    _use_trigger = False

    @property
    def use_entity(self):
        return self._use_entity

    @property
    def use_trigger(self):
        return self._use_trigger



class IdentityArgumentQuery(ArgumentQueryTemplate):
    _use_entity = True

    def encode(self, event_label, role, entity, entity_type):
        event_label = IdentityQuery()(depunctuate(event_label))
        query = event_label + ", "
        adds = [role, entity, entity_type]
        for add in adds:
            if add is not None:
                query += add + ", "
        return query[:-2]


class ARG_EQ1(ArgumentQueryTemplate):
    _use_entity = True

    def encode(self, event_label, role, entity, entity_type=None, template=None):
        event_label = depunctuate(event_label)
        hypernym, event = self.split_label(event_label)
        if entity_type is None:
            return "Did {} participate as role {} in event {}?".format(entity, role, IdentityQuery()(event_label))
        else:
            return "Did {}, {} participate as role {} in event {}?".format(entity, entity_type, role, IdentityQuery()(event_label))


class ARG_EQ2(ArgumentQueryTemplate):
    _use_entity = False
    def encode(self, event_label, role=None, entity=None, entity_type=None, template=None):
        event_label = depunctuate(event_label)
        hypernym, event = self.split_label(event_label)
        return "What is the {} in event {}?".format(role, IdentityQuery()(event_label))

class ARG_EQ_TRI_1(ArgumentQueryTemplate):
    _use_entity = False
    _use_trigger = True
    def encode(self, event_label, role, trigger_word):
        event_label = depunctuate(event_label)
        hypernym, event = self.split_label(event_label)
        return 'What is the {} in event {} triggered by "{}"?'.format(role, IdentityQuery()(event_label), trigger_word)

class ARG_EQ_TRI_2(ArgumentQueryTemplate):
    _use_entity = False
    _use_trigger = True
    def encode(self, event_label, role, trigger_word):
        event_label = depunctuate(event_label)
        hypernym, event = self.split_label(event_label)
        return 'What is the {} in "{}" ?'.format(role, trigger_word)


class ARG_DESC_WITH_EN(ArgumentQueryTemplate):
    _use_entity = True

    def encode(self, event_label, role=None, entity=None, entity_type=None, template=None):
        return template if entity is None else template.format(entity)


class ARG_DESC_WITHOUT_EN(ArgumentQueryTemplate):
    _use_entity = False

    def encode(self, event_label, role=None, entity=None, entity_type=None, template=None):
        return template if entity is None else template.format(entity)


class EventQueryMaker:
    def __init__(self,
                 event_label_processor: EventLabelProcessor,
                 event_query_template: EventQueryTemplate,
                 use_event_descrption: str
                 ):

        assert use_event_descrption in ['none', 'prefix', 'postfix']
        self.event_label_processor = event_label_processor()
        self.event_query_template = event_query_template()
        self.use_event_descrption = use_event_descrption

    def encode(self, event_label, max_desc_sentences=None):
        event_label_processed = self.event_label_processor.encode(event_label)
        if self.use_event_descrption == 'none':
            desc = ''
        else:
            try:
                desc = event_descriptions[event_label.split(':')[1]].replace("{}", event_label_processed)
            except KeyError:
                raise KeyError("{} is not in {}".format(
                    event_label_processed,
                    list(event_descriptions.keys())
                ))
            if max_desc_sentences:
                desc = ' '.join(nltk.sent_tokenize(desc)[:max_desc_sentences])
        query_text = self.event_query_template.encode(event_label_processed)
        if self.use_event_descrption == 'prefix':
            query_text = desc + " " + query_text
        elif self.use_event_descrption == 'postfix':
            query_text = query_text + " " + desc
        return query_text

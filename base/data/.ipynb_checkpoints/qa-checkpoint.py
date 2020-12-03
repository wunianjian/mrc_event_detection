from abc import abstractmethod, abstractproperty
import re

norm_map = {
    "Start": "Starting",
    "End": "Ending",
    "Declare": "Declaration"
}

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

class EQ2(EventQueryTemplate):
    def encode(self, event_label):
        return "Did any event on {} happen involving {}?".format(event_label)

class EQ3(EventQueryTemplate):
    def encode(self, event_label):
        return "Did {} happen?".format(event_label)

class EQ4(EventQueryTemplate):
    _nary = 1
    def encode(self, event_label):
        return "Did any event happen that's related to {}?".format(event_label)

class EQ5(EventQueryTemplate):
    _nary = 2
    def encode(self, event_label):
        return "Did any event happen that's related to {}, and more specifically, {}?".format(hypernym, event)

class EQ6(EventQueryTemplate):
    _nary = 2
    def encode(self, event_label):
        event_label = depunctuate(event_label)
        hypernym, event = self.split_label(event_label)
        return "Did {}, or more specifically, {} happen?".format(hypernym, event)

class EQ7(EventQueryTemplate):
    _nary = 2
    def encode(self, event_label):
        event_label = depunctuate(event_label)
        hypernym, event = self.split_label(event_label)
        return "Did any event related to {} and {} happen?".format(hypernym, event)

class EQ8(EventQueryTemplate):
    _nary = 1
    def encode(self, event_label):
        event_label = depunctuate(event_label)
        return "Find the event {}.".format(event_label)

class EQ9(EventQueryTemplate):
    _nary = 2
    def encode(self, event_label):
        event_label = depunctuate(event_label)
        hypernym, event = self.split_label(event_label)
        return "Find the event {} and {}.".format(hypernym, event)

class EQ10(EventQueryTemplate):
    _nary = 2
    def encode(self, event_label):
        event_label = depunctuate(event_label)
        return "Find {}.".format(event_label)

class ST1(EventQueryTemplate):
    _nary = 2
    def encode(self, event_label):
        return "Hence, an event about {} happened.".format(event_label)

class ST2(EventQueryTemplate):
    _nary = 2
    def encode(self, event_label):
        return "Hence, something about {} happened.".format(event_label)

class ArgumentQueryTemplate(EventQueryTemplate):
    @property
    def use_entity(self):
        return self._use_entity

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
    def encode(self, event_label, role, entity, entity_type=None):
        event_label = depunctuate(event_label)
        hypernym, event = self.split_label(event_label)
        if entity_type is None:
            return "Did {} participate as role {} in event {}?".format(entity, role, IdentityQuery()(event_label))
        else:
            return "Did {}, {} participate as role {} in event {}?".format(entity, entity_type, role, IdentityQuery()(event_label))

class ARG_EQ2(ArgumentQueryTemplate):
    _use_entity = False
    def encode(self, event_label, role, entity=None):
        event_label = depunctuate(event_label)
        hypernym, event = self.split_label(event_label)
        return "Who or what participated as role {} in event {}?".format(role, IdentityQuery()(event_label))
from abc import ABC, abstractmethod
from collections.abc import Iterable
from argparse import Namespace

import seqeval.metrics as metrics
import sklearn.metrics as sk_metrics
import torch
import numpy


class DefaultParamsMixin:
    def __init__(self):
        pass

    def safely_set_attribute(self, attribute, default_value, assertion=None):
        assert hasattr(self, 'params')
        if attribute in self.hparams:
            setattr(self.params, attribute, getattr(self.hparams, attribute))
        else:
            setattr(self.params, attribute, default_value)
        if assertion is not None:
            if type(assertion) is not list:
                assertion = [assertion]
            assert getattr(self.params, attribute) in assertion

    def safely_set_attributes_by_dict(self, attr_dict):
        # TODO: type assertion support?
        for key, value in attr_dict.items():
            self.safely_set_attribute(key, value)

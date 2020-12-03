import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import pytorch_lightning as pl

import base.data.dataset as dataset
from torch.utils.data import DataLoader, Sampler
from base.data.processor import *
import argparse
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from base.nn.utils import DefaultParamsMixin, SequenceTruncateMixin, SequenceBatcherMixin
import base.nn.modules.classifier as classifier
from base.nn import *
import os
import numpy as np

class EventLinearModel(
    pl.LightningModule,
    DefaultParamsMixin,
    SequenceBatcherMixin):

    def __init__(self, args):
        super().__init__()

        # self.params = args
        self.args = args
        self.initialize_params()
        self.initialize_dataset()
        self.initialize_models()
        self.initialize_evaluators()

    def initialize_params(self):
        self.params = argparse.Namespace()
        self.safely_set_attributes_by_dict({
            'input_train': 'data/train_process.json',
            'input_val': 'data/dev_process.json',
            'input_test': 'data/test_process.json',
            'batch_size': 32,
            'nevents': 34,
            'bert_lr': 5e-5,
            'cls_lr': 5e-5,
            'nepochs': 15,

            'event_distill_mode': 'bert',
            'consistency_type': 'binary_cross_entropy',
            'consistency_weight': 0.0,
            'event_matrix_norm': False,
            'pretrain_model': 'bert-base-uncased',
            'classifier_class': 'EventTokenClassifier'
        })

        self.classifier_params = []
        self.transformer_params = []

    def initialize_models(self):
        self.bert = transformers.AutoModel.from_pretrained(self.params.pretrain_model)
        self.params.nhid = self.bert.embeddings.word_embeddings.weight.shape[-1]
        self.classifier = classifier.LinearClassifier(self.params)

        self.bert_params = self.bert.parameters()
        self.cls_params = self.classifier.parameters()

    def batch_states_values(self):
        return ['sentences', 'masks',
             'event_labels', 'event_label_masks', 
             'event_matrices']
    
    def sequence_mask_names(self):
        return [['sentences', 'event_labels'],
                ['masks', 'event_label_masks']]

    def initialize_dataset(self):
        dataset_class = dataset.create_dataset_template(
            {TokensPreprocessor: {},
             EventsBIOPreprocessor: {},
             EventTokensPreprocessor: {}},
            self.batch_states_values(),
            'sentences'
        )
        self.train_data = dataset_class(self.params.input_train)
        self.val_data = dataset_class(self.params.input_val)
        self.test_data = dataset_class(self.params.input_test)

        self.event_tokens = self.train_data.event_tokens
        self.event_token_masks = self.train_data.event_token_masks

        self.params.ntypes = len(self.train_data.preprocessors[EventsBIOPreprocessor].event_tag_map)

    def initialize_evaluators(self):
        self.batch = {'train': None, 'test': None, 'val': None}
        self.evaluators = {
            'train': AccumEvaluator(),
            'test': AccumEvaluator(),
            'val': AccumEvaluator()
        }

    def configure_optimizers(self):
        optimizer = transformers.AdamW([
            {'params': self.bert_params, 'lr': self.params.bert_lr},
            {'params': self.cls_params, 'lr': self.params.cls_lr}
        ])

        return {'optimizer': optimizer,
                'lr_scheduler': lr_scheduler.CosineAnnealingLR(optimizer,
                     T_max=self.params.nepochs)}

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True, num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.params.batch_size, shuffle=False, num_workers=10)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.params.batch_size, shuffle=False, num_workers=10)

    def prepare_data(self):
        # already prepared in initialize_dataset; nothing left to do.
        pass

    def forward(self, batch, mode):
        batch.tokens_embed, _ = self.bert(input_ids=batch.sentences,
                                                 attention_mask=batch.masks)

        event_tokens = torch.tensor(self.event_tokens).to(self.bert.device)
        event_token_masks = torch.tensor(self.event_token_masks).to(self.bert.device)
        batch.events_embed, _ = self.bert(input_ids=event_tokens,
                                                 attention_mask=event_token_masks)
        #batch.events_embed = batch.events_embed
        logits, event_attns = self.classifier(batch)

        #batch.consistency_loss = self.consistency_loss(event_attns, batch.event_matrices)
        batch.consistency_loss = 0.0

        label = batch.event_labels.flatten(0, 1)
        mask = batch.event_label_masks.flatten(0, 1)
        loss_cls = F.cross_entropy(logits.flatten(0, 1), label, reduction='none')
        batch.loss_cls = (loss_cls * mask).sum() / mask.sum()
        batch.logits = logits

    def step(self, batch, batch_idx, mode='train'):
        batch = self.process_batch(batch, mode)
        self(batch, mode)
        loss = batch.loss_cls + self.params.consistency_weight * batch.consistency_loss
        preds = batch.logits.argmax(-1)
        rev_event_tag_map = self.train_data.preprocessors[EventsBIOPreprocessor].rev_event_tag_map

        evaluator = self.evaluators[mode]
        evaluator.accumulate(
            [[rev_event_tag_map[int(x)] for x in y] for y in preds],
            [[rev_event_tag_map[int(x)] for x in y] for y in batch.event_labels],
            [[rev_event_tag_map[int(x)] for x in y] for y in batch.event_label_masks],
        )
        return loss, batch

    def training_epoch_end(self, outputs):
        evaluator = self.evaluators['train']
        logger = {
            'train_pr': evaluator.metric('precision', self.params.batch_size * 5),
            'train_re': evaluator.metric('recall', self.params.batch_size * 5),
            'train_f1': evaluator.metric('f1', self.params.batch_size * 5)}
        display = {x: logger[x] for x in ['train_f1']}
        self.evaluators['train'].restart()
        return {'log': logger, 'progress_bar': display}

    def training_step(self, batch, batch_idx):
        loss, batch = self.step(batch, batch_idx, 'train')
        evaluator = self.evaluators['train']
        logger = {'train_loss_cls': batch.loss_cls,
                  'train_loss_con': batch.consistency_loss}
        display = {x: logger[x] for x in ['train_loss_cls', 'train_loss_con']}
            
        return {'loss': loss, 'log': logger, 'progress_bar': display}

    def validation_step(self, batch, batch_idx):
        loss, batch = self.step(batch, batch_idx, 'val')
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        loss, batch = self.step(batch, batch_idx, 'test')
        return {'test_loss': loss}

    def validation_epoch_end(self, outputs):
        loss = torch.mean(self.gather_outputs(outputs, 'val_loss'))
        logger = {'val_loss': loss}
        evaluator = self.evaluators['val']
        logger['val_pr'] = torch.tensor(evaluator.metric('precision'))
        logger['val_re'] = torch.tensor(evaluator.metric('recall'))
        logger['val_f1'] = torch.tensor(evaluator.metric('f1'))
        self.evaluators['val'].restart()

        return {'progress_bar': logger, 'log': logger}

    def test_epoch_end(self, outputs):
        loss = torch.mean(self.gather_outputs(outputs, 'test_loss'))
        logger = {'test_loss': loss}
        evaluator = self.evaluators['test']
        
        logger['test_pr'] = evaluator.metric('precision')
        logger['test_re'] = evaluator.metric('recall')
        logger['test_f1'] = evaluator.metric('f1')
        self.evaluators['test'].restart()
        return {'progress_bar': logger, 'log': logger}

    def gather_outputs(self, outputs, keyword):
        target = [x[keyword] for x in outputs]
        if len(target[0].shape) == 0:
            return torch.stack(target, 0)
        target = torch.cat(target, 0)
        return target

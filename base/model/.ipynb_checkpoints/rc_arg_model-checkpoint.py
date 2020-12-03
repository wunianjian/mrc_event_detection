from base.nn import *
from base.nn.utils import *
import torch
import torch.nn
import torch.nn.functional as F
import base.data.qa as qa
import base.data.dataset as dataset
from base.data.processor import RCArgumentPreprocessor
import pickle

class RCArgumentModel(
    pl.LightningModule,
    DefaultParamsMixin,
    SequenceTruncateMixin):
    """ Sentence-level Bert based model for event detection.
    """

    def __init__(self, args):
        super(RCArgumentModel, self).__init__()
        if type(args) is dict:
            args = Namespace(**args)

        self.hparams= args
        self.initialize_params()
        self.initialize_dataset()
        self.initialize_models()
        self.initialize_evaluators()

    def initialize_params(self):
        self.params = Namespace()
        self.safely_set_attributes_by_dict({
            'input_train': 'ACE05/train_process.json',
            'input_val': 'ACE05/dev_process.json',
            'input_test': 'ACE05/test_process.json',
            'input_train_processed': None,
            'input_val_processed': None,
            'input_test_processed': None,
            'batch_size': 16,
            'nevents': 33,
            'bert_lr': 2e-5,
            'cls_lr': 1e-4,
            'lr': 2e-5,
            'nepochs': 25,
            'query_template': "ARG_EQ1",
            'pretrain_model': 'bert-base-uncased',
            'head_num': 12,
            'reverse_query': False,
            'negative_sampling': 'all'
        })
        print('model params:')
        print(self.params)

    def initialize_models(self):
        self.bert = transformers.AutoModel.from_pretrained(self.params.pretrain_model)
        self.params.nhid = self.bert.embeddings.word_embeddings.weight.shape[-1]
        self.linear = nn.Linear(self.params.nhid, 1)
        self.criterion = F.binary_cross_entropy_with_logits

    def initialize_dataset(self):
        dataset_train_cls = dataset.create_dataset_template({
                RCArgumentPreprocessor: {
                    'argument_query_template': getattr(qa, self.params.query_template),
                    'negative_sampling': self.params.negative_sampling,
                    'use_entity_type': True,
                    'is_eval_data': False
                }
            },
            returns=['query_tokens', 'query_masks', 'query_labels'],
            length_by='query_tokens'
        )
        dataset_eval_cls = dataset.create_dataset_template({
                RCArgumentPreprocessor: {
                    'argument_query_template': getattr(qa, self.params.query_template),
                    'negative_sampling': self.params.negative_sampling,
                    'use_entity_type': True,
                    'is_eval_data': True
                }
            },
            returns=['query_tokens', 'query_masks', 'query_labels'],
            length_by='query_tokens'
        )

        if self.params.input_train_processed:
            with open(self.params.input_train_processed, 'rb') as f:
                self.train_data = pickle.load(f)
        else:
            self.train_data = dataset_train_cls(
                self.params.input_train
            )
        if self.params.input_val_processed:
            with open(self.params.input_val_processed, 'rb') as f:
                self.val_data = pickle.load(f)
        else:
            self.val_data = dataset_eval_cls(
                self.params.input_val
            )
        if self.params.input_test_processed:
            with open(self.params.input_test_processed, 'rb') as f:
                self.test_data = pickle.load(f)
        else:
            self.test_data = dataset_eval_cls(
                self.params.input_test,
            )

        self.sep_token_idx = self.train_data.tokenizer.encode(
            '[SEP]', add_special_tokens=False)[0]
        self.cls_token_idx = self.train_data.tokenizer.encode(
            '[CLS]', add_special_tokens=False)[0]

    def initialize_evaluators(self):
        self.batch_states = {'train': None, 'test': None, 'val': None}
        self.event_evaluators = {
            'train': SimpleAccumEvaluator(),
            'test': SimpleAccumEvaluator(),
            'val': SimpleAccumEvaluator()
        }
        self.extra_event_evaluators = {
            'train': SimpleAccumEvaluator(),
            'test': SimpleAccumEvaluator(),
            'val': SimpleAccumEvaluator()
        }
        self.argument_evaluators = {
            'train': SimpleAccumEvaluator(),
            'test': SimpleAccumEvaluator(),
            'val': SimpleAccumEvaluator()
        }

    def configure_optimizers(self):
        params_list = [
            {'params': self.bert.parameters(), 'lr': self.params.bert_lr},
            {'params': self.linear.parameters(), 'lr': self.params.cls_lr}
        ]

        optimizer = transformers.AdamW(params_list)
        return {'optimizer': optimizer,
                'lr_scheduler': 
                    lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=self.params.nepochs
                    )
                }

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True, num_workers=48)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.params.batch_size, shuffle=False, num_workers=48)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.params.batch_size, shuffle=False, num_workers=48)

    def forward(self, query_texts, masks):
        #print(query_texts, masks)
        logits = self.linear(self.bert(query_texts, masks)[1]).squeeze(-1)
        return logits

    def event_loss(self, logits, labels):
        #print(logits.shape, labels.shape)
        return self.criterion(logits, labels)

    def step(self, batch, batch_idx, mode='train'):
        [query_tokens, query_masks, query_labels] = batch
        logits = self(query_tokens, query_masks)
        event_evaluator = self.event_evaluators[mode]
        preds = torch.sigmoid(logits).round().bool()
        event_evaluator.accumulate(
            preds,
            query_labels
        )
        loss = self.event_loss(logits, query_labels.float())

        return loss

    def training_epoch_end(self, outputs):
        self.event_evaluators['train'].restart()
        return {}

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'train')
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'val')
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'test')
        return {'test_loss': loss}

    def validation_epoch_end(self, outputs):
        loss = torch.mean(self.gather_outputs(outputs, 'val_loss'))
        display = {'val_loss': loss}
        evaluator = self.event_evaluators['val']
        display['ev_pr'] = evaluator.metric('precision', average='binary', labels=[1])
        display['ev_re'] = evaluator.metric('recall', average='binary', labels=[1])
        display['ev_f1'] = evaluator.metric('f1', average='binary', labels=[1])
        display = {x: torch.tensor(t) for x, t in display.items()}
        evaluator.restart()
        return {'progress_bar': display, 'log': display}

    def test_epoch_end(self, outputs):
        loss = torch.mean(self.gather_outputs(outputs, 'test_loss'))
        display = {'test_loss': loss}
        evaluator = self.event_evaluators['test']
        display['ev_pr'] = evaluator.metric('precision', average='binary', labels=[1])
        display['ev_re'] = evaluator.metric('recall', average='binary', labels=[1])
        display['ev_f1'] = evaluator.metric('f1', average='binary', labels=[1])
        display = {x: torch.tensor(t) for x, t in display.items()}
        evaluator.restart()
        return {'progress_bar': display, 'log': display}

    def gather_outputs(self, outputs, keyword):
        target = [x[keyword] for x in outputs]
        if len(target[0].shape) == 0:
            return torch.stack(target, 0)
        target = torch.cat(target, 0)
        return target
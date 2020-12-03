from base.nn import *
from base.nn.utils import *
import torch
import torch.nn
import torch.nn.functional as F
import base.data.qa as qa
from base.data import *
import base.utils
import base.data.utils
from transformers import *
from transformers.optimization import AdamW, get_linear_schedule_with_warmup


class RCEntailModel(
    pl.LightningModule,
    DefaultParamsMixin,
    SequenceTruncateMixin):
    """ Sentence-level Bert based model for event detection.
    """

    def __init__(self, args):
        super(RCEntailModel, self).__init__()
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
            'batch_size': 16,
            'nevents': 33,
            'bert_lr': 2e-5,
            'cls_lr': 1e-4,
            'lr': 2e-5,
            'nepochs': 25,
            'event_query_template': "ST1",
            'event_label_processor': "IdentityEventProcessor",
            'use_event_description': 'none',
            'pretrain_model': 'bert-base-uncased',
            'reverse_query': True,
            'use_sep': True,
            'nsamps': 'all',
            'neutral_as': 'none',
            'warmup_steps': 0,
            'max_desc_sentences': 10,
        })
        print('model params:')
        print(self.params)

    def initialize_models(self):
        self.event_bert = transformers.AutoModelForSequenceClassification.from_pretrained(self.params.pretrain_model)

        #self.params.nhid = self.event_bert.embeddings.word_embeddings.weight.shape[-1]
        #self.event_criterion = F.binary_cross_entropy_with_logits
        self.event_criterion = F.cross_entropy


    def initialize_dataset(self):
        self.params.query_position = 'postfix' if self.params.reverse_query else 'prefix'
        self.params.tokenizer = self.params.pretrain_model

        def make_data(path, nsamps):
            return RCEventDataset(
                path,
                self.params.event_label_processor,
                self.params.event_query_template,
                self.params.use_event_description,
                self.params.tokenizer,
                self.params.query_position,
                self.params.use_sep,
                nsamps,
                self.params.max_desc_sentences
            )

        self.train_data = make_data(self.params.input_train, self.params.nsamps)
        self.train_data.grouped_shuffle()
        self.val_data = make_data(self.params.input_val, 'all')
        self.test_data = make_data(self.params.input_test, 'all')

    def initialize_evaluators(self):
        self.batch_states = {'train': None, 'test': None, 'val': None}
        self.event_evaluators = {
            'train': BinaryAccumEvaluator(),
            'test': BinaryAccumEvaluator(),
            'val': BinaryAccumEvaluator()
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

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Edited from https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.create_optimizer_and_scheduler
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.params.bert_lr
        )
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.params.warmup_steps, num_training_steps=num_training_steps
        )
        return optimizer, lr_scheduler

    def configure_optimizers(self):
        '''
        params_list = [
            {'params': self.event_bert.bert.parameters(), 'lr': self.params.bert_lr},
            {'params': self.event_bert.cls.parameters(), 'lr': self.params.cls_lr}
        ]
        '''
        params_list = [
            {'params': self.event_bert.parameters(), 'lr': self.params.bert_lr},
        ]
        optimizer = transformers.AdamW(params_list)
        return {'optimizer': optimizer,
                'lr_scheduler': 
                    lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=self.params.nepochs
                    )
                }
        '''
        optimizer, scheduler = self.create_optimizer_and_scheduler(
            len(self.train_dataloader()) * self.params.nepochs
        )
        return {
            'optimizer': optimizer,
            'scheduler': scheduler
        }
        '''

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True, num_workers=48)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.params.batch_size, shuffle=False, num_workers=48)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.params.batch_size, shuffle=False, num_workers=48)

    def prepare_data(self):
        pass

    def forward(self, inputs):
        res = base.utils.dispatch_args_to_func(
            self.event_bert.forward, 
            inputs,
            return_dict=True
        )
        return res['logits']

    def event_loss(self, logits, labels):
        #print(logits.shape, labels.shape)
        return self.event_criterion(logits, labels.long())

    def step(self, batch, batch_idx, mode='train'):
        inputs = self.process_batch(
            batch=batch,
            val_names=self.train_data.register_returns(),
            mask_names=['attention_mask']
        )
        #print(self.train_data.tokenizer.decode(inputs['input_ids'][0]))
        mnli_logits = self(inputs)

        # 0: contradict, 1: entailment, 2: neural
        assert self.params.neutral_as in ['positive', 'negative', 'none']
        event_logits = mnli_logits[:, :2]
        if self.params.neutral_as == 'positive':
            event_logits[:, 1] += mnli_logits[:, 2]
        elif self.params.neutral_as == 'negative':
            event_logits[:, 0] += mnli_logits[:, 2]

        event_evaluator = self.event_evaluators[mode]
        event_preds = torch.softmax(event_logits, -1).argmax(-1)
        event_evaluator.accumulate(
            event_preds,
            inputs['labels']
        )
        loss = self.event_loss(event_logits, inputs['labels'].float())

        return loss

    def training_epoch_end(self, outputs):
        self.event_evaluators['train'].restart()
        self.train_data.grouped_shuffle()
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
        display['ev_pr'] = evaluator.metric('precision')
        display['ev_re'] = evaluator.metric('recall')
        display['ev_f1'] = evaluator.metric('f1')
        display = {x: torch.tensor(t) for x, t in display.items()}
        evaluator.restart()
        return {'progress_bar': display, 'log': display}

    def test_epoch_end(self, outputs):
        loss = torch.mean(self.gather_outputs(outputs, 'test_loss'))
        display = {'test_loss': loss}
        evaluator = self.event_evaluators['test']
        display['ev_pr'] = evaluator.metric('precision')
        display['ev_re'] = evaluator.metric('recall')
        display['ev_f1'] = evaluator.metric('f1')
        display = {x: torch.tensor(t) for x, t in display.items()}
        evaluator.restart()
        return {'progress_bar': display, 'log': display}

    def gather_outputs(self, outputs, keyword):
        target = [x[keyword] for x in outputs]
        if len(target[0].shape) == 0:
            return torch.stack(target, 0)
        target = torch.cat(target, 0)
        return target
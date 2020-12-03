from base.nn import *
from base.nn.utils import *
from base.utils import *
import torch
import torch.nn
import torch.nn.functional as F
import base.data.qa as qa
import base.data.dataset as dataset
from base.data import *
import pickle
import platform

ENV = platform.system().lower()


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

        self.hparams = args
        self.initialize_params()
        self.initialize_dataset()
        self.initialize_models()
        self.initialize_evaluators()

    def initialize_params(self):
        self.params = Namespace()
        self.safely_set_attributes_by_dict(self.hparams)
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
            'desc_type': None,
            'pretrain_model': 'deepset/bert-base-cased-squad2',
            'head_num': 12,
            'reverse_query': False,
            'negative_sampling': 'all',
            'nsamp': 'all',
            'train_without_guide': False
        })
        print('model params:')
        print(self.params)

    def initialize_models(self):
        if self.use_entity:
            self.bert = transformers.AutoModel.from_pretrained(self.params.pretrain_model)
            self.params.nhid = self.bert.embeddings.word_embeddings.weight.shape[-1]
            self.linear = nn.Linear(self.params.nhid, 1)
        else:
            self.bert = transformers.BertForQuestionAnswering.from_pretrained(self.params.pretrain_model)
        self.criterion = F.binary_cross_entropy_with_logits

    def initialize_dataset(self):
        if ENV == 'windows':
            self.num_workers = 0
        else:
            self.num_workers = 48
        if self.params.val_pred_path is not None:
            self.val_preds = np.load(self.params.val_pred_path)
        else:
            self.val_preds = None
        if self.params.test_pred_path is not None:
            self.test_preds = np.load(self.params.test_pred_path)
        else:
            self.test_preds = None
        self.use_entity = getattr(qa, self.params.query_template)().use_entity
        if self.params.input_train_processed and os.path.exists(self.params.input_train_processed):
            with open(self.params.input_train_processed, 'rb') as f:
                self.train_data = pickle.load(f)
        else:
            self.train_data = RCArgumentDataset(
                input_file=self.params.input_train,
                argument_query_template=self.params.query_template,
                desc_type=self.params.desc_type,
                n_sample=self.params.nsamp,
                use_entity_type=True,
                is_eval_data=False,
                tokenizer=self.params.pretrain_model,
                train_without_guide=self.params.train_without_guide
            )
        if self.params.input_val_processed and os.path.exists(self.params.input_val_processed):
            with open(self.params.input_val_processed, 'rb') as f:
                self.val_data = pickle.load(f)
        else:
            self.val_data = RCArgumentDataset(
                input_file=self.params.input_val,
                argument_query_template=self.params.query_template,
                desc_type=self.params.desc_type,
                use_entity_type=True,
                is_eval_data=True,
                tokenizer=self.params.pretrain_model,
                event_preds=self.val_preds
            )
        if self.params.input_test_processed and os.path.exists(self.params.input_test_processed):
            with open(self.params.input_test_processed, 'rb') as f:
                self.test_data = pickle.load(f)
        else:
            self.test_data = RCArgumentDataset(
                input_file=self.params.input_test,
                argument_query_template=self.params.query_template,
                desc_type=self.params.desc_type,
                use_entity_type=True,
                is_eval_data=True,
                tokenizer=self.params.pretrain_model,
                event_preds=self.test_preds
            )
        self.event_map, self.argument_map = self.train_data.preprocessors[RCArgumentPreprocessor].event_map, self.train_data.preprocessors[RCArgumentPreprocessor].argument_map

    def initialize_evaluators(self):
        self.batch_states = {'train': None, 'test': None, 'val': None}
        self.ground_truth_argument_evaluators = {
            'train': SimpleAccumEvaluator(),
            'test': SimpleAccumEvaluator(),
            'val': SimpleAccumEvaluator()
        }
        self.event_pred_argument_evaluators = {
            'train': SimpleAccumEvaluator(),
            'test': SimpleAccumEvaluator(),
            'val': SimpleAccumEvaluator()
        }

    def configure_optimizers(self):
        params_list = [{'params': self.bert.parameters(), 'lr': self.params.bert_lr}]
        if self.use_entity:
            params_list.append({'params': self.linear.parameters(), 'lr': self.params.cls_lr})
        optimizer = transformers.AdamW(params_list)
        return {'optimizer': optimizer,
                'lr_scheduler':
                    lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=self.params.nepochs
                    )
                }

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.params.batch_size, shuffle=False,
                          num_workers=self.num_workers)

    def forward(self, inputs):
        res = dispatch_args_to_func(
            self.bert.forward,
            inputs
        )
        if self.use_entity:
            res = self.linear(res[1]).squeeze(-1)
        return res

    def event_loss(self, logits, labels):
        return self.criterion(logits, labels)

    def argument_loss(self, start_logits, end_logits, start_labels, end_labels):
        return self.criterion(start_logits, start_labels) + self.criterion(end_logits, end_labels)

    def predict_argument(self, start_logits, end_logits, entity_spans, possible_mask):
        start_preds = start_logits.sigmoid().round()
        end_preds = end_logits.sigmoid().round()
        n_batch = start_logits.shape[0]
        entity_num = possible_mask.shape[1]
        assert entity_spans.shape == torch.Size([n_batch, entity_num, 2]), (entity_spans.shape, n_batch, entity_num)
        argument_preds = torch.zeros([n_batch, entity_num], dtype=torch.bool, device=self.device)
        for row in range(n_batch):
            for col in range(entity_num):
                if possible_mask[row, col] == 1 and start_preds[row, entity_spans[row, col, 0]] == 1 and end_preds[row, entity_spans[row, col, 1]] == 1:
                    argument_preds[row, col] = 1
        return argument_preds

    def step(self, batch, batch_idx, mode='train'):
        inputs = self.process_batch(
            batch=batch,
            val_names=self.train_data.register_returns(),
            excepts=['possible_entity_masks', 'query_labels', 'id_tuples'],
            mask_names=['attention_mask']
        )
        inputs['input_ids'] = inputs['input_ids'].long()
        inputs['token_type_ids'] = inputs['token_type_ids'].long()
        if not self.use_entity:
            start_logits, end_logits = self(inputs)
            loss = self.argument_loss(
                start_logits, end_logits,
                inputs['argument_start_labels'], inputs['argument_end_labels']
            )
            preds = self.predict_argument(start_logits, end_logits, inputs['entity_spans'], inputs['possible_entity_masks'])
        else:
            logits = self(inputs)
            preds = torch.sigmoid(logits).round().bool()
            loss = self.event_loss(logits, inputs['query_labels'].float())
        event_evaluator = self.ground_truth_argument_evaluators[mode]
        event_evaluator.accumulate(
            preds.bool().detach().cpu(),
            inputs['query_labels'].bool().detach().cpu(),
            masks=inputs['label_masks'].detach().cpu()
        )
        event_evaluator = self.event_pred_argument_evaluators[mode]
        event_evaluator.accumulate(
            preds.bool().detach().cpu(),
            inputs['query_labels'].bool().detach().cpu(),
            masks=inputs['pred_masks'].detach().cpu()
        )
        return loss

    def training_epoch_end(self, outputs):
        self.ground_truth_argument_evaluators['train'].restart()
        self.event_pred_argument_evaluators['train'].restart()
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
        evaluator = self.ground_truth_argument_evaluators['val']
        display['gr_truth_arg_pr'] = evaluator.metric('precision')
        display['gr_truth_arg_re'] = evaluator.metric('recall')
        display['gr_truth_arg_f1'] = evaluator.metric('f1')
        evaluator.restart()
        evaluator = self.event_pred_argument_evaluators['val']
        display['ev_pred_arg_pr'] = evaluator.metric('precision')
        display['ev_pred_arg_re'] = evaluator.metric('recall')
        display['ev_pred_arg_f1'] = evaluator.metric('f1')
        display = {x: torch.tensor(t) for x, t in display.items()}
        evaluator.restart()
        return {'progress_bar': display, 'log': display}

    def test_epoch_end(self, outputs):
        loss = torch.mean(self.gather_outputs(outputs, 'test_loss'))
        display = {'test_loss': loss}
        evaluator = self.ground_truth_argument_evaluators['test']
        display['gr_truth_arg_pr'] = evaluator.metric('precision')
        display['gr_truth_arg_re'] = evaluator.metric('recall')
        display['gr_truth_arg_f1'] = evaluator.metric('f1')
        evaluator.restart()
        evaluator = self.event_pred_argument_evaluators['test']
        display['ev_pred_arg_pr'] = evaluator.metric('precision')
        display['ev_pred_arg_re'] = evaluator.metric('recall')
        display['ev_pred_arg_f1'] = evaluator.metric('f1')
        display = {x: torch.tensor(t) for x, t in display.items()}
        evaluator.restart()
        return {'progress_bar': display, 'log': display}

    def gather_outputs(self, outputs, keyword):
        target = [x[keyword] for x in outputs]
        if len(target[0].shape) == 0:
            return torch.stack(target, 0)
        target = torch.cat(target, 0)
        return target

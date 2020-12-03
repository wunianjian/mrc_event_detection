import torch
import torch.nn as nn
import torch.nn.functional as F
from base.nn.utils import DefaultParamsMixin
from abc import abstractmethod


class ClassifierBase:
    """ Abstract class that does absolutely nothing
    """

    @abstractmethod
    def __init__(self):
        pass


class LinearClassifier(
    nn.Module,
    DefaultParamsMixin,
    ClassifierBase):
    def __init__(self, params):
        super().__init__()

        self.params = params
        self.nhid = params.nhid
        self.ntypes = params.ntypes
        self.linear = nn.Linear(self.nhid, self.ntypes)

    def forward(self, batch):
        tokens = batch.tokens_embed  # expected shape of [nbatch, ntokens, nhid]
        logits = self.linear(tokens)

        return logits, None


class MLPProjector(
    nn.Module,
    DefaultParamsMixin,
    ClassifierBase):
    def __init__(self, params, output_dim=128):
        super(MLPProjector, self).__init__()

        self.params = params
        self.nhid = params.nhid
        self.mlp = nn.Sequential(
                nn.Linear(self.nhid, self.nhid),
                nn.ReLU(inplace=True),
                nn.Linear(self.nhid, output_dim)
            )

    def forward(self, batch):
        tokens = batch.tokens_embed  # expected shape of [nbatch, ntokens, nhid]
        logits = self.mlp(tokens)

        return logits, None


class EventTokenClassifier(
    nn.Module,
    DefaultParamsMixin,
    ClassifierBase):
    def __init__(self, params):
        super().__init__()

        self.params = params
        self.nhid = params.nhid
        self.ntypes = params.ntypes
        self.multihead_attention = nn.MultiheadAttention(self.nhid, 2)
        self.linear = nn.Linear(2 * self.nhid, self.ntypes)
        self.sentence_linear = nn.Linear(self.nhid, self.nhid)
        self.event_key = nn.Linear(self.nhid, self.nhid)
        self.event_value = nn.Linear(self.nhid, self.nhid)

    def forward(self, batch):
        tokens = batch.tokens_embed  # expected shape of [nbatch, ntokens, nhid]
        events = batch.events_embed[:, 0, :]  # expected shape of [nevents, nhid]
        assert tokens.shape[-1] == events.shape[-1]

        nbatch = tokens.shape[0]
        ntokens = tokens.shape[1]
        sentence_vectors = tokens[:, 0, :]
        sentence_vectors = sentence_vectors.unsqueeze(0)  # [1, nbatch, nhid]
        events = events.unsqueeze(1).repeat(1, nbatch, 1)  # [nevents, nbatch, nhid]
        event_vectors, attns = self.multihead_attention(
            self.sentence_linear(sentence_vectors),
            self.event_key(events),
            self.event_value(events))
        event_vectors = event_vectors.transpose(0, 1)
        attns = attns.squeeze()  # [nbatch, nevents]

        event_vectors = event_vectors.repeat(1, ntokens, 1)  # [nbatch, ntokens, nhid]
        x = torch.cat([tokens, event_vectors], dim=2)
        logits = self.linear(x)

        return logits, attns


class TypeAwareEventClassifier(
    nn.Module,
    DefaultParamsMixin,
    ClassifierBase):
    def __init__(self, nhid):
        super(TypeAwareEventClassifier, self).__init__()

        self.nhid = nhid
        self.head_num = 2
        self.multihead_attention = nn.MultiheadAttention(self.nhid, self.head_num)

    def forward(self, batch, _t1, _t2):
        nbatch = batch.tokens_embed.shape[0]
        nevent = _t1.shape[0]
        t1 = _t1.unsqueeze(1).repeat(1, nbatch, 1)          # [nevent, nbatch, nhid]
        tokens_embed = batch.tokens_embed.transpose(0, 1)   # [seq_len, nbatch, nhid]
        attn_mask = ~batch.masks
        attn_mask[:, 0] = True
        sentence_att, att = self.multihead_attention(t1, tokens_embed, tokens_embed,
                                                    key_padding_mask=attn_mask)
        sentence_att = sentence_att.transpose(0, 1)
        v_att = (sentence_att * _t1).sum(-1)
        sentence_embed = batch.tokens_embed[:, 0, :]
        v_global = (sentence_embed.unsqueeze(1).repeat(1, nevent, 1) * _t2).sum(-1)
        return v_att, v_global

class TypeAwareHeterEventClassifier(
    nn.Module,
    DefaultParamsMixin,
    ClassifierBase):
    def __init__(self, nhid):
        super(TypeAwareHeterEventClassifier, self).__init__()

        self.nhid = nhid
        self.head_num = 2
        self.multihead_attention = nn.MultiheadAttention(self.nhid, self.head_num)

    def forward(self, batch, _t1, _t2):
        nbatch = batch.tokens_embed.shape[0]
        nevent = _t1.shape[1]
        t1 = _t1.transpose(0, 1)       
        tokens_embed = batch.tokens_embed.transpose(0, 1)   
        attn_mask = ~batch.masks
        attn_mask[:, 0] = True
        sentence_att, att = self.multihead_attention(t1, tokens_embed, tokens_embed,
                                                    key_padding_mask=attn_mask)
        sentence_att = sentence_att.transpose(0, 1)
        v_att = (sentence_att * _t1).sum(-1)
        sentence_embed = batch.tokens_embed[:, 0, :]
        v_global = (sentence_embed.unsqueeze(1).repeat(1, nevent, 1) * _t2).sum(-1)
        return v_att, v_global

class TypeAwareTokensClassifier(
    nn.Module,
    DefaultParamsMixin,
    ClassifierBase
):
    def __init__(self, nhid):
        super(TypeAwareTokensClassifier, self).__init__()

        self.nhid = nhid
        self.head_num = 16
        self.t1_attention = nn.MultiheadAttention(self.nhid, self.head_num)
        self.t2_attention = nn.MultiheadAttention(self.nhid, self.head_num)

        self.t1_linear = nn.Linear(nhid, 1)
        self.t2_linear = nn.Linear(nhid, 1)

    def forward(self, batch, _t1, _t2):
        nbatch = batch.tokens_embed.shape[0]
        nevent = _t1.shape[0]

        tokens_embed = batch.tokens_embed
        nbatch, seqlen = tokens_embed.shape[:2]
        sentence_embed = batch.tokens_embed[:, 0, :].unsqueeze(1).unsqueeze(1).repeat(1, seqlen, 1, 1)

        tokens_embed = tokens_embed.unsqueeze(2)
        _t1 = _t1.unsqueeze(0).unsqueeze(0).repeat(nbatch, 1, 1, 1)
        _t2 = _t2.unsqueeze(0).unsqueeze(0).repeat(nbatch, 1, 1, 1)

        '''
        v_att = self.t1_linear(
            F.relu(
                tokens_embed + _t1 
            ) 
        ).squeeze(-1) + (tokens_embed * _t1).sum(-1)

        v_global = self.t2_linear(
            F.relu(
                sentence_embed + _t2
            ) 
        ).squeeze(-1) + (sentence_embed * _t2).sum(-1)
        '''
        v_att = (tokens_embed * _t1).sum(-1)
        v_global = (sentence_embed * _t2).sum(-1)

        return v_att, v_global

class EntityAttention(nn.Module):
    def __init__(self, nhid):
        super(EntityAttention, self).__init__()
        self.nhid = nhid
        self.head_num = 2
        self.multihead_attention = nn.MultiheadAttention(self.nhid, self.head_num)

    def forward(self, tokens_embed, entities, events_embed, entity_num, entity_masks, select_event):
        nbatch = tokens_embed.shape[0]
        seq_len = tokens_embed.shape[1]
        nevents = events_embed.shape[0]
        entity_index = torch.tensor(range(entity_num * nbatch)).to(entity_masks.device).masked_select(entity_masks.flatten())
        event_entity_select = (select_event.unsqueeze(1).repeat(1, entity_num, 1) * entity_masks.unsqueeze(2).repeat(1, 1, nevents)).view(-1, nevents)[entity_index, :].flatten()
        event_entity_index = torch.tensor(range(entity_index.shape[0] * nevents)).to(entity_masks.device).masked_select(event_entity_select)
        attn_mask = ~entities.view(-1, seq_len)[entity_index, :]
        tokens_embed = tokens_embed.transpose(0, 1).unsqueeze(2).repeat(1, 1, entity_num, 1).view(seq_len, -1, self.nhid)[:, entity_index, :]
        events_embed = events_embed.unsqueeze(1).repeat(1, entity_num * nbatch, 1)[:, entity_index, :]
        entity_embed, attn = self.multihead_attention(events_embed, tokens_embed, tokens_embed, key_padding_mask=attn_mask)
        entity_embed = entity_embed.transpose(0, 1).reshape(-1, self.nhid)[event_entity_index]
        return entity_embed


class ReadingComprehensionClassifier(nn.Module):
    def __init__(self, nhid, model_ref):
        super(ReadingComprehensionClassifier, self).__init__()
        self.event_classifier = nn.Linear(nhid, 1)
        self.sep_token_idx = model_ref.sep_token_idx
        #self.model_ref = model_ref
        self.model_ref = [model_ref]

    def forward(self, batch):
        tokens, masks = batch.sentences, batch.masks

        cls_col = tokens[:, :1]
        tokens = tokens[:, 1:]
        masks = masks[:, 1:]
        #event_tokens = event_tokens[:, :1]
        #event_token_masks = event_token_masks[:, :1]
        event_tokens = self.model_ref[0].event_tokens.to(tokens.device)
        event_token_masks = self.model_ref[0].event_token_masks.to(tokens.device)
        sep_col = torch.ones_like(cls_col) * self.sep_token_idx
        one_col_true = torch.ones_like(cls_col).bool()

        tokens = torch.cat([sep_col, tokens, sep_col], dim=1)
        masks = torch.cat([one_col_true, masks, one_col_true], dim=1)

        # tokens: [nbatch, seqlen]
        # event_tokens: [nevents, seqlen]
        nevents, netoks = event_tokens.shape
        nbatch, ntoks = tokens.shape
        tokens = tokens.repeat_interleave(nevents, 0)   # [nbatch * nevents, seqlen]
        masks = masks.repeat_interleave(nevents, 0)
        event_tokens = event_tokens.repeat(nbatch, 1)
        event_token_masks = event_token_masks.repeat(nbatch, 1)
        
        combined_tokens = torch.cat([event_tokens, tokens], 1)  # [nevents * nbatch, netoks + ntoks]
        combined_masks = torch.cat([event_token_masks, masks], 1)
        #combined_tokens = combined_tokens.reshape([nbatch, nevents, netoks+ntoks]
        #self.model_ref[0].event_bert.to(combined_masks.device)
        combined_hid, _ = self.model_ref[0].event_bert(combined_tokens, combined_masks)  # [nevents * nbatch, netoks + ntoks, nhid]
        combined_hid = combined_hid.reshape([nbatch, nevents, netoks+ntoks, combined_hid.shape[-1]])

        # option 1: direct reading comprehension approach. 
        event_logits = self.event_classifier(combined_hid[:, :, 0, :])
        return event_logits.squeeze(-1)
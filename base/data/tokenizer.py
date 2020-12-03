
import json
import transformers
from collections import Iterable
import copy
from nltk.tokenize import word_tokenize
import numpy as np
import unicodedata
from . import utils

def strip_accents(text):

    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3 
        pass

    text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")

    return str(text)

def encode_pretty(words, tokenizer_fn, decoder_fn, verbose=False):
    """ Encode a sequence with the given tokenizer and returns a 
        dictionary mapping original indices and indices in the tokenized sequence.
    """
    sentence = ' '.join(words)
    encoded = tokenizer_fn(sentence)
    reverse_hash = {}
    i = 0
    cur_word = words[i]

    p = 0
    finished = False
    while p < len(encoded) and not finished:
        for q in range(p+1, len(encoded)+1):
            t = decoder_fn(encoded[p:q]).replace(' ', '')
            cur_word = strip_accents(cur_word.lower().replace(' ', ''))
            t = strip_accents(t.lower())
            if verbose:
                print('comparing: {} ; {}'.format(cur_word, t))
            if not cur_word.startswith(t):
                if verbose: print('not startswith; break out')
                break
            if t.lower() == cur_word.lower():
                if verbose: print('succeed for ' + str(i))
                reverse_hash[i] = (p, q)
                i += 1
                p = q-1
                try:
                    cur_word = words[i]
                except IndexError:
                    finished = True
                break
            else:
                if verbose: print('failed')
        p += 1
    reverse_hash[len(words)] = [len(encoded), len(encoded)]
    return encoded, reverse_hash

class IndexedToken(int):
    def __new__(cls, word, index):
        x = super(IndexedToken, cls).__new__(cls, index)
        x.word = word
        x.index = index
        return x

    def __int__(self):
        return self.index

    def __str__(self):
        return self.word
    
    def __repr__(self):
        return str(self.index)

    def __hash__(self):
        return self.index
    
class DictTokenizer:

    default_special_tokens = {
        'pad_token': '<PAD>',
        'unk_token': '<UNK>'
    }

    def add_special_token(self, special_token):
        if special_token in self.token2id:
            return
        self.token2id[special_token] = len(self.token2id)

    def remove_token(self, token):
        if token not in self.token2id:
            return
        idx = self.token2id[token]
        del self.token2id[token]
        del self.id2token[idx]

    def configure_default_special_tokens(self):
        for name, token in DictTokenizer.default_special_tokens.items():
            setattr(self, name, token)
            self.add_special_token(token)

    def __init__(self, token2id):
        assert type(token2id) == dict, \
            """Expected dictionary type as token-to-index map;
            received type {} instead.""".format(type(token2id))
        self.token2id = token2id
        self.configure_default_special_tokens()
        self.id2token = {t: i for i, t in self.token2id.items()}

    def token_to_index(self, token):
        if token in self.token2id:
            return IndexedToken(token, self.token2id[token])
        else:
            return IndexedToken(token, self.token2id[self.unk_token])

    def index_to_token(self, index):
        return self.id2token[index]

    def __encode_one_sequence(self, sequence):
        return [self.token_to_index(x) for x in word_tokenize(sequence)]

    def encode(self, sequence):
        return utils.apply_to_sequences(self.__encode_one_sequence, sequence)

    def __decode_one_sequence(self, sequence):
        return ' '.join([self.index_to_token(x) if type(x) is int else str(x) for x in sequence])

    def decode(self, sequence):
        return utils.apply_to_sequences(self.__decode_one_sequence, sequence)

class Tokenizer:
    def __init__(self, tokenizer):
        """ Initialize a tokenizer according to input.
        tokenizer (str): specified tokenizer type. Should either be a path 
            or a Transformer tokenizer identifier such as 'bert-base-uncased'.
            This function will attempt to load a Transformer tokenizer first. 
            When input is a path, it should point to a json file which includes
            a dictionary that is a token-to-index map. 
        """
        loaded = False
        # attempt to load a Transformer tokenizer.
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)
            loaded = True
        except (OSError,ValueError):
            pass

        try: 
            with open(tokenizer, 'r') as f:
                self.tokenizer = DictTokenizer(json.load(f))
            loaded = True
        except FileNotFoundError:
            pass
        
        if not loaded:
            raise RuntimeError("""Tokenizer expected to be either a valid Transformer 
                tokenizer or a path to a json dictionary file.""")

    def encode(self, sequence, **kwargs):
        return utils.apply_to_sequences(self.tokenizer.encode, sequence, **kwargs)
    
    def decode(self, sequence, **kwargs):
        return utils.apply_to_sequences(self.tokenizer.decode, sequence, **kwargs)
import json
import math
from collections import OrderedDict
import torch
from torch import nn, Tensor
from modules.utils import cosine_sim
from typing import Union, Tuple, List, Iterable, Dict
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import AdamW
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
import numpy as np
import gzip
import csv
import pandas as pd
from tqdm.auto import tqdm


def gelu(x):
    """
    The Gelu activation function.
    :param x:
    :return:
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Config(object):
    def __init__(self,
                vocab_size,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                dropout_prob=0.9,
                max_position_embeddings=512,
                type_vocab_size=2,
                initializer_range=0.02):

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = dropout_prob
        self.attention_probs_dropout_prob = dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, dict_object):
        config = Config(vocab_size=None)
        for (key, value) in dict_object.items():
            config.__dict__[key] = value
        return config


class LayerNorm(nn.Module):

    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This function needs some editing
        :param x: torch.Tensor
            The input parameter
        :return normalized: torch.Tensor
            The normalized tensor in the LayerNorm
        """
        """
        Bug #1,2,3
        Code Edit
        u = x.mean(0, keepdim=True)
        s = (x + u).pow(2).mean(0, keepdim=True)
        x = (x + u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta
        """
        mean = x.mean(dim=-1, keepdim=True)  # Edited line #1
        var = ((x - mean).pow(2)).mean(dim=-1, keepdim=True)  # Edited line #2
        std = (var + self.variance_epsilon).sqrt()  # Edited line #3
        y = (x - mean) / std
        return self.gamma * y + self.beta




class MLP(nn.Module):

    def __init__(self, hidden_size, intermediate_size):

        super(MLP, self).__init__()
        self.dense_expansion = nn.Linear(hidden_size, intermediate_size)
        self.dense_contraction = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        x = self.dense_expansion(x)
        x = self.dense_contraction(gelu(x))
        return x


class Layer(nn.Module):
    def __init__(self, config):
        super(Layer, self).__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        """
        Bug #4
        Replace
        >>
        self.attn_out = nn.Linear(config.hidden_size, config.hidden_size) -> 
        self.attn_out = nn.Linear(self.all_head_size, config.hidden_size)
        """
        self.attn_out = nn.Linear(self.all_head_size, config.hidden_size)  # Edited line #4
        self.ln1 = LayerNorm(config.hidden_size)

        # BertSelfOut ends here
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.ln2 = LayerNorm(config.hidden_size)

    def split_heads(self, tensor, num_heads, attention_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attention_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)

    def merge_heads(self, tensor, num_heads, attention_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attention_head_size,)
        return tensor.view(new_shape)

    def attn(self, q, k, v, attention_mask):
        mask = attention_mask == 1
        mask = mask.unsqueeze(1).unsqueeze(2)

        """
        Bug #5
        >>
        s = torch.matmul(q, k) -> s = torch.matmul(q, k.transpose(-1, -2))
        """
        s = torch.matmul(q, k.transpose(-1, -2))  # Edited line #5

        s = s / math.sqrt(self.attention_head_size)

        """
        Bug #6
        >>
        s = torch.where(mask, s, torch.tensor(float('inf'))) -> 
        s = torch.where(mask, s, torch.tensor(float('-inf')))
        """
        s = torch.where(mask, s, torch.tensor(float('-inf')))  # Edited line #6

        """
        Bug #7
        >>
        p = s -> 
        p = nn.functional.softmax(s, dim=-1)
        """
        p = nn.functional.softmax(s, dim=-1)  # Edited line #7
        p = self.dropout(p)

        a = torch.matmul(p, v)
        return a

    def forward(self, x, attention_mask):
        q, k, v = self.query(x), self.key(x), self.value(x)

        q = self.split_heads(q, self.num_attention_heads, self.attention_head_size)
        k = self.split_heads(k, self.num_attention_heads, self.attention_head_size)
        v = self.split_heads(v, self.num_attention_heads, self.attention_head_size)

        a = self.attn(q, k, v, attention_mask)
        a = self.merge_heads(a, self.num_attention_heads, self.attention_head_size)
        a = self.attn_out(a)
        a = self.dropout(a)
        a = self.ln1(a)

        m = self.mlp(a)
        m = self.dropout(m)
        m = self.ln2(m)

        return m


class Bert(nn.Module):
    def __init__(self, config_dict):
        super(Bert, self).__init__()
        """
        Replaced 
        # padding_idx=0 -> padding_idx=self.config.pad_token_id
        """
        self.config = Config.from_dict(config_dict)

        self.embeddings = nn.ModuleDict({
          'token': nn.Embedding(self.config.vocab_size, self.config.hidden_size, padding_idx=0),
          'position': nn.Embedding(self.config.max_position_embeddings, self.config.hidden_size),
          'token_type': nn.Embedding(self.config.type_vocab_size, self.config.hidden_size),
        })

        self.ln = LayerNorm(self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.layers = nn.ModuleList([
            Layer(self.config) for _ in range(self.config.num_hidden_layers)
        ])

        self.pooler = nn.Sequential(OrderedDict([
            ('dense', nn.Linear(self.config.hidden_size, self.config.hidden_size)),
            ('activation', nn.Tanh()),
        ]))

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, ):
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        """
        Replaced 
        >>>
        x = torch.cat((self.embeddings.token(input_ids),
                       self.embeddings.position(position_ids),
                       self.embeddings.token_type(token_type_ids)),
                      dim=-1)
        >>>
        x = self.embeddings.token(input_ids) + self.embeddings.position(position_ids) + self.embeddings.token_type(token_type_ids)
        """
        x = self.embeddings.token(input_ids) + self.embeddings.position(position_ids) + self.embeddings.token_type(token_type_ids)

        # x = torch.cat((self.embeddings.token(input_ids),
        #                self.embeddings.position(position_ids),
        #                self.embeddings.token_type(token_type_ids)),
        #               dim=-1)

        x = self.dropout(self.ln(x))

        for layer in self.layers:
            x = layer(x, attention_mask)

        o = self.pooler(x[:, 0])
        return (x, o)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        return self


class Softmax(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(Softmax, self).__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        output = self.linear(x)
        return output

class BertClassifier(nn.Module):
    def __init__(self, pretrained_model: nn.Module, pool: str, max_length: int, num_labels: int):
        super(BertClassifier, self).__init__()
        pretrained_model.requires_grad = True
        self.pretrained_model = pretrained_model

        self.softmax_classifier = Softmax(max_length * 3, num_labels)
        self.sf = nn.Softmax(dim=1)
        assert pool == 'mean' or pool == 'max', "Pooling method not valid!"
        self.pool = pool
        self.model_type = 'classification'

    def forward(self, sentence1, sentence2):

        sentence1_embed = self.pretrained_model(input_ids=sentence1[0], attention_mask=sentence1[1])[0]
        sentence2_embed = self.pretrained_model(input_ids=sentence2[0], attention_mask=sentence2[1])[0]
        if self.pool == 'max':
            sentence1_embed = torch.max(sentence1_embed, dim=2)
            sentence2_embed = torch.max(sentence2_embed, dim=2)
        elif self.pool == 'mean':
            sentence1_embed = sentence1_embed.mean(2)
            sentence2_embed = sentence2_embed.mean(2)

        embedding = torch.cat([sentence1_embed, sentence2_embed, torch.abs(sentence1_embed - sentence2_embed)], dim=1)
        output = self.softmax_classifier(embedding)
        output = self.sf(output)
        return output


class BertContrastive(nn.Module):
    def __init__(self, pretrained_model: nn.Module, pool: str, max_length: int, num_labels: int):
        super(BertContrastive, self).__init__()
        self.pretrained_model = pretrained_model
        assert pool == 'mean' or pool == 'max', "Pooling method not valid!"
        self.pool = pool
        self.model_type = 'regression'


    def forward(self, sentence1, sentence2):
        sentence1_embed = self.pretrained_model(input_ids=sentence1[0], attention_mask=sentence1[1])[0]
        sentence2_embed = self.pretrained_model(input_ids=sentence2[0], attention_mask=sentence2[1])[0]
        if self.pool == 'max':
            sentence1_embed = torch.max(sentence1_embed, dim=2)
            sentence2_embed = torch.max(sentence2_embed, dim=2)
        elif self.pool == 'mean':
            sentence1_embed = sentence1_embed.mean(2)
            sentence2_embed = sentence2_embed.mean(2)
        cosine_similarity = cosine_sim(sentence1_embed, sentence2_embed)
        return torch.diagonal(cosine_similarity)


class EmbedingClassifier(nn.Module):
    def __init__(self):
        super(EmbedingClassifier).__init__()
        pass

    def forward(self):
        pass
import json
import math
from collections import OrderedDict
import torch
from torch import nn, Tensor
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
from transformers import AutoTokenizer
from modules.models import Bert
from modules.utils import load_sts_dataset, tokenize_sentence_pair_dataset, get_dataloader, eval_loop
from transformers import AutoModel

torch.manual_seed(0)
np.random.seed(0)


def main():
    MODEL_NAME = 'prajjwal1/bert-tiny'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    ## IF YOU CANNOT SOLVE PREVIOUS BUGS, USE THE LINE BELOW:
    ## bert = AutoModel.from_pretrained(MODEL_NAME)

    bert_config = {"hidden_size": 128, "num_attention_heads": 2, "num_hidden_layers": 2, "intermediate_size": 512,
                   "vocab_size": 30522}

    """
    new config
    """
    bert_config = {"hidden_size": 128, "num_attention_heads": 2, "num_hidden_layers": 2, "intermediate_size": 512,
                   "vocab_size": 30522, "hidden_dropout_prob": 0.1, "layer_norm_eps": 1e-12}

    bert = Bert(bert_config).load_model('bert_tiny.bin')
    # bert = AutoModel.from_pretrained(MODEL_NAME)
    # EXAMPLE USE
    sentence = 'An example use of pretrained BERT with transformers library to encode a sentence'
    tokenized_sample = tokenizer(sentence, return_tensors='pt', padding='max_length', max_length=512)
    output = bert(input_ids=tokenized_sample['input_ids'],
                  attention_mask=tokenized_sample['attention_mask'], )

    # We use "pooler_output" for simplicity. This corresponds the last layer
    # hidden-state of the first token of the sequence (CLS token) after
    # further processing through the layers used for the auxiliary pretraining task.
    embedding = output[1]
    print(f'\nResulting embedding shape: {embedding.shape}')


def main2():
    data = pd.read_csv('stsbenchmark.tsv.gz', nrows=5, compression='gzip', delimiter='\t')
    data.head()


    # INFO: model and tokenizer
    model_name = 'prajjwal1/bert-tiny'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # INFO: load bert
    bert_config = {"hidden_size": 128, "num_attention_heads": 2, "num_hidden_layers": 2, "intermediate_size": 512,
                   "vocab_size": 30522}
    bert = Bert(bert_config).load_model('bert_tiny.bin')

    # INFO: load dataset
    sts_dataset = load_sts_dataset('stsbenchmark.tsv.gz')

    # INFO: tokenize dataset
    tokenized_test = tokenize_sentence_pair_dataset(sts_dataset['test'], tokenizer)

    # INFO: generate dataloader
    test_dataloader = get_dataloader(tokenized_test, batch_size=5)

    # INFO: run evaluation loop
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    results_from_pretrained = eval_loop(bert, test_dataloader, device)

    print(
        f'\nPearson correlation: {results_from_pretrained[0]:.2f}\nSpearman correlation: {results_from_pretrained[1]:.2f}')


if __name__ == "__main__":
    # main()
    main2()

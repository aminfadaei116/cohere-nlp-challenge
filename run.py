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
from modules.models import Bert, BertClassifier, BertContrastive, SupremeBert, EmbedingClassifier
from modules.utils import load_sts_dataset, tokenize_sentence_pair_dataset, get_dataloader, eval_loop
from transformers import AutoModel
from modules.utils import load_nli_dataset, train_loop

torch.manual_seed(0)
np.random.seed(0)


def main():
    MODEL_NAME = 'prajjwal1/bert-tiny'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    bert_config = {"hidden_size": 128, "num_attention_heads": 2, "num_hidden_layers": 2, "intermediate_size": 512,
                   "vocab_size": 30522}

    """
    new config
    """
    bert_config = {"hidden_size": 128, "num_attention_heads": 2, "num_hidden_layers": 2, "intermediate_size": 512,
                   "vocab_size": 30522, "hidden_dropout_prob": 0.1, "layer_norm_eps": 1e-12}

    # bert = Bert(bert_config).load_model('bert_tiny.bin')
    bert = AutoModel.from_pretrained(MODEL_NAME)
    bert = bert.eval()

    # EXAMPLE USE
    sentence = 'An example use of pretrained BERT with transformers library to encode a sentence'
    tokenized_sample = tokenizer(sentence, return_tensors='pt', padding='max_length', max_length=512)
    output = bert(input_ids=tokenized_sample['input_ids'],
                  attention_mask=tokenized_sample['attention_mask'], )

    # We use "pooler_output" for simplicity. This corresponds the last layer
    # hidden-state of the first token of the sequence (CLS token) after
    # further processing through the layers used for the auxiliary pretraining task.
    embedding = output[1]
    print(embedding)
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
    # bert = Bert(bert_config).load_model('bert_tiny.bin')

    # check here
    MODEL_NAME = 'prajjwal1/bert-tiny'
    # bert = AutoModel.from_pretrained(MODEL_NAME)
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
        f"\nPearson correlation: {results_from_pretrained[0]:.2f}\nSpearman correlation: {results_from_pretrained[1]:.2f}")


def main3():
    # INFO: model and training configs
    model_name = 'prajjwal1/bert-tiny'
    num_epochs = 3
    batch_size = 8
    num_labels = 3
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_config = {"hidden_size": 128, "num_attention_heads": 2, "num_hidden_layers": 2, "intermediate_size": 512,
                   "vocab_size": 30522}
    bert_path = 'bert_tiny.bin'

    # INFO: load nli dataset
    nli_dataset = load_nli_dataset('AllNLI.tsv.gz')

    # INFO: tokenize dataset
    # WARNING: Use only first 50000 samples and maximum sequence length of 128
    tokenized_train = tokenize_sentence_pair_dataset(nli_dataset['train'][:50000], tokenizer, max_length=128)

    # INFO: generate train_dataloader
    train_dataloader = get_dataloader(tokenized_train, batch_size=batch_size, shuffle=True)

    ###    Replace None with required input based on yor implementation

    bert = Bert(bert_config).load_model('bert_tiny.bin')
    # bert = AutoModel.from_pretrained(model_name)
    bert.eval()
    # bert.train()
    bert_classifier = BertClassifier(bert, max_length=128, num_labels=num_labels)

    # INFO: create optimizer and run training loop
    optimizer = AdamW([param for param in bert_classifier.parameters() if param.requires_grad], lr=5e-5)
    train_loop(bert_classifier, optimizer, train_dataloader, num_epochs, device)

    sts_dataset = load_sts_dataset('stsbenchmark.tsv.gz')

    # INFO: tokenize dataset
    tokenized_test = tokenize_sentence_pair_dataset(sts_dataset['test'], tokenizer)

    # INFO: generate dataloader
    test_dataloader = get_dataloader(tokenized_test, batch_size=5)
    # INFO: generate dataloader
    # test_dataloader = get_dataloader(tokenized_test, batch_size=5)
    result_from_classification = eval_loop(bert_classifier, test_dataloader, device)
    print(
        f'\nPearson correlation: {result_from_classification[0]:.2f}\nSpearman correlation: {result_from_classification[1]:.2f}')


def main4():
    # INFO: model and training configs
    model_name = 'prajjwal1/bert-tiny'
    num_epochs = 3
    train_batch_size = 8
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ## drop_out = 0.1
    bert_config = {"hidden_size": 128, "num_attention_heads": 2, "num_hidden_layers": 2, "intermediate_size": 512,
                   "vocab_size": 30522}
    bert_path = 'bert_tiny.bin'

    # WARNING: Change this code if you implemented a different nli loader for this part
    nli_dataset = load_nli_dataset('AllNLI.tsv.gz')

    # INFO: tokenize dataset
    # WARNING: Use only first 50000 samples and maximum sequence lenght of 128
    tokenized_train = tokenize_sentence_pair_dataset(nli_dataset['train'][:50000], tokenizer, max_length=128)

    # INFO: generate train_dataloader
    train_dataloader = get_dataloader(tokenized_train, batch_size=train_batch_size)

    # bert = AutoModel.from_pretrained(model_name)
    bert = Bert(bert_config).load_model('bert_tiny.bin')
    bert.eval()

    ###    Replace None with required input based on yor implementation
    bert_contrastive = BertContrastive(bert)

    # INFO: create optimizer and run training loop
    optimizer = AdamW([param for param in bert_contrastive.parameters() if param.requires_grad == True], lr=5e-5)
    train_loop(bert_contrastive, optimizer, train_dataloader, num_epochs, device)

    sts_dataset = load_sts_dataset('stsbenchmark.tsv.gz')

    # INFO: tokenize dataset
    tokenized_test = tokenize_sentence_pair_dataset(sts_dataset['test'], tokenizer)

    # INFO: generate dataloader
    test_dataloader = get_dataloader(tokenized_test, batch_size=5)

    # INFO: generate dataloader
    # test_dataloader = get_dataloader(tokenized_test, batch_size=5)
    result_from_classification = eval_loop(bert_contrastive, test_dataloader, device)
    print(
        f'\nPearson correlation: {result_from_classification[0]:.2f}\nSpearman correlation: {result_from_classification[1]:.2f}')


def main5():
    model_name = 'prajjwal1/bert-tiny'
    num_epochs = 3
    train_batch_size = 8
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_config = {"hidden_size": 128, "num_attention_heads": 2, "num_hidden_layers": 2, "intermediate_size": 512,
                   "vocab_size": 30522}
    bert_path = 'bert_tiny.bin'

    # WARNING: Change this code if you implemented a different nli loader for this part
    nli_dataset = load_nli_dataset('AllNLI.tsv.gz')

    # INFO: tokenize dataset
    # WARNING: Use only first 50000 samples and maximum sequence lenght of 128
    tokenized_train = tokenize_sentence_pair_dataset(nli_dataset['train'][:50000], tokenizer, max_length=128)

    # INFO: generate train_dataloader
    train_dataloader = get_dataloader(tokenized_train, batch_size=train_batch_size)

    tokenized_test = tokenize_sentence_pair_dataset(nli_dataset['test'], tokenizer, max_length=128)

    # INFO: generate train_dataloader
    test_dataloader = get_dataloader(tokenized_test, batch_size=8, shuffle=True)


    bert = AutoModel.from_pretrained(model_name)
    # bert = Bert(bert_config).load_model('bert_tiny.bin')
    # bert.train()

    # supreme_bert = BertClassifier(bert, pool="mean", max_length=128, num_labels=3) #SupremeBert(bert, pool="mean")
    supreme_bert = SupremeBert(bert)
    embedding_NLI = EmbedingClassifier(input_channel=512, hidden_channel=100, output_channel=3)
    supreme_bert.set_head(embedding_NLI)
    # embedding_STS =

    # supreme_bert.set_head(embedding_NLI)

    # INFO: create optimizer and run training loop
    supreme_bert.turn_off_base()
    optimizer = AdamW(supreme_bert.parameters(), lr=5e-5)
    train_loop(supreme_bert, optimizer, train_dataloader, num_epochs, device)

    result_from_classification = eval_loop(supreme_bert, test_dataloader, device)
    print(
        f'\nPearson correlation: {result_from_classification[0]:.2f}\nSpearman correlation: {result_from_classification[1]:.2f}')

    supreme_bert.turn_on_base()
    optimizer = AdamW(supreme_bert.parameters(), lr=5e-5)
    train_loop(supreme_bert, optimizer, train_dataloader, num_epochs, device)

    result_from_classification = eval_loop(supreme_bert, test_dataloader, device)
    print(
        f'\nPearson correlation: {result_from_classification[0]:.2f}\nSpearman correlation: {result_from_classification[1]:.2f}')


def plot_results():
    pass

if __name__ == "__main__":
    # main()
    # main2()
    # main3()
    main4()
    # main5()

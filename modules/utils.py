from torch.utils.data import DataLoader
import pandas as pd
import csv
from typing import Dict, List
from modules.dataset import STSDataSet
import numpy as np

def load_sts_dataset(file_name: str):
    data = pd.read_csv(file_name, on_bad_lines='skip', quoting=csv.QUOTE_NONE, compression='gzip',
                       delimiter='\t')
    train = data[data['split'] == 'train'][['score', 'sentence1', 'sentence2']]
    test = data[data['split'] == 'test'][['score', 'sentence1', 'sentence2']]
    dev = data[data['split'] == 'dev'][['score', 'sentence1', 'sentence2']]
    sts_samples = {'train': train, 'test': test, 'dev': dev}
    return sts_samples


def tokenize_sentence_pair_dataset(dataset, tokenizer, max_length=512):
    # TODO: add code to generate tokenized version of the dataset
    # tokenizer(dataset[0], return_tensors='pt', padding='max_length', max_length=max_length)
    # STSDataSet()
    score = dataset['score'].to_numpy().tolist()
    sentence1_embedding = tokenizer(dataset['sentence1'].to_numpy().tolist(), return_tensors='pt', padding='max_length',
                                    max_length=max_length)
    sentence2_embedding = tokenizer(dataset['sentence2'].to_numpy().tolist(), return_tensors='pt', padding='max_length',
                                    max_length=max_length)
    tokenized_dataset = STSDataSet(score=score, sentence1=sentence1_embedding, sentence2=sentence2_embedding)
    return tokenized_dataset


def get_dataloader(tokenized_dataset, batch_size, shuffle=False):
    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=shuffle)


def cosine_sim(a: torch.Tensor, b: torch.Tensor):
    # TODO: Implement cosine similarity function **from scrach**:
    # This method should expect two 2D matrices (batch, vector_dim) and
    # return a 2D matrix (batch, batch) that contains all pairwise cosine similarities
    return torch.zeros(a.shape[0], a.shape[0])


def eval_loop(model, eval_dataloader, device):
    # TODO: add code to for evaluation loop
    # TODO: Use cosine_sim function above as distance metric for pearsonr and spearmanr functions that are imported

    cosine = np.dot(A, B) / (norm(A) * norm(B))
    return [eval_pearson_cosine, eval_spearman_cosine]

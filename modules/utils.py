from torch.utils.data import DataLoader
import pandas as pd
import csv
import torch
from typing import Dict, List
from modules.dataset import STSDataSet, NLIDataSet
import numpy as np
from tqdm.auto import tqdm
from torch import nn
from scipy.stats import pearsonr, spearmanr


def load_sts_dataset(file_name: str):
    data = pd.read_csv(file_name, on_bad_lines='skip', quoting=csv.QUOTE_NONE, compression='gzip',
                       delimiter='\t')
    data = data.rename(columns={'score': 'label'})
    train = data[data['split'] == 'train'][['label', 'sentence1', 'sentence2']]
    test = data[data['split'] == 'test'][['label', 'sentence1', 'sentence2']]
    dev = data[data['split'] == 'dev'][['label', 'sentence1', 'sentence2']]
    sts_samples = {'train': train, 'test': test, 'dev': dev}
    return sts_samples


def tokenize_sentence_pair_dataset(dataset, tokenizer, max_length=512) -> DataLoader:
    score = dataset['label'].to_numpy().tolist()
    sentence1_embedding = tokenizer(dataset['sentence1'].to_numpy().tolist(), return_tensors='pt', padding='max_length',
                                    max_length=max_length)
    sentence2_embedding = tokenizer(dataset['sentence2'].to_numpy().tolist(), return_tensors='pt', padding='max_length',
                                    max_length=max_length)
    if type(score[0]) == str:
        tokenized_dataset = NLIDataSet(score=score, sentence1=sentence1_embedding, sentence2=sentence2_embedding)
    elif type(score[0]) == float:
        tokenized_dataset = STSDataSet(score=score, sentence1=sentence1_embedding, sentence2=sentence2_embedding)
    else:
        tokenized_dataset = STSDataSet(score=score, sentence1=sentence1_embedding, sentence2=sentence2_embedding)
        print("Warning! Are you sure about the correct dataset?")
    return tokenized_dataset


def get_dataloader(tokenized_dataset, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=shuffle)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Cosine similarity between vectors.
    :param a: torch.Tensor, [B, v_dim]
        The first embedding vector
    :param b: torch.Tensor, [B, v_dim]
        The second embedding vector
    :return similarity: torch.Tensor [B, B]
        The similarity matrix showing the similarity between vectors
    """
    similarity = a@b.T / (torch.linalg.vector_norm(a, dim=1) * torch.linalg.vector_norm(b, dim=1))
    return similarity


def eval_loop(model: nn.Module, eval_dataloader: DataLoader, device: str):
    human_score = []
    model_score = []
    model = model.to(device)
    model = model.eval()
    for score, sentence1, sentence2 in tqdm(eval_dataloader):

        sentence1_embed = model(input_ids=sentence1[0].to(device), attention_mask=sentence1[1].to(device))[1].cpu()
        sentence2_embed = model(input_ids=sentence2[0].to(device), attention_mask=sentence2[1].to(device))[1].cpu()

        cosine_similarity = cosine_sim(sentence1_embed, sentence2_embed)
        model_score.append(torch.diagonal(cosine_similarity).detach())
        human_score.append(score)

    model_score = torch.cat(model_score).detach().numpy()
    human_score = torch.cat(human_score).detach().numpy()
    eval_pearson_cosine = pearsonr(human_score, model_score).statistic
    eval_spearman_cosine = spearmanr(human_score, model_score).statistic

    return [eval_pearson_cosine, eval_spearman_cosine]


def load_nli_dataset(file_name: str):

    data = pd.read_csv(file_name, on_bad_lines='skip', quoting=csv.QUOTE_NONE, compression='gzip',
                       delimiter='\t')
    train = data[data['split'] == 'train'][['label', 'sentence1', 'sentence2']]
    test = data[data['split'] == 'test'][['label', 'sentence1', 'sentence2']]
    dev = data[data['split'] == 'dev'][['label', 'sentence1', 'sentence2']]
    nli_samples = {'train': train, 'test': test, 'dev': dev}
    return nli_samples


# A periodic eval on dev test can be added (validation_dataloader)
def train_loop(model, optimizer, train_dataloader, num_epochs, device):

    # Cross-Entropy Loss example
    loss_function = nn.CrossEntropyLoss()

    model.to(device)
    for _ in tqdm(range(num_epochs)):
        for label, sentence1, sentence2 in train_dataloader:

            output = model(sentence1, sentence2)

            loss = loss_function(output, label)

            # model_score.append(torch.diagonal(cosine_similarity).detach())
            # human_score.append(score)
  #TODO: add code to for training loop
  #TODO: use optimizer, train_dataloader, num_epoch and device for training



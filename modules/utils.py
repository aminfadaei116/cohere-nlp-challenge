from torch.utils.data import DataLoader
import pandas as pd
import csv
import torch
from typing import Dict, List, Union
from modules.dataset import STSDataSet, NLIDataSet
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from torch import nn
from scipy.stats import pearsonr, spearmanr


def load_sts_dataset(file_name: str) -> Dict:
    """
    Load the STS data loader
    :param file_name: str
        The location of the dataset.
    :return: Dict
        A dictionary that contains the dataset.
    """
    data = pd.read_csv(file_name, on_bad_lines='skip', quoting=csv.QUOTE_NONE, compression='gzip',
                       delimiter='\t')
    data = data.rename(columns={'score': 'label'})
    train = data[data['split'] == 'train'][['label', 'sentence1', 'sentence2']]
    test = data[data['split'] == 'test'][['label', 'sentence1', 'sentence2']]
    dev = data[data['split'] == 'dev'][['label', 'sentence1', 'sentence2']]
    sts_samples = {'train': train, 'test': test, 'dev': dev}
    return sts_samples


def tokenize_sentence_pair_dataset(dataset: pd.DataFrame, tokenizer: AutoTokenizer, max_length=128) -> DataLoader:
    """
    Extracts the sentences from the data frame and return the data loader.
    :param dataset: pd.DataFrame
    :param tokenizer: AutoTokenizer
        The tokenizer model
    :param max_length: int
        The maximum length of the sentence that will be tokenized.
    :return:
    """
    score = dataset['label'].to_numpy().tolist()
    sentence1_embedding = tokenizer(dataset['sentence1'].to_numpy().tolist(), return_tensors='pt', padding='max_length',
                                    max_length=max_length, truncation=True)
    sentence2_embedding = tokenizer(dataset['sentence2'].to_numpy().tolist(), return_tensors='pt', padding='max_length',
                                    max_length=max_length, truncation=True)
    if type(score[0]) == str:
        tokenized_dataset = NLIDataSet(score=score, sentence1=sentence1_embedding, sentence2=sentence2_embedding)
    elif type(score[0]) == float:
        tokenized_dataset = STSDataSet(score=score, sentence1=sentence1_embedding, sentence2=sentence2_embedding)
    else:
        tokenized_dataset = STSDataSet(score=score, sentence1=sentence1_embedding, sentence2=sentence2_embedding)
        print("Warning! Are you sure about the correct dataset?")
    return tokenized_dataset


def get_dataloader(tokenized_dataset: DataLoader, batch_size: int, shuffle: bool = False) -> DataLoader:
    """
    Modify some features of the data loader for training
    :param tokenized_dataset:
        The dataset that needs to be modified.
    :param batch_size: int
        Batch size for training the model
    :param shuffle: bool
        If the data loader should shuffle the data samples
    :return: DataLoader
        The created DataLoader
    """
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
    L2_norm = torch.linalg.vector_norm(a, dim=1).unsqueeze(1)@torch.linalg.vector_norm(b, dim=1).unsqueeze(0)
    similarity = (a@b.T) / L2_norm
    return similarity


def eval_loop(model: nn.Module, eval_dataloader: DataLoader, device: str) -> List[Union[float, float]]:
    """
    The function that will evaluate the performance of the model and will return two pearson and spearman cosine error.
    :param model: nn.Module
        The model that will be evaluated
    :param eval_dataloader: DataLoader
        The data loader that will be used for evaluation.
    :param device: str
        Which device is this process going to run on.
    :return eval_pearson_cosine: float
        The pearson error of evaluating our model
    :return eval_spearman_cosine: float
        The spearman error of evaluating our model
    """
    ground_score = []
    predict_score = []
    model = model.to(device)
    model = model.eval()
    model_name = model.__class__.__name__
    for score, sentence1, sentence2 in tqdm(eval_dataloader):
        assert model_name == 'BertClassifier' or model_name == 'BertModel' or model_name == 'Bert' or model_name == \
               'BertContrastive' or model_name == 'SupremeBert', 'Model type not valid!'
        sentence1 = convert_list(sentence1, device)  # [sentence1[0].to(device), sentence1[1].to(device)]
        sentence2 = convert_list(sentence2, device)  # [sentence2[0].to(device), sentence2[1].to(device)]

        if model_name == 'BertClassifier' or model_name == 'SupremeBert':
            predict_score.append(torch.argmax(model(sentence1, sentence2), dim=1).detach().cpu())
            ground_score.append(score.detach().cpu())
        elif model_name == 'BertContrastive':
            predict_score.append(model(sentence1, sentence2).detach().cpu())
            ground_score.append(score.detach().cpu())
        elif model_name == 'BertModel' or model_name == 'Bert':
            sentence1_embed = model(input_ids=sentence1[0].to(device), attention_mask=sentence1[1].to(device))[1].cpu()
            sentence2_embed = model(input_ids=sentence2[0].to(device), attention_mask=sentence2[1].to(device))[1].cpu()

            # sentence1_embed = torch.mean(model(input_ids=sentence1[0].to(device), attention_mask=sentence1[1].to(device))[0], dim=2).cpu()
            # sentence2_embed = torch.mean(model(input_ids=sentence2[0].to(device), attention_mask=sentence2[1].to(device))[0], dim=2).cpu()

            cosine_similarity = cosine_sim(sentence1_embed, sentence2_embed)
            predict_score.append(torch.diagonal(cosine_similarity).detach())
            ground_score.append(score.detach())

    predict_score = torch.cat(predict_score).detach().numpy()
    ground_score = torch.cat(ground_score).detach().numpy()
    eval_pearson_cosine = pearsonr(ground_score, predict_score).statistic
    eval_spearman_cosine = spearmanr(ground_score, predict_score).statistic
    return [eval_pearson_cosine, eval_spearman_cosine]


def load_nli_dataset(file_name: str) -> Dict:
    """
    Load the NLI dataset as a dictionary
    :param file_name: str
        The location of the NLI dataset
    :return: Dict
        Dictionary containing the train, test, dev segments of the dataset
    """
    data = pd.read_csv(file_name, on_bad_lines='skip', quoting=csv.QUOTE_NONE, compression='gzip',
                       delimiter='\t')
    train = data[data['split'] == 'train'][['label', 'sentence1', 'sentence2']]
    test = data[data['split'] == 'test'][['label', 'sentence1', 'sentence2']]
    dev = data[data['split'] == 'dev'][['label', 'sentence1', 'sentence2']]
    nli_samples = {'train': train, 'test': test, 'dev': dev}
    return nli_samples


def train_loop(model: nn.Module, optimizer, train_dataloader: DataLoader, num_epochs: int, device: str):
    """
    The loop for training our models
    :param model: nn.Module
        The network that we want to train
    :param optimizer: Optimizer
        The Optimization method we are using for training the model.
    :param train_dataloader: DataLoader
        The DataLoader module, for the dataset.
    :param num_epochs: int
        Number of epochs to be trained
    :param device: str
        The destination device
    :return:
    """
    assert model.model_type == 'classification' or model.model_type == 'regression', "Model type not valid!"
    if model.model_type == 'classification':
        loss_function = nn.CrossEntropyLoss()
    elif model.model_type == 'regression':
        loss_function = nn.MSELoss()
    else:
        loss_function = None
        print("Loss function not defined!")

    model.to(device)
    for _ in tqdm(range(num_epochs)):
        for label, sentence1, sentence2 in train_dataloader:
            label = label.to(device)
            if model.model_type == 'regression':
                label = label.float() - 1.0
            sentence1 = convert_list(sentence1, device)  # [sentence1[0].to(device), sentence1[1].to(device)]
            sentence2 = convert_list(sentence2, device)  # [sentence2[0].to(device), sentence2[1].to(device)]

            output = model(sentence1, sentence2)

            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def convert_list(source_list: List, device: str) -> List:
    """
    Convert the list to device (cpu or cuda), since the
    :param source_list: List
        The initial list
    :param device: str
        The destination device
    :return:
        list of the destination
    """
    return [source_list[i].to(device) for i in range(len(source_list))]

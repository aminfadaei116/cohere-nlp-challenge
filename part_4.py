import torch
from torch.optim import AdamW
import numpy as np
from transformers import AutoTokenizer
from modules.models import Bert, BertContrastive
from modules.utils import load_sts_dataset, tokenize_sentence_pair_dataset, get_dataloader, eval_loop
from modules.utils import load_nli_dataset, train_loop

torch.manual_seed(0)
np.random.seed(0)


def part_4():
    # Model Configuration
    model_name = 'prajjwal1/bert-tiny'
    num_epochs = 3
    train_batch_size = 8
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    bert_config = {"hidden_size": 128, "num_attention_heads": 2, "num_hidden_layers": 2, "intermediate_size": 512,
                   "vocab_size": 30522}

    # WARNING: Change this code if you implemented a different nli loader for this part
    nli_dataset = load_nli_dataset('data/AllNLI.tsv.gz')

    # INFO: tokenize dataset
    tokenized_train = tokenize_sentence_pair_dataset(nli_dataset['train'][:50000], tokenizer, max_length=128)

    # INFO: generate train_dataloader
    train_dataloader = get_dataloader(tokenized_train, batch_size=train_batch_size)

    bert = Bert(bert_config).load_model('data/bert_tiny.bin')

    bert_contrastive = BertContrastive(bert)

    # INFO: create optimizer and run training loop
    optimizer = AdamW([param for param in bert_contrastive.parameters() if param.requires_grad], lr=5e-5)
    train_loop(bert_contrastive, optimizer, train_dataloader, num_epochs, device)

    sts_dataset = load_sts_dataset('data/stsbenchmark.tsv.gz')

    # INFO: tokenize dataset
    tokenized_test = tokenize_sentence_pair_dataset(sts_dataset['test'], tokenizer)

    # INFO: generate dataloader
    test_dataloader = get_dataloader(tokenized_test, batch_size=5)

    # INFO: generate dataloader
    result_from_classification = eval_loop(bert_contrastive, test_dataloader, device)
    print(
        f'\nPearson correlation: {result_from_classification[0]:.2f}\nSpearman correlation: '
        f'{result_from_classification[1]:.2f}')
    result_from_classification = eval_loop(bert, test_dataloader, device)
    print(
        f'\nPearson correlation: {result_from_classification[0]:.2f}\nSpearman correlation: '
        f'{result_from_classification[1]:.2f}')


if __name__ == "__main__":
    part_4()

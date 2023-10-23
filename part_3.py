import torch
from torch.optim import AdamW
import numpy as np
from transformers import AutoTokenizer
from modules.models import Bert, BertClassifier
from modules.utils import load_sts_dataset, tokenize_sentence_pair_dataset, get_dataloader, eval_loop
from modules.utils import load_nli_dataset, train_loop

torch.manual_seed(0)
np.random.seed(0)


def part_3():
    # Model Configuration
    model_name = 'prajjwal1/bert-tiny'
    num_epochs = 3
    batch_size = 8
    num_labels = 3
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_config = {"hidden_size": 128, "num_attention_heads": 2, "num_hidden_layers": 2, "intermediate_size": 512,
                   "vocab_size": 30522}

    # INFO: load nli dataset
    nli_dataset = load_nli_dataset('data/AllNLI.tsv.gz')

    # INFO: tokenize dataset
    # WARNING: Use only first 50000 samples and maximum sequence length of 128
    tokenized_train = tokenize_sentence_pair_dataset(nli_dataset['train'][:50000], tokenizer, max_length=128)

    # INFO: generate train_dataloader
    train_dataloader = get_dataloader(tokenized_train, batch_size=batch_size, shuffle=True)

    # We will need the original Bert model for the Bert classifier
    bert = Bert(bert_config).load_model('data/bert_tiny.bin')

    bert_classifier = BertClassifier(bert, pool='mean', max_length=128, num_labels=num_labels)

    # INFO: create optimizer and run training loop
    optimizer = AdamW(bert_classifier.parameters(), lr=5e-5)
    train_loop(bert_classifier, optimizer, train_dataloader, num_epochs, device)

    sts_dataset = load_sts_dataset('data/stsbenchmark.tsv.gz')

    # INFO: tokenize dataset
    tokenized_test = tokenize_sentence_pair_dataset(sts_dataset['test'], tokenizer)

    # INFO: generate dataloader
    test_dataloader = get_dataloader(tokenized_test, batch_size=5)
    # INFO: generate dataloader
    result_from_classification = eval_loop(bert_classifier, test_dataloader, device)
    print(
        f'\nPearson correlation: {result_from_classification[0]:.2f}\nSpearman correlation: '
        f'{result_from_classification[1]:.2f}')


if __name__ == "__main__":
    part_3()

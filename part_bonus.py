import torch
from torch.optim import AdamW
import numpy as np
from transformers import AutoTokenizer
from modules.models import Bert, SupremeBert, EmbedingClassifier
from modules.utils import load_sts_dataset, tokenize_sentence_pair_dataset, get_dataloader, eval_loop
from modules.utils import load_nli_dataset, train_loop

torch.manual_seed(0)
np.random.seed(0)


def part_6():
    # Model Configuration
    model_name = 'prajjwal1/bert-tiny'
    num_epochs = 3
    train_batch_size = 8
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_config = {"hidden_size": 128, "num_attention_heads": 2, "num_hidden_layers": 2, "intermediate_size": 512,
                   "vocab_size": 30522}

    nli_dataset = load_nli_dataset('data/AllNLI.tsv.gz')

    # INFO: tokenize dataset
    tokenized_train = tokenize_sentence_pair_dataset(nli_dataset['train'][:50000], tokenizer, max_length=128)

    # INFO: generate train_dataloader
    train_dataloader = get_dataloader(tokenized_train, batch_size=train_batch_size)

    # INFO: generate train_dataloader
    sts_dataset = load_sts_dataset('data/stsbenchmark.tsv.gz')

    # INFO: tokenize dataset
    tokenized_test = tokenize_sentence_pair_dataset(sts_dataset['test'], tokenizer)
    test_dataloader = get_dataloader(tokenized_test, batch_size=5)
    bert = Bert(bert_config).load_model('data/bert_tiny.bin')

    supreme_bert = SupremeBert(bert)
    embedding_NLI = EmbedingClassifier(input_channel=512, hidden_channel=100, output_channel=3)
    supreme_bert.set_head(embedding_NLI)

    # INFO: create optimizer and run training loop
    supreme_bert.turn_off_base()
    optimizer = AdamW(supreme_bert.parameters(), lr=5e-5)
    train_loop(supreme_bert, optimizer, train_dataloader, num_epochs, device)

    result_from_classification = eval_loop(supreme_bert, test_dataloader, device)
    print(
        f'\nPearson correlation: {result_from_classification[0]:.2f}\nSpearman correlation: '
        f'{result_from_classification[1]:.2f}')

    supreme_bert.turn_on_base()
    optimizer = AdamW(supreme_bert.parameters(), lr=5e-5)
    train_loop(supreme_bert, optimizer, train_dataloader, num_epochs, device)

    result_from_classification = eval_loop(supreme_bert, test_dataloader, device)
    print(
        f'\nPearson correlation: {result_from_classification[0]:.2f}\nSpearman correlation: '
        f'{result_from_classification[1]:.2f}')


if __name__ == "__main__":
    part_6()

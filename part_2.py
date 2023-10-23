import torch
import numpy as np
from transformers import AutoTokenizer
from modules.models import Bert
from modules.utils import load_sts_dataset, tokenize_sentence_pair_dataset, get_dataloader, eval_loop

torch.manual_seed(0)
np.random.seed(0)


def part_2():
    # Model Configuration
    model_name = 'prajjwal1/bert-tiny'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    bert_config = {"hidden_size": 128, "num_attention_heads": 2, "num_hidden_layers": 2, "intermediate_size": 512,
                   "vocab_size": 30522}

    # Load the Bert Model
    bert = Bert(bert_config).load_model('data/bert_tiny.bin')

    # INFO: load dataset
    sts_dataset = load_sts_dataset('data/stsbenchmark.tsv.gz')

    # INFO: tokenize dataset
    tokenized_test = tokenize_sentence_pair_dataset(sts_dataset['test'], tokenizer)

    # INFO: generate dataloader
    test_dataloader = get_dataloader(tokenized_test, batch_size=5)

    # INFO: run evaluation loop
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    results_from_pretrained = eval_loop(bert, test_dataloader, device)

    print(
        f"\nPearson correlation: {results_from_pretrained[0]:.2f}\nSpearman correlation: "
        f"{results_from_pretrained[1]:.2f}")


if __name__ == "__main__":
    part_2()

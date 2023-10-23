import torch
import numpy as np
from transformers import AutoTokenizer
from modules.models import Bert

torch.manual_seed(0)
np.random.seed(0)


def part_1():
    # Model Configuration
    MODEL_NAME = 'prajjwal1/bert-tiny'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    bert_config = {"hidden_size": 128, "num_attention_heads": 2, "num_hidden_layers": 2, "intermediate_size": 512,
                   "vocab_size": 30522}

    bert = Bert(bert_config).load_model('data/bert_tiny.bin')

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


if __name__ == "__main__":
    part_1()

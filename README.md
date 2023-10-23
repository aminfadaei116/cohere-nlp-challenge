# Cohere-NLP-Challenge

Cohere's take-home test. Take a look at [python notebook]([https://website-name.com](https://github.com/aminfadaei116/cohere-nlp-challenge/blob/c524d7b7c6bb3d28322c6b2f3d837de38d05e8f3/Amin_Fadaeinejad_C4AIScholarsChallenge.ipynb)https://github.com/aminfadaei116/cohere-nlp-challenge/blob/c524d7b7c6bb3d28322c6b2f3d837de38d05e8f3/Amin_Fadaeinejad_C4AIScholarsChallenge.ipynb).

### If you also found it helpful, please star this repo! Thanks

Before anything, we want to download the required datasets.

```console
wget -P data https://sbert.net/datasets/stsbenchmark.tsv.gz
wget -P data https://github.com/for-ai/bert/raw/master/bert_tiny.bin
wget -P data 'https://sbert.net/datasets/AllNLI.tsv.gz'
```

## Install the requirements

```
pip install transformers
pip install torch
pip install numpy
pip install pandas
pip install scipy
```
## Coding Challenge Part 1: Debugging custom BERT code [7 points]

Tasks:
- [7 points] Your goal is to get the code working. There are 7 bugs in the code, some of them lead to error in the code but some of them are designed to impair test accuracy but not break the code. You will one point for each of the 7 bugs you find.
- [1 points] We will give extra points for also adding improved documentation to each of the functions we introduce in this section, and for describing the fixes to the bugs.

```
python part_1.py
```

## Coding Challenge Part 2: Evaluate a pretrained BERT model on STS benchmark [4 points]

In this part, we are going to evaluate a pretrained BERT model on STS benchmark without applying any additional training. For the evaluation we provide Pearson/Spearman correlation functions and cosine similarity method.

Tasks:
- [2 Points] Prepare an evaluation data loader and evaluation loop: Read in the STS data, tokenize it as shown in the example, generate the dataloader and return Pearson and Spearman correlation scores.
- [1 Point] Implement cosine similarity function, explained as TODO

```
python part_2.py
```

## Coding Challenge Part 3: Learning sentence embeddings using Natural Language Inference (NLI) dataset [4 Points]

Conneue et al. (2018) showed that a good sentence embedding model can be learned using NLI dataset. This method proposes using a shared encoder to encode both premise and hypothesis and then combine them before using a softmax classifier. Here , we will use a pretrained BERT model as shared encoder.

<img title="NLI model" alt="NLI model" src="/images/model.png">

Tasks:
- [2 Point] Prepare a training dataloader and training loop: Read in NLI data, tokenize and generate the corresponding data loader
- [2 Point] BertClassifier: Construct a model that uses above method. Please follow the architecture illustrated in the given figure.

```
python part_3.py
```

## Coding Challenge Part 4: Learning sentence embedding using a contrastive approach based on NLI dataset [3 Points]

In this part, you are asked to explore another method that leverages a contrastive approach using NLI dataset.

Tasks [3 Points] :
- Generate a dataloader if this is required for your approach
- Construct a BERT based model using a contrastive method

```
python part_4.py
```

## Coding Challenge Part 5: Comparison [1 Point]

These are the tasks [1 Point]:
- Plot the result for each model
- Explain the difference between methods and their impact on the result and comparison

```
python part_5.py
```

<img title="The Accuracy Metrics" alt="The Accuracy Metrics" src="/images/metrics.png">

## [OPTIONAL] Explore an alternative way to improve sentence encoder in terms of performance or efficiency [6 Points]

Potential directions:
- Improve the methodology to compute higher quality sentence embeddings
- Improve the efficiency during fine-tuning in terms of memory or training time
- Use different machine learning methods that leverages other resources such as auxillary/teacher models
- Use different datasets with other training objectives

```
python part_bonus.py
```


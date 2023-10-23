# cohere-nlp-challenge

Cohere's take-home test. Take a look at [python notebook]([https://website-name.com](https://github.com/aminfadaei116/cohere-nlp-challenge/blob/c524d7b7c6bb3d28322c6b2f3d837de38d05e8f3/Amin_Fadaeinejad_C4AIScholarsChallenge.ipynb)https://github.com/aminfadaei116/cohere-nlp-challenge/blob/c524d7b7c6bb3d28322c6b2f3d837de38d05e8f3/Amin_Fadaeinejad_C4AIScholarsChallenge.ipynb) for the final submission.

## If you also found it useful, please star this repo! Thanks

## Coding Challenge Part 1: Debugging custom BERT code [7 points]

Tasks:

[7 points] Your goal is to get the code working. There are 7 bugs in the code, some of them lead to error in the code but some of them are designed to impair test accuracy but not break the code. You will one point for each of the 7 bugs you find.

[1 points] We will give extra points for also adding improved documentation to each of the functions we introduce in this section, and for describing the fixes to the bugs.

## Coding Challenge Part 2: Evaluate a pretrained BERT model on STS benchmark [4 points]

In this part, we are going to evaluate a pretrained BERT model on STS benchmark without applying any additional training. For the evaluation we provide Pearson/Spearman correlation functions and cosine similarity method.

Tasks:

[2 Points] Prepare an evaluation data loader and evaluation loop: Read in the STS data, tokenize it as shown in the example, generate the dataloader and return Pearson and Spearman correlation scores.
[1 Point] Implement cosine similarity function, explained as TODO

## Coding Challenge Part 3: Learning sentence embeddings using Natural Language Inference (NLI) dataset [4 Points]

Conneue et al. (2018) showed that a good sentence embedding model can be learned using NLI dataset. This method proposes using a shared encoder to encode both premise and hypothesis and then combine them before using a softmax classifier. Here , we will use a pretrained BERT model as shared encoder.

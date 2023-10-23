from torch.utils.data import DataLoader
from typing import List


class STSDataSet(DataLoader):

    def __init__(self, score: List, sentence1, sentence2):
        assert len(score) == len(sentence1.data['input_ids']) and len(score) == len(sentence2.data['input_ids']), \
            "Sentences don't have the same size."
        self.score = score
        self.sentence1_id = sentence1.data['input_ids']
        self.sentence2_id = sentence2.data['input_ids']
        self.sentence1_mask = sentence1.data['attention_mask']
        self.sentence2_mask = sentence2.data['attention_mask']

    def __len__(self):
        return len(self.score)

    def __getitem__(self, item):
        return self.score[item], (self.sentence1_id[item], self.sentence1_mask[item]), (self.sentence2_id[item],
                                                                                        self.sentence2_mask[item])


class NLIDataSet(DataLoader):

    def __init__(self, score: List, sentence1, sentence2):
        assert len(score) == len(sentence1.data['input_ids']) and len(score) == len(sentence2.data['input_ids']), \
            "Sentences don't have the same size."
        classes = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
        score = [classes.get(key) for key in score]

        self.score = score
        self.sentence1_id = sentence1.data['input_ids']
        self.sentence2_id = sentence2.data['input_ids']
        self.sentence1_mask = sentence1.data['attention_mask']
        self.sentence2_mask = sentence2.data['attention_mask']

    def __len__(self):
        return len(self.score)

    def __getitem__(self, item):
        return self.score[item], (self.sentence1_id[item], self.sentence1_mask[item]), (self.sentence2_id[item],
                                                                                        self.sentence2_mask[item])

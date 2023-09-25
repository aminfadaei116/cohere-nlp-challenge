from torch.utils.data import DataLoader


class STSDataSet(DataLoader):

    def __init__(self, score, sentence1, sentence2):
        assert len(score) == len(sentence1.data['input_ids']) and len(score) == len(sentence2.data['input_ids']), "Sentences don't have the same size."
        self.score = score
        self.sentence1 = sentence1
        self.sentence2 = sentence2

    def __len__(self):
        len(self.score)

    def __getitem__(self, item):
        return self.score, self.sentence1, self.sentence2

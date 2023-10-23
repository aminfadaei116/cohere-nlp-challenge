import pandas as pd


def part_5():
    test_df = pd.DataFrame({'Metrics': ['Pearson correlation', 'Spearman correlation'],
                            'Pretrained BERT': [0.32, 0.33],
                            'BertClassifier max pool': [0.30, 0.29],
                            'BertClassifier mean pool': [0.11, 0.11],
                            'BertClassifier layer pool': [0.42, 0.42],
                            'Bert model in BertClassifier': [0.60, 0.61],
                            'BertContrastive': [0.33, 0.33]})

    # set name as the index and then Transpose the dataframe
    test_df = test_df.set_index('Metrics').T

    # plot and annotate
    p1 = test_df.plot(kind="barh")

    for p in p1.containers:
        p1.bar_label(p, fmt='%.2f', label_type='edge')


if __name__ == "__main__":
    part_5()

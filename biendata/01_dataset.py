import warnings
warnings.simplefilter('ignore')
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

train = pd.read_json('../raw_data/bmes_train.json')
train_data = list()
for i, row in tqdm(train.iterrows()):
    entities = row['entities']
    entities = sorted(entities, key=lambda x: -len(x))
    text = row['text']
    id_list = [i] * len(text)
    label_list = ["O"] * len(text)
    for entity in entities:
        name, label = entity.split('-')
        for m in re.finditer(name, text):
            start = m.start()
            end = m.end()
            if any([label != 'O' for label in label_list[start: end]]):
                continue
            label_list[start] = f'B-{label}'
            for i in range(start+1, end):
                label_list[i] = f'I-{label}'
    train_data.extend(zip(id_list, text, label_list))

train_data = pd.DataFrame(
    train_data, columns=["sentence_idx", "word", "tag"]
)

test = pd.read_json('../raw_data/bmes_test.json')
test_data = list()
for i, row in tqdm(test.iterrows()):
    text = row['text']
    id_list = [i] * len(text)
    label_list = ["O"] * len(text)
    test_data.extend(zip(id_list, text, label_list))

test_data = pd.DataFrame(
    test_data, columns=["sentence_idx", "word", "tag"]
)

train_df = train_data[train_data.sentence_idx % 10 != 0]
valid_df = train_data[train_data.sentence_idx % 10 == 0]

def get_samples(df):
    samples = list()
    idx_ = df.iloc[0]['sentence_idx']

    for i, row in tqdm(df.iterrows()):
        idx = row['sentence_idx']
        char = row['word']
        tag = row['tag']
        if idx != idx_:
            samples.append('')
            samples.append(f'{char} {tag}')
            idx_ = idx
        else:
            samples.append(f'{char} {tag}')
            
    return samples


train_samples = get_samples(train_df)
valid_samples = get_samples(valid_df)
test_samples = get_samples(test_df)

with open('../datasets/bien/train.char.bmes', 'w') as handler:
    handler.write("\n".join(train_samples))
    
with open('../datasets/bien/dev.char.bmes', 'w') as handler:
    handler.write("\n".join(valid_samples))
    
with open('../datasets/bien/test.char.bmes', 'w') as handler:
    handler.write("\n".join(test_samples))

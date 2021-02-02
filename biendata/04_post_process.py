import warnings
warnings.simplefilter('ignore')

import re
import json
from collections import Counter

import pandas as pd
from tqdm import tqdm

train = pd.read_json('../raw_data/bmes_train.json')
test = pd.read_json('../raw_data/bmes_test.json')
sub = pd.read_json('./sub_ensemble.json')

train_entities = Counter()
for i, row in train.iterrows():
    for ent in row['entities']:
        train_entities[ent] += 1

# 只处理 TER 标签且文本长度 >=4 且在训练集中出现 > 1 次的 避免误杀
ter_keys = [ent for ent in train_entities if ent.endswith('-TER') and len(ent) >= 4+4 and train_entities[ent] > 1] + to_use
ter_map = list()

for i, row in tqdm(test.iterrows()):
    li = list()
    for key in [k for k in ter_keys if k not in not_use + not_use2]:
        if row['text'].find(key[:-4]) != -1:
            li.append(key)
    # 只收录最长的
    if len(li) > 1:
        for k1 in li:
            for k2 in [k for k in li if k != k1]:
                if k2.find(k1) != -1:
                    li.remove(k1)
    ter_map.append(li)

sub2 = pd.read_json('sub_ensemble.json')

res = list()
for i, row in tqdm(sub2.iterrows()):
    ters = ter_map[i]
    ents = row['entities']
    if ters:
        for ter in ters:
            if ter not in ents:
                for ent in ents:
                    if ter.find(ent) >= 0: # 存在包含情况
                        ents.remove(ent)
                ents.append(ter)
    # 保持下顺序用来做前后对比 list(set(ents)) 会乱
    ret = list()
    for ent in ents:
        if ent not in ret:
            ret.append(ent)
    res.append(ret)
    
sub2['entities'] = res
sub2.to_json('sub_ensemble_post.json', orient='records', force_ascii=False)

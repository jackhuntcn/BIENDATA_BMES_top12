import pandas as pd

from glob import glob
import re
import json
from collections import defaultdict, Counter

filenames = glob("preds_dir/*.json")

d = {}
for filename in filenames:
    basename = os.path.basename(filename)
    ents = pd.read_json(filename)['entities']
    d[basename] = ents

df = pd.DataFrame(d)

test = pd.read_json('../raw_data/bmes_test.json')

cols = list(df)
df['text'] = test.text

for col in cols:
    s = df[col]

    ## 如果在同一个sample中有重复的预测，则删除较长的预测
    res_new2 = []
    for idx, (ents, text) in enumerate(zip(s.values, test.text.values)):
        remove_ents = []
        for ent_a in ents:
            ent_a_text, ent_a_label = ent_a.rsplit('-', 1)
            for ent_b in ents:
                ent_b_text, ent_b_label = ent_b.rsplit('-', 1)
                if ent_a_label == ent_b_label and ent_a_text != ent_b_text and ent_a_text in ent_b_text:
                    if ent_a_text not in re.sub(ent_b_text, ' ', text):
                        # 如果entity在原文中出现次数超过1次，则认为是真实的entity，否则删除
                        if text.count(ent_b_text) > 1:
                            remove_ents.append(ent_a)
                        else:
                            remove_ents.append(ent_b)
        if remove_ents:
            ents = [ent for ent in ents if ent not in remove_ents]
        
        
        ents_new = []
        for ent in ents:
            ent_text, ent_label = ent.rsplit('-', 1)
            if len(ent_text) > 15:
                print(idx, ent)
            elif ent_text in text:
                # 如果有以下字符，则分割实体
                if re.search(r'(、|至|，|（)', ent_text):
                    ent_text_splits = re.sub(r'(、|至|，|（)', ' ', ent_text).split(' ')
                    for ent_text_split in ent_text_splits:
                        ent_new = f'{ent_text_split}-{ent_label}'
                        if ent_new not in ents_new:
                            ents_new.append(ent_new)
                elif ent not in ents_new:
                    ents_new.append(ent)
            else:
                print(ent)
        res_new2.append(ents_new)
    
    df[col] = res_new2

df = df.drop('text', axis=1)

res = []
diffs = []
tmp = []
for idx, row in df.iterrows():
    inv_counter = Counter()
    counts = row.value_counts()
    if len(counts) == 1:
        res.append(counts.index[0])
    else:
        counts2 = counts[counts >= counts[0] - 1]
        minlen, ents = 100, None
        for cand in counts2.index:
            if len(cand) < minlen:
                ents = cand
                minlen = len(cand)
        for ents_row in row.values:
            for ent in list(set(ents_row)):
                inv_counter[ent] += 1
        new_ents = []
        for ent in ents:
            if ent[-3:] == 'RIV' and inv_counter[ent] > 7:
                new_ents.append(ent)
            elif ent[-3:] != 'RIV' and inv_counter[ent] > 6:
                new_ents.append(ent)
        tmp.append((idx, new_ents))
        if new_ents != ents:
            diffs.append((idx, ents, new_ents))
        res.append(new_ents)

res = [{"id": f'test_{idx}', 'entities': ents} for idx, ents in enumerate(res)]
with open("sub_ensemble.json", 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=4, ensure_ascii=False)

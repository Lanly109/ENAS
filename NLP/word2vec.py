import io
import numpy as np
from tqdm import tqdm
import csv
import json

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    return int(d), data

n, data = load_vectors("wiki-news-300d-1M.vec")

texts = []

with open('normolize.csv', 'r') as f:
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        text = row[0].split(' ')
        emo = row[1].strip()
        emo = 1 if emo[0] == 'p' else 0
        texts.append((text, emo))

word2vec = []
for text, emo in tqdm(texts):
    tmp = dict()
    tmp['text'] = np.zeros(n)
    tmp['lable'] = emo
    cnt = 0
    for word in text:
        try:
            tmp['text'] += data[word]
        except Exception as e:
            cnt += 1
    tmp['text'] /= len(text)
    tmp['text'] = tmp['text'].tolist()
    tmp['missing word'] = cnt
    word2vec.append(tmp)

print("ok, begin writing...")

with open("word2vec_train.json", "w") as f:
    f.write(json.dumps(word2vec[:30000] ,sort_keys=True, indent=4, separators=(',', ': ')))

with open("word2vec_test.json", "w") as f:
    f.write(json.dumps(word2vec[30000:40000] ,sort_keys=True, indent=4, separators=(',', ': ')))

with open("word2vec_check.json", "w") as f:
    f.write(json.dumps(word2vec[40000:] ,sort_keys=True, indent=4, separators=(',', ': ')))

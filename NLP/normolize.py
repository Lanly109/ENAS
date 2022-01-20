import re
import csv
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm
import json

data = []
stopword = ['', ' ']
stem = PorterStemmer()
lem = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

replacement_patterns = [
(r'won\'t', 'will not'),
(r'can\'t', 'cannot'),
(r'i\'m', 'i am'),
(r'ain\'t', 'is not'),
(r'(\w+)\'ll', '\g<1> will'),
(r'(\w+)n\'t', '\g<1> not'),
(r'(\w+)\'ve', '\g<1> have'),
(r'(\w+)\'s', '\g<1> is'),
(r'(\w+)\'re', '\g<1> are'),
(r'(\w+)\'d', '\g<1> would')]

class RegexpReplacer(object):
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            (s, _) = re.subn(pattern, repl, s)
        return s

with open('IMDB Dataset.csv', 'r') as f:
    reader = csv.reader(f, delimiter=",")
    next(reader)
    rex = RegexpReplacer()
    for row in tqdm(reader):
        text = row[0].replace(r"<br />", " ").lower()
        text = rex.replace(text)
        text = re.sub(r"[^a-z ]", " ", text)
        text = [i for i in text.split(' ') if i not in stopword]
        tag = pos_tag(text)
        text = [lem.lemmatize(word, pos=get_wordnet_pos(pos) or wordnet.NOUN) for word, pos in tag]
        # text = [stem.stem(lem.lemmatize(word, pos=get_wordnet_pos(pos) or wordnet.NOUN)) for word, pos in tag]
        label = row[1]
        data.append((text, label))

with open('normolize.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    for text, emo in data:
        writer.writerow([' '.join(text), emo])

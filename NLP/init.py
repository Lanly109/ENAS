import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords

stopword = stopwords.words('english')

with open('stopword.txt', 'w') as f:
    f.write(' '.join(stopword))

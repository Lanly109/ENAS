# NLP 

NLP reviews data [download](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) saved as `IMDB Dataset.csv`

word2vec model [download](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip) saved as `wiki-news-300d-1M.vec`

## Run

```bash
# install requirements
pip3 install -r requirements.txt

# data download（may need magic）
python3 init.py

# clean text
python3 normalize.py

# prepare training data
python3 word2vec.py
```

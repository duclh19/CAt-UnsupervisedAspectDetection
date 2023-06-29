"""
To create word2vec.vec with training dataset of CitySearch (tokenized)
"""

import json
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from gensim.models import Word2Vec
from collections import defaultdict, Counter
from reach import Reach
from tqdm import tqdm 

file_path = 'data/citysearch/train.txt'

def generate_nouns(file_path, 
                   word2vec="embeddings/restaurant_vecs_w2v.vec", 
                   out_path='data/nouns_restaurant.json'
                   ): 
    print('\tStarting generate nouns ... ')
    with open(file_path, 'r') as f:
        text = f.readlines()
    nouns = []
    noun_counts = defaultdict(int)

    for sentence in tqdm(text):
        words = word_tokenize(sentence)
        tagged_words = pos_tag(words)
        nouns = [word for word, pos in tagged_words if pos[0] == 'N']
        for noun in nouns:
            noun_counts[noun] +=1
    
    r = Reach.load(word2vec,
                   unk_word="<UNK>")
    nouns_dict = Counter()
    for k, v in noun_counts.items():
        if k.lower() in r.items:
            nouns_dict[k.lower()] += v
    # candidates, _ = zip(*nouns_dict.most_common(topk))
    json.dump(nouns_dict, open(out_path, 'w'))
    print(f"\tFinishing generate nouns, output path: {out_path}")
    # return candidates


def word2vec(file_path, output="embeddings/restaurant_vecs_w2v.vec"): 
    corpus = [x.lower().strip().split() for x in open(file_path)]
    
    scores = defaultdict(dict)
    f = Word2Vec(corpus, 
                 negative=5,
                 window=10,
                 vector_size=200,
                 min_count=1,
                 epochs=5,
                 workers=10)
    f.wv.save_word2vec_format(output)

if __name__ == "__main__": 
    word2vec(file_path=file_path)
    generate_nouns(file_path=file_path)
    
import json
# from typing import Counter

from cat.simple import get_nouns, get_scores, rbf_attention, attention, get_aspect
from reach import Reach
from collections import defaultdict, Counter



GAMMA       = .03
N_NOUNS     = 200
w2v_path    = "embeddings/restaurant_vecs_w2v.vec"
nouns_path  = "data/nouns_restaurant.json"

if __name__ == "__main__":
    print("\tLoading words embedding ...")
    w2v = Reach.load(w2v_path, unk_word="<UNK>")     
    
    print("\tLoading top most frequent nouns ... ")
    top_nouns = get_nouns(w2v, nouns_path, N_NOUNS)

    ### TESTING 

    # sentences = ["The seafood is so fresh, but the hotdog is much more better".split(), 
    #              "the waiter is friendly but he is too short".split(), 
    #              ]
    sentences = ["The restaurant is modern".split()]
    label_set = ['food', 'staff', 'ambience']

    s = get_scores(sentences,
                   top_nouns,
                   w2v,
                   label_set,
                   gamma=GAMMA,
                   attention_func=rbf_attention)

    pred = s.argmax(1)

    label_pred = get_aspect(label_set, pred)
    print(label_pred)


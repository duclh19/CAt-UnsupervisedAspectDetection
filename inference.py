import json
# from typing import Counter

from cat.simple import get_nouns, get_scores, rbf_attention, attention, get_aspect, softmax
from reach import Reach
from collections import defaultdict, Counter
import numpy as np

# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum(axis=0)

att         = rbf_attention # `rbf_attention` or `attention`
GAMMA       = .03           # if attention then GAMMA is not in use
N_NOUNS     = 200           # 200 for rbf_attention, 950 for attention
w2v_path = "embeddings/w2v_restaurant_200_ep_5.vec"
nouns_path = "data/nouns_restaurant_200_ep_5.json"

if __name__ == "__main__":
    print("\tLoading words embedding ...")
    w2v = Reach.load(w2v_path, unk_word="<UNK>")     
    
    print("\tLoading top most frequent nouns ... ")
    top_nouns = get_nouns(w2v, nouns_path, N_NOUNS)

    ### TESTING 

    sentences = ["The seafood is so fresh, but the hotdog is much more better".split(), 
                 "the waiter is friendly but he is too short".split(), 
                 ]
    # sentences = [
    #     "The bread is top notch as well.".split(), 
    # ]
    # sentences = [
        # "The design and atmosphere is just as good.".split()
    # ]
    label_set = ['food', 'staff', 'ambience']

    s = get_scores(sentences,
                   top_nouns,
                   w2v,
                   label_set,
                   gamma=GAMMA,
                   attention_func=att)
    logit = softmax(s)
    pred = logit.argmax(1)
    
    label_pred = get_aspect(label_set, pred)
    print(label_pred, logit)
    # print(label_pred)


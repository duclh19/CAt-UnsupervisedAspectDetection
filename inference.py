import json
# from typing import Counter

from cat.simple import get_scores, rbf_attention, attention, normalize
from reach import Reach
from collections import defaultdict, Counter



GAMMA = .03
N_ASPECT_WORDS = 200


if __name__ == "__main__":

    """
        scores      : 
        r           : 
        aspects     : top N_ASPECT_WORDS
        instances   : list of senctences, each sentence is a list of words
        lable_set   : 
    """


    scores = defaultdict(dict)
    r = Reach.load("embeddings/restaurant_vecs_w2v.vec",
                   unk_word="<UNK>")                        ## load word embedding 
    

    nouns_restaurant = [[x] for x in json.load(open("data/nouns_restaurant.json"))]
    nouns = Counter()
    for k, v in nouns_restaurant.items():
        if k.lower() in r.items:
            nouns[k.lower()] += v

    top_nouns, _ = zip(*nouns.most_common(N_ASPECT_WORDS))
    top_nouns = [[x] for x in top_nouns]
    
    instances = ["text_1".split(), "text_2".split()]
    label_set = ['food', 'staff', 'ambience']

    s = get_scores(instances,
                   top_nouns,
                   r,
                   label_set,
                   gamma=GAMMA,
                   remove_oov=False,
                   attention_func=rbf_attention)

    pred = s.argmax(1)



    
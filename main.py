import json
from pprint import pprint

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score, precision_score, recall_score

from cat.simple import get_scores, rbf_attention, attention
from cat.dataset import semeval_loader, citysearch_loader
from collections import defaultdict, Counter
from reach import Reach
gamma = [0.01, 0.03, 0.07, 0.1, 0.15, 0.04, 0.3]
best = 0
N_NOUNS = 200

scores = defaultdict(dict)
r = Reach.load("embeddings/restaurant_vecs_w2v.vec",
                unk_word="<UNK>")

d = json.load(open("data/nouns_restaurant.json"))


nouns = Counter()
## For each key, value pair in dictionary of restaurant nouns, if k is known -> add to nouns
# Remove unknown nouns from nous_restaurant.json.
for k, v in d.items():
    if k.lower() in r.items:
        nouns[k.lower()] += v

embedding_paths = ["embeddings/restaurant_vecs_w2v.vec"]

    # bundles = ((rbf_attention, attention), embedding_paths)

    # for att, path in product(*bundles):
        # r = Reach.load(path, unk_word="<UNK>")

        # Get top candidates - top nouns 
        # if att == rbf_attention:
candidates, _ = zip(*nouns.most_common(N_NOUNS))
        # else:
        #     candidates, _ = zip(*nouns.most_common(BEST_ATT["n_noun"]))

        # let each candidate is an aspect
aspects = [[x] for x in candidates]



for idx, (instances, y, label_set) in enumerate(semeval_loader()):
    for gamma_ in gamma:
        s = get_scores(instances,
                        aspects,
                        r,
                        label_set,
                        gamma=gamma_,
                        remove_oov=False,
                        attention_func=rbf_attention)
        y_pred = s.argmax(1)
        f1 = f1_score(y, y_pred, average='weighted',)
        print(gamma_, f1)
        if f1 > best:
            best = f1
            print(gamma_)
            # print('score head shape:', s.shape)

            # print(y_pred[:10])
            # print(y[:10], '\n')

            f1 = precision_recall_fscore_support(y, y_pred, average=None)

            f1_macro = precision_recall_fscore_support(y, y_pred, average='macro')
            f1_weighted = precision_recall_fscore_support(y,
                                                       y_pred,
                                                       average="weighted")
            f1_micro = precision_recall_fscore_support(y,
                                                       y_pred,
                                                       average="micro")
            print('f1:')
            pprint(f1)
            print('f1_macro')
            pprint(f1_macro)

            print('f1_weighted')
            pprint(f1_weighted)

            print('f1_micro\n', f1_micro)

            print('-----' * 5)
            print('-----' * 5)
            f1 = f1_score(y, y_pred, average='weighted',)

            print(f1)
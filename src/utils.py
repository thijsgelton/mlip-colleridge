import re
import numpy as np
from pathlib import Path
import collections
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


def read_tags(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    contains_ds = False
    for doc in raw_docs:
        if not doc:
            continue
        tokens = []
        tags = []
        for line in doc.strip().split('\n'):
            token, tag = line.split()
            tokens.append(token)
            tags.append(tag)
        if 'B-DS' in tags:
            contains_ds = True
            token_docs.append(tokens)
            tag_docs.append(tags)
        # if 'B-DS' not in tags and contains_ds:
        #     contains_ds = False
        #     token_docs.extend(tokens)
        #     tag_docs.extend(tags)
    return token_docs, tag_docs


def read_tags_with_window(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    token_window = collections.deque(maxlen=3)
    tag_window = collections.deque(maxlen=3)
    token_window.append(['placeholder'])
    tag_window.append(['O'])
    for doc in raw_docs:
        if not doc:
            continue
        tokens = []
        tags = []
        for line in doc.strip().split('\n'):
            token, tag = line.split()
            tokens.append(token)
            tags.append(tag)
        token_window.append(tokens)
        tag_window.append(tags)
        if 'B-DS' in tag_window[1]:
            token_docs.extend(token_window)
            token_window.popleft()
            tag_docs.extend(tag_window)
            tag_window.popleft()
    return token_docs, tag_docs


token_docs, tag_docs = read_tags("../conll_formatted_annotated_training_corpus_train_0.8.txt")
# class_weights = compute_class_weight('balanced', np.array(['O', 'B-DS', 'I-DS']), [y for x in tag_docs for y in x])
# print(class_weights)
# print(np.unique([y for x in tag_docs for y in x], return_counts=True))


with open("../balanced_only_bds_conll_formatted_annotated_training_corpus_train_0.8.txt", "w") as f:
    for tokens, tags in zip(token_docs, tag_docs):
        f.writelines([f"{token} {tag}\n" for token, tag in zip(tokens, tags)])
        f.write("\n")


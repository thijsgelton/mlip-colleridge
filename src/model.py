import json
import logging
import os
import random
import re
from itertools import chain

import numpy as np
import pandas as pd
import torch
import wandb
from simpletransformers.ner import NERModel, NERArgs
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import compute_class_weight

random.seed(123)
np.random.seed(456)

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.NOTSET)

# Configure the model
model_args = NERArgs()
model_args.num_train_epochs = 4
model_args.classification_report = True
model_args.overwrite_output_dir = True
model_args.train_batch_size = 16
model_args.lazy_loading = True
model_args.max_seq_length = 128
model_args.reprocess_input_data = True
model_args.evaluate_during_training = True
model_args.eval_batch_size = 32
model_args.use_multiprocessing = True  # Set to false for prediction
model_args.use_early_stopping = True
model_args.early_stopping_delta = 0.0001
model_args.early_stopping_metric_minimize = True
model_args.early_stopping_patience = 5
model_args.evaluate_during_training_steps = 800
model_args.wandb_project = "SciBert-colleridge"
model_args.wandb_kwargs = {"resume": True}
model_args.learning_rate = 1e-5
model_args.custom_parameter_groups = [
    {
        "params": ["classifier.weight"],
        "lr": 3e-4,
    },
    {
        "params": ["classifier.bias"],
        "lr": 3e-4
    },
]

custom_labels = ["O", "B-DS", "I-DS"]

rob_ner = NERModel(
    "auto", "allenai/scibert_scivocab_uncased", args=model_args, labels=custom_labels,
    class_weights=torch.tensor([0.36544896, 10.21214971, 6.03437623], dtype=torch.float32)
)
print(rob_ner.get_named_parameters())


def clean_text(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt))


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def deduplicate(predicted_labels):
    filtered_pred_labels = []
    for labels in predicted_labels:
        filtered = []

        for label in sorted(labels, key=len):
            label = clean_text(label).lower()
            if len(filtered) == 0 or all(jaccard(label, got_label) < 0.50 for got_label in filtered):
                filtered.append(label)

        filtered_pred_labels.append('|'.join(filtered))
    return filtered_pred_labels


if not model_args.use_multiprocessing:
    test_results = pd.DataFrame(columns=['Id', 'PredictionString'])
    for test_json in os.listdir("../Data analysis/test"):
        paper = json.loads(open(f"../Data analysis/test/{test_json}").read())
        sentences = [clean_text(sentence) for section in paper for sentence in section['text'].split(". ")]
        predictions, raw_output = rob_ner.predict(sentences)
        tokens = []
        tags = []
        paper_datasets_tokens = []
        paper_datasets_tags = []
        for pred in predictions:
            pred_tokens, pred_tags = [], []
            if "B-DS" in [list(single.values())[0] for single in pred]:
                for single in pred:
                    if len({'B-DS', 'I-DS'}.intersection(set(single.values()))) > 0:
                        pred_tokens.append(list(single.keys())[0])
                        pred_tags.append(list(single.values())[0])
            if len(pred_tokens) > 0 and len(pred_tags) > 0:
                paper_datasets_tokens.append(pred_tokens)
                paper_datasets_tags.append(pred_tags)

        test_results = test_results.append({'Id': test_json.strip(".json"),
                                            "PredictionString":
                                                deduplicate([[" ".join(pred) for pred in paper_datasets_tokens]])[0]},
                                           ignore_index=True)
    print(test_results.head(100))
else:
    mlb = MultiLabelBinarizer().fit([custom_labels])
    le = LabelEncoder().fit(custom_labels)


    def custom_f1(y_true, y_pred):
        y_true = mlb.transform(y_true)
        y_pred = mlb.transform(y_pred)
        return f1_score(y_true, y_pred, average="weighted")


    def flat_accuracy(y_true, y_pred):
        wandb.log(
            {"confusion_matrix": wandb.plot.confusion_matrix(y_true=le.transform(list(chain.from_iterable(y_true))),
                                                             preds=le.transform(list(chain.from_iterable(y_pred))),
                                                             class_names=sorted(custom_labels))})

        pred_flat = np.array(y_pred).flatten()
        labels_flat = np.array(y_true).flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)


    # Train the model
    rob_ner.train_model(
        train_data="../balanced_only_bds_conll_formatted_annotated_training_corpus_train_0.8.txt",
        eval_data="../balanced_only_bds_conll_formatted_annotated_training_corpus_eval_0.2.txt",
        flat_accuracy=flat_accuracy
    )

import re
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, BertForTokenClassification

MAX_LENGTH = 128


def read_tags(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        if not doc:
            continue
        tokens = []
        tags = []
        for line in doc.strip().split('\n'):
            token, tag = line.split()
            tokens.append(token)
            tags.append(tag)
        if 'B-DS' in tags and len(tags) < MAX_LENGTH - 64:
            token_docs.append(tokens)
            tag_docs.append(tags)
    return token_docs, tag_docs


train_tokens, train_tags = read_tags(r'..\balanced_only_bds_conll_formatted_annotated_training_corpus_train_0.8.txt')
val_tokens, val_tags = read_tags(r'..\balanced_only_bds_conll_formatted_annotated_training_corpus_eval_0.2.txt')

print(train_tokens[0][0:17], train_tags[0][0:17], sep='\n')

unique_tags = set(tag for doc in train_tags for tag in doc)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

label_list = [
    "O",
    "B-DS",
    "I-DS"
]
token_settings = dict(is_split_into_words=True, return_offsets_mapping=True, padding="max_length", truncation=True,
                      max_length=MAX_LENGTH)
train_encodings = tokenizer(train_tokens, **token_settings)
val_encodings = tokenizer(val_tokens, **token_settings)


def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels


train_labels = encode_tags(train_tags, train_encodings)
val_labels = encode_tags(val_tags, val_encodings)


class WNUTDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_encodings.pop("offset_mapping")  # we don't want to pass this to the model
val_encodings.pop("offset_mapping")
train_dataset = WNUTDataset(train_encodings, train_labels)
val_dataset = WNUTDataset(val_encodings, val_labels)

model = BertForTokenClassification.from_pretrained("allenai/scibert_scivocab_uncased", num_labels=len(unique_tags))


class BalancedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super(BalancedTrainer, self).__init__(*args, **kwargs)
        self.class_weights = torch.from_numpy(class_weights).type(torch.float32).to(torch.device("cuda:0"))

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, len(label_list)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


training_args = TrainingArguments(
    output_dir='./results',  # output directory
    num_train_epochs=3,  # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./logs',  # directory for storing logs
    logging_steps=10,
    report_to="wandb",
    run_name="scibert_uncased"
)
print(training_args.device)
class_weights = compute_class_weight('balanced', classes=label_list, y=list(chain(*train_tags)))

trainer = BalancedTrainer(
    class_weights=class_weights,
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=val_dataset  # evaluation dataset
)

trainer.train()

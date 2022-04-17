import argparse
import json
import os
from collections import defaultdict
from functools import partial
from functools import reduce
from lib2to3.pgen2.tokenize import tokenize

import data_utils
import dataset
import finetuning_utils
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from torch.nn.utils import prune
from transformers import RobertaForSequenceClassification
from transformers import RobertaTokenizer
from transformers import Trainer
from transformers import TrainingArguments
from yaml import load


parser = argparse.ArgumentParser(description="Bag RoBERTa model on the CB dataset.")
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    help="Directory containing the relevant SuperGLUE dataset.",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="Directory containing the relevant SuperGLUE dataset.",
)
args = parser.parse_args()

print("Data is in: ", args.data_dir)


task_name = os.path.basename(args.data_dir)

if task_name != "MultiRC":
    train_df = pd.read_json(
        f"{args.data_dir}/train.jsonl", lines=True, orient="records"
    )
    val_df, test_df = train_test_split(
        pd.read_json(f"{args.data_dir}/val.jsonl", lines=True, orient="records"),
        test_size=0.5,
    )
else:
    train_df = data_utils.process_multirc_jsonl(f"{args.data_dir}/train.jsonl", " ")
    val_df, test_df = train_test_split(
        data_utils.process_multirc_jsonl(f"{args.data_dir}/val.jsonl", " "),
        test_size=0.5,
    )


def load_models(model_dir):
    print("Loading models from " + str(model_dir))
    models = []
    tokenizers = []
    for directory in os.listdir(model_dir):
        print("dir" + directory)
        if not os.path.isdir(model_dir+"/"+directory+"/"):
            continue
        directory = model_dir+"/"+directory+"/"
        model = RobertaForSequenceClassification.from_pretrained(directory)
        tokenizer = RobertaTokenizer.from_pretrained(directory)
        models.append(model)
        tokenizers.append(tokenizer)
    return models, tokenizers


def get_module_by_name(module, access_string):
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_nonzero(model):
    return sum(torch.count_nonzero(p) for p in model.parameters() if p.requires_grad)


def prune_models(models):
    for model in models:
        print(count_parameters(model), count_nonzero(model))
        module_tups = []
        for layer in list(model.state_dict().keys()):
            if ".weight" not in layer:
                continue
            x = layer.split(".weight")[0]
            module_tups.append((get_module_by_name(model, x), "weight"))
        prune.global_unstructured(
            parameters=module_tups, pruning_method=prune.L1Unstructured, amount=0.5
        )

        for module, _ in module_tups:
            prune.remove(module, "weight")
        print(count_parameters(model), count_nonzero(model))


def run_evaluation(models, task_name, test_df):
    y_preds = []
    inputs = test_df['question_answer_concat']
    y_true = test_df['label']
    for input in inputs:
        votingDict = defaultdict(int)
        for model in models:
            if task_name == "MultiRC":
                with torch.no_grad():
                    logits = model(**input).logits

                predicted_class_id = int(torch.argmax(logits, axis=-1)[0])
                votingDict[model.config.id2label[predicted_class_id]] += 1
            else:
                y_pred = max(votingDict, key=votingDict.get)
                y_preds.append(y_pred)
    return precision_recall_fscore_support(y_true, y_preds, average='macro')


models = load_models("../models/")
print("Models are - " + str(models))
models = prune_models(models)
print(run_evaluation(models, task_name, test_df))

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
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.nn.utils import prune
from transformers import GPT2ForSequenceClassification
from transformers import GPT2Tokenizer
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

    test_df = pd.read_json(f"{args.data_dir}/val.jsonl", lines=True, orient="records")

else:
    test_df = data_utils.process_multirc_jsonl(f"{args.data_dir}/val.jsonl", " ")


tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


def load_models(model_dir):
    print("Loading models from " + str(model_dir))
    models = []
    for directory in os.listdir(model_dir):
        print("dir" + directory)
        if not os.path.isdir(model_dir + "/" + directory + "/"):
            continue
        directory = model_dir + "/" + directory + "/"
        model = RobertaForSequenceClassification.from_pretrained(
            directory, problem_type="multi_label_classification"
        )
        print("first model - " + str(type(model)))
        models.append(model)
    return models


def get_module_by_name(module, access_string):
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_nonzero(model):
    return sum(torch.count_nonzero(p) for p in model.parameters() if p.requires_grad)


def prune_models(models):
    new_models = []
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
        new_models.append(model)
        print(count_parameters(model), count_nonzero(model))
    print("models pruned!")
    return new_models


import numpy as np


def run_evaluation(models, task_name, test_df):
    y_preds = []
    y_true = data_utils.extract_labels(test_df, task_name)[:25]

    for i in range(len(y_true)):
        votingDict = defaultdict(int)
        summed_probs = []
        for model in models:
            if task_name == "MultiRC" or task_name == "BoolQ":
                tokenizedinput = data_utils.encode_data(
                    test_df[i : i + 1], tokenizer, task_name
                )
                with torch.inference_mode():
                    logits = model(**tokenizedinput).logits
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    summed_probs.append(np.array(probs[0]))

                predicted_class_id = int(torch.argmax(logits, axis=-1)[0])
                votingDict[model.config.id2label[predicted_class_id]] += 1
            else:
                continue
        y_pred = np.array(summed_probs)
        y_pred = np.sum(summed_probs, axis=0)
        predicted_class_id = int(np.argmax(y_pred, axis=-1))
        y_preds.append(predicted_class_id)
    target_names = ["0", "1"]
    return classification_report(y_true, y_preds, target_names=target_names)
    # return precision_recall_fscore_support(y_true, y_preds, average="macro")


model_dir = "../models/"
models = load_models(model_dir)
print(len(models), "models")
# models = prune_models(models)
report = run_evaluation(models, task_name, test_df)
print(report)

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
from sklearn.metrics import classification_report
from torch.nn.utils import prune
from transformers import RobertaForSequenceClassification
from transformers import RobertaTokenizer
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
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
        tokenizer = RobertaTokenizer.from_pretrained(directory)
        model = RobertaForSequenceClassification.from_pretrained(directory,problem_type="multi_label_classification")
        print("first model - " + str(type(model)))
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
        #print(count_parameters(model), count_nonzero(model))
        new_models.append(model)
    print("models pruned!")
    return new_models



def run_evaluation(models, tokenizers,task_name, test_df):
    y_preds = []
    y_true = data_utils.extract_labels(test_df,task_name)[:5]

    for i in range(len(y_true)):
        votingDict = defaultdict(int)
        for model in models:
            if task_name == "MultiRC" or task_name == "BoolQ":
                tokenizedinput = data_utils.encode_data(test_df[i:i+1],tokenizers[0],task_name)
                print("Tokenized input \n")
                print(tokenizedinput)
                with torch.no_grad():
                    logits = model(**tokenizedinput).logits
                    print(logits)
                predicted_class_id = int(torch.argmax(logits, axis=-1)[0])
                votingDict[model.config.id2label[predicted_class_id]] += 1
            else:
                continue
        #print(votingDict)
        y_pred = max(votingDict, key=votingDict.get)
        y_preds.append(int((y_pred.replace("LABEL_1","1").replace("LABEL_0","0"))))
    target_names = ['0','1']
    print(y_preds)
    return classification_report(y_true, y_preds, target_names=target_names)
    #return precision_recall_fscore_support(y_true, y_preds, average="macro")

model_dir = "../models/"
models,tokenizers = load_models(model_dir)
#pruned_models = prune_models(models)
print(run_evaluation(models, tokenizers,task_name, test_df))
